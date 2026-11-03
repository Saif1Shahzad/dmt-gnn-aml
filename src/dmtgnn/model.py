"""Pure PyTorch implementation; no compiled graph extensions required."""
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class StreamState:
    def __init__(self, nodes, hidden, budget):
        if budget < 1:
            raise ValueError('sampling budget must be positive')
        self.memory = torch.zeros(nodes, hidden)  # CPU: only active nodes move to GPU.
        self.live = {}
        self.history = np.full((nodes,2,budget), -1, dtype=np.int64)
        self.count = np.zeros((nodes,2), dtype=np.int64)
        self.stats = np.zeros((nodes,5), dtype=np.float64)
        self.stats[:,4] = -np.inf
        self.budget = budget
        self.version = 0

    def get(self, node, device):
        return self.live.get(node, self.memory[node].to(device))

    def detach(self):
        with torch.no_grad():
            for n,m in self.live.items():
                self.memory[n].copy_(m.detach().cpu())
        self.live.clear()

    def neighbors(self, node, direction, t, data):
        ids = self.history[node,direction]
        ids = ids[ids>=0]
        ids = ids[data.events[ids]['t'] < t]
        return sorted(ids.tolist(),key=lambda i:(data.events[i]['t'],i))

    def insert(self, index, event):
        u,v,t,a = int(event['src']),int(event['dst']),float(event['t']),float(event['amount'])
        for node,d in ((u,1),(v,0)):
            pos = self.count[node,d] % self.budget
            self.history[node,d,pos] = index
            self.count[node,d] += 1
        self.stats[u,1] += 1
        self.stats[v,0] += 1
        self.stats[u,3] += a
        self.stats[v,2] += a
        self.stats[u,4] = self.stats[v,4] = t
        self.version += 1

    def node_features(self, node, t, data):
        s = self.stats[node]
        recent = sum(data.events[i]['t'] >= t-3600 for d in (0,1)
                     for i in self.neighbors(node,d,t,data))
        gap = max(0,t-s[4]) if np.isfinite(s[4]) else 0
        return np.log1p(np.r_[s[:4],gap,recent]).astype(np.float32)


class HarmonicTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.frequency = nn.Parameter(torch.logspace(-5,0,dim//2))

    def forward(self, delta):
        phase = (delta.clamp(min=0)/3600.).unsqueeze(-1)*self.frequency
        return torch.cat([phase.cos(),phase.sin()],dim=-1)


class AttentionPool(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        if heads < 1 or hidden < 1 or hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        self.heads = heads
        self.score = nn.Linear(hidden,heads,bias=False)

    def forward(self, values):
        weights = self.score(values).softmax(dim=0)
        chunks = values.reshape(len(values),self.heads,-1)
        return (weights.unsqueeze(-1)*chunks).sum(0).reshape(-1)


class DMTGNN(nn.Module):
    def __init__(self, feature_dim, hidden=64, time_dim=16, dropout=.1, heads=4,
                 variant='dmt', task='edge'):
        super().__init__()
        if variant not in ('dmt','merged','no_direction','no_memory','single_stage','tgn_reference','multignn_reference'):
            raise ValueError(f'Unknown model variant: {variant}')
        if time_dim < 2 or time_dim % 2:
            raise ValueError('time_dim must be positive and even')
        self.hidden,self.variant,self.task = hidden,variant,task
        self.time = HarmonicTime(time_dim)
        # Extra scalar is a causal port/order feature for the internal Multi-GNN control.
        self.edge = nn.ModuleList([nn.Linear(feature_dim+time_dim+1,hidden) for _ in range(2)])
        self.edge_pool = nn.ModuleList([AttentionPool(hidden,heads) for _ in range(2)])
        self.peer = nn.ModuleList([nn.Linear(2*hidden,hidden) for _ in range(2)])
        self.peer_pool = nn.ModuleList([AttentionPool(hidden,heads) for _ in range(2)])
        self.fuse = nn.Sequential(nn.Linear(3*hidden+6,hidden),nn.ReLU(),nn.Dropout(dropout))
        self.message = nn.Linear(2*hidden+feature_dim,hidden)
        self.gru = nn.GRUCell(hidden,hidden)
        self.edge_head = nn.Sequential(nn.Linear(2*hidden+feature_dim,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        self.node_head = nn.Linear(hidden,1)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def uses_memory(self):
        return self.variant not in ('no_memory','multignn_reference')

    def aggregate(self, node, direction, t, data, state):
        d = 0 if self.variant=='no_direction' else direction
        ids = state.neighbors(node,direction,t,data)
        if not ids:
            return torch.zeros(self.hidden,device=self.device)
        by_peer = defaultdict(list)
        for i in ids:
            e = data.events[i]
            peer = int(e['src'] if direction==0 else e['dst'])
            by_peer[peer].append(i)
        if self.variant in ('single_stage','tgn_reference'):
            # TGN reference retains latest event per peer (simple historical graph).
            groups = [[v[-1]] for v in by_peer.values()] if self.variant=='tgn_reference' else [[i] for i in ids]
        else:
            groups = list(by_peer.values())
        values = []
        for port,group in enumerate(groups):
            events = [data.events[i] for i in group]
            if self.variant=='merged':
                # Sum RAW values before scaling, within the identical sampled window.
                e = events[-1].copy()
                e['amount'] = sum(float(x['amount']) for x in events)
                e['fee'] = sum(float(x['fee']) for x in events)
                events = [e]
            x = torch.as_tensor(np.stack([data.features(e) for e in events]),device=self.device)
            delta = torch.tensor([t-float(e['t']) for e in events],device=self.device,dtype=torch.float32)
            port_value = np.log1p(port+1) if self.variant=='multignn_reference' else 0.
            ports = torch.full((len(events),1),float(port_value),device=self.device)
            edge = F.relu(self.edge[d](torch.cat([x,self.time(delta),ports],-1)))
            pooled = self.edge_pool[d](edge)
            e = events[-1]
            peer = int(e['src'] if direction==0 else e['dst'])
            mem = state.get(peer,self.device) if self.uses_memory else torch.zeros_like(pooled)
            values.append(F.relu(self.peer[d](torch.cat([mem,pooled]))))
        return self.peer_pool[d](torch.stack(values))

    def encode(self, node, t, data, state):
        incoming = self.aggregate(node,0,t,data,state)
        outgoing = self.aggregate(node,1,t,data,state)
        if self.variant=='no_direction':
            incoming = outgoing = (incoming+outgoing)/2
        mem = state.get(node,self.device) if self.uses_memory else torch.zeros_like(incoming)
        stats = torch.as_tensor(state.node_features(node,t,data),device=self.device)
        if self.variant=='no_direction':
            stats = stats.clone()
            stats[0:2] = stats[0:2].mean()
            stats[2:4] = stats[2:4].mean()
        return self.fuse(torch.cat([mem,incoming,outgoing,stats]))

    def score(self, index, data, state):
        e = data.events[index]
        u,v,t = int(e['src']),int(e['dst']),float(e['t'])
        zu,zv = self.encode(u,t,data,state),self.encode(v,t,data,state)
        x = torch.as_tensor(data.features(e),device=self.device)
        if self.task=='edge':
            logits = self.edge_head(torch.cat([zu,zv,x])).reshape(1)
            labels,entities = [int(e['y'])],[-1]
        else:
            logits = torch.stack([self.node_head(zu).squeeze(),self.node_head(zv).squeeze()])
            labels,entities = [int(e['ys']),int(e['yd'])],[u,v]
        return logits,labels,entities,(u,v,zu,zv,x)

    def update(self, tokens, indices, data, state):
        """All tokens were scored before this call; equal timestamps are simultaneous."""
        messages = defaultdict(list)
        if self.uses_memory:
            for u,v,zu,zv,x in tokens:
                messages[u].append(F.relu(self.message(torch.cat([zu,zv,x]))))
                messages[v].append(F.relu(self.message(torch.cat([zv,zu,x]))))
            new = {n:self.gru(torch.stack(ms).mean(0),state.get(n,self.device)) for n,ms in messages.items()}
            state.live.update(new)
        for i in indices:
            state.insert(i,data.events[i])


def focal_loss(logits, targets, alpha=.75, gamma=2., bce=False):
    base = F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    if bce:
        return base.mean()
    pt = torch.exp(-base)
    weight = torch.where(targets==1,alpha,1-alpha)
    return (weight*(1-pt).pow(gamma)*base).mean()
