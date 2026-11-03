"""Causal leave-one-event-out replay; descriptive evidence, not GNNExplainer."""
from pathlib import Path

import torch

from .engine import load_checkpoint, make_state, run_range
from .utils import write_json


def explain(data_path,checkpoint,event_id,out,max_candidates=20,entity=None):
    data,model,ck = load_checkpoint(data_path,checkpoint)
    matches = (data.events['id']==event_id).nonzero()[0]
    if len(matches)!=1:
        raise ValueError('event_id must identify exactly one source row')
    target = int(matches[0])
    if target<data.meta['val_end']:
        raise ValueError('Case studies must use held-out test events')
    event = data.events[target]
    group_start = target
    while group_start>0 and data.events[group_start-1]['t']==event['t']:
        group_start-=1
    state = make_state(data,model,ck['config'])
    run_range(model,data,state,0,group_start,cfg=ck['config'],collect=False)
    with torch.no_grad():
        logits,labels,entities,_ = model.score(target,data,state)
        pos = 0 if model.task=='edge' else entities.index(entity)
        probability = float(logits[pos].sigmoid())
        if labels[pos]!=1 or probability<ck['threshold']:
            raise ValueError('Forensic case study target must be a verified true-positive alert')
        candidates = sorted({i for n in (int(event['src']),int(event['dst'])) for d in (0,1)
                             for i in state.neighbors(n,d,float(event['t']),data)},reverse=True)[:max_candidates]
        results = []
        for omitted in candidates:
            altered = make_state(data,model,ck['config'])
            # Recompute the whole prefix so the deleted event cannot persist in memory/counters.
            for group in data.groups(0,group_start):
                indices = [i for i in group if i!=omitted]
                tokens = [model.score(i,data,altered)[3] for i in indices]
                model.update(tokens,indices,data,altered)
                altered.detach()
            p = float(model.score(target,data,altered)[0][pos].sigmoid())
            e = data.events[omitted]
            results.append(dict(removed_event_id=int(e['id']),src=int(e['src']),dst=int(e['dst']),
                                timestamp=float(e['t']),probability_without=p,probability_drop=probability-p))
    result = dict(event_id=event_id,entity=entity,probability=probability,threshold=ck['threshold'],
                  method='causal prefix replay with one event deleted',candidates=sorted(results,key=lambda r:-r['probability_drop']),
                  limitation='Local candidate set; single-event deletion effects are not additive and do not prove real-world causation.')
    write_json(out,result)
    return result
