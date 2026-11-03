import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import Dataset
from .metrics import metric_rows, summarize, threshold_on_validation
from .model import DMTGNN, StreamState, focal_loss
from .utils import seed_all, write_json

COLUMNS = ['event_id','t','entity','y','p','typology','degree','multiplicity']


def make_model(data,cfg):
    m = cfg.get('model',{})
    return DMTGNN(data.feature_dim,task=data.meta['task'],**m).to(cfg.get('device','cpu'))


def make_state(data,model,cfg):
    return StreamState(data.meta['nodes'],model.hidden,cfg.get('sampling_budget',10))


def predictions_for(i,logits,labels,entities,data,state):
    e = data.events[i]
    u,v = int(e['src']),int(e['dst'])
    hist = state.neighbors(u,1,float(e['t']),data)
    multiplicity = sum(int(data.events[j]['dst'])==v for j in hist)
    degree = int(state.count[u].sum()+state.count[v].sum())
    return [[int(e['id']),float(e['t']),entity,y,float(p),int(e['typology']),degree,multiplicity]
            for p,y,entity in zip(logits.detach().sigmoid().cpu().tolist(),labels,entities)]


def run_range(model,data,state,start,end,train=False,optimizer=None,cfg=None,collect=True):
    cfg = cfg or {}
    model.train(train)
    rows,losses = [],[]
    seen,total_loss,steps = 0,0.,0
    with torch.set_grad_enabled(train):
        for group in data.groups(start,end):
            tokens = []
            for i in group:
                logits,labels,entities,token = model.score(i,data,state)
                tokens.append(token)
                if collect:
                    rows.extend(predictions_for(i,logits,labels,entities,data,state))
                valid = np.array(labels)>=0
                if train and valid.any():
                    mask = torch.as_tensor(valid,device=model.device)
                    targets = torch.tensor(labels,device=model.device,dtype=torch.float32)[mask]
                    losses.append(focal_loss(logits[mask],targets,cfg.get('alpha',.75),cfg.get('gamma',2.),cfg.get('loss','focal')=='bce'))
            model.update(tokens,group,data,state)
            seen += len(group)
            # Keep GRU gradients through several timestamp groups, then truncate.
            if seen >= cfg.get('batch_size',200) or group.stop==end:
                if train and losses:
                    loss = torch.stack(losses).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get('grad_clip',1.))
                    optimizer.step()
                    total_loss += float(loss.detach())
                    steps += 1
                losses.clear()
                state.detach()
                seen = 0
    return pd.DataFrame(rows,columns=COLUMNS),total_loss/max(steps,1)


def train(data_path,cfg,out):
    data,out = Dataset(data_path),Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Run directory {out} is not empty')
    out.mkdir(parents=True,exist_ok=True)
    seed_all(cfg.get('seed',42))
    torch.set_num_threads(cfg.get('threads',1))
    model = make_model(data,cfg)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.get('lr',.001),weight_decay=cfg.get('weight_decay',1e-5))
    best,stale,logs = -1.,0,[]
    a,b = data.meta['train_end'],data.meta['val_end']
    # Fail early rather than train on entirely unknown labels.
    key = 'y' if data.meta['task']=='edge' else 'ys'
    if not np.any(data.events[:a][key]>=0) and not (data.meta['task']=='node' and np.any(data.events[:a]['yd']>=0)):
        raise ValueError('Training partition has no known labels')
    for epoch in range(cfg.get('epochs',10)):
        start = time.perf_counter()
        state = make_state(data,model,cfg)
        _,loss = run_range(model,data,state,0,a,True,optimizer,cfg,False)
        # Rebuild memory with final epoch weights; stale training memory is not reused.
        state = make_state(data,model,cfg)
        run_range(model,data,state,0,a,cfg=cfg,collect=False)
        val,_ = run_range(model,data,state,a,b,cfg=cfg)
        rows = metric_rows(val,data.meta['task'])
        threshold = threshold_on_validation(rows.y.to_numpy(),rows.p.to_numpy())
        metrics = summarize(val,data.meta['task'],threshold)
        score = metrics['pr_auc']
        logs.append(dict(epoch=epoch+1,loss=loss,seconds=time.perf_counter()-start,**metrics))
        print(json.dumps(logs[-1]),flush=True)
        if score > best:
            best,stale = score,0
            torch.save(dict(model=model.state_dict(),config=cfg,threshold=threshold,
                            manifest=data.meta,feature_dim=data.feature_dim,epoch=epoch+1),out/'checkpoint.pt')
            val.to_csv(out/'validation_predictions.csv',index=False)
            write_json(out/'validation_metrics.json',metrics)
        else:
            stale += 1
        if stale >= cfg.get('patience',3):
            break
    write_json(out/'history.json',logs)
    write_json(out/'config.json',cfg)
    write_json(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
                                         numpy=np.__version__,platform=platform.platform(),device=str(model.device)))
    return best


def load_checkpoint(data_path,checkpoint):
    data = Dataset(data_path)
    ck = torch.load(checkpoint,map_location='cpu',weights_only=False)  # Load only your own artifacts.
    if ck['manifest'] != data.meta:
        raise ValueError('Checkpoint dataset manifest differs from prepared dataset')
    model = make_model(data,ck['config'])
    model.load_state_dict(ck['model'])
    model.eval()
    return data,model,ck


def evaluate(data_path,checkpoint,out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    # Exclusive marker prevents accidental repeated test evaluation in the same output.
    marker = out/'evaluation_started.json'
    with marker.open('x') as f:
        json.dump({'checkpoint':str(checkpoint),'policy':'frozen configuration; no test tuning'},f)
    data,model,ck = load_checkpoint(data_path,checkpoint)
    state = make_state(data,model,ck['config'])
    b = data.meta['val_end']
    run_range(model,data,state,0,b,cfg=ck['config'],collect=False)
    frame,_ = run_range(model,data,state,b,len(data.events),cfg=ck['config'])
    frame.to_csv(out/'predictions.csv',index=False)
    metrics = summarize(frame,data.meta['task'],ck['threshold'])
    write_json(out/'metrics.json',metrics)
    write_json(out/'evaluation.json',dict(task=data.meta['task'],threshold=ck['threshold'],config=ck['config']))
    return metrics
