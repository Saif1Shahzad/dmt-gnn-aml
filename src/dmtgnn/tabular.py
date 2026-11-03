"""XGBoost over current event features and strictly causal running graph statistics."""
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .data import Dataset
from .engine import COLUMNS
from .metrics import metric_rows, summarize, threshold_on_validation
from .model import StreamState
from .utils import write_json


def tabular_features(data,refresh=1):
    if refresh<1:
        raise ValueError('refresh must be positive')
    state = StreamState(data.meta['nodes'],1,10)
    pending = []
    for group in data.groups():
        for i in group:
            e = data.events[i]
            u,v,t = int(e['src']),int(e['dst']),float(e['t'])
            su,sv = state.node_features(u,t,data),state.node_features(v,t,data)
            if data.meta['task']=='edge':
                yield np.r_[data.features(e),su,sv],int(e['y']),[int(e['id']),t,-1,int(e['y']),0.,int(e['typology']),int(state.count[u].sum()+state.count[v].sum()),0],i
            else:
                for node,stats,y in ((u,su,int(e['ys'])),(v,sv,int(e['yd']))):
                    # Node predictions do not consume the current edge's attributes.
                    yield stats,y,[int(e['id']),t,node,y,0.,int(e['typology']),int(state.count[node].sum()),0],i
            pending.append(i)
        if len(pending)>=refresh:
            for i in pending:
                state.insert(i,data.events[i])
            pending.clear()


def train_xgboost(data_path,out,seed=42,refresh=1,evaluate_test=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    data = Dataset(data_path)
    a,b = data.meta['train_end'],data.meta['val_end']
    n = len(data.events)*(2 if data.meta['task']=='node' else 1)
    dim = 6 if data.meta['task']=='node' else data.feature_dim+12
    x = np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='float32',shape=(n,dim))
    ys,indices = np.empty(n,dtype=np.int8),np.empty(n,dtype=np.int64)
    records = []
    for j,(features,y,row,i) in enumerate(tabular_features(data,refresh)):
        x[j],ys[j],indices[j] = features,y,i
        records.append(row)
    train_mask = (indices<a)&(ys>=0)
    val_mask = (indices>=a)&(indices<b)&(ys>=0)
    if len(np.unique(ys[train_mask]))!=2:
        raise ValueError('XGBoost requires both training classes')
    positive = int(ys[train_mask].sum())
    model = XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,tree_method='hist',
                          random_state=seed,n_jobs=1,scale_pos_weight=(train_mask.sum()-positive)/positive,
                          eval_metric='aucpr',early_stopping_rounds=20)
    model.fit(x[train_mask],ys[train_mask],eval_set=[(x[val_mask],ys[val_mask])],verbose=False)
    model.save_model(out/'model.json')
    frame = pd.DataFrame(records,columns=COLUMNS)
    val_rows = (indices>=a)&(indices<b)
    val = frame[val_rows].copy()
    val['p'] = model.predict_proba(x[val_rows])[:,1]
    known = metric_rows(val,data.meta['task'])
    threshold = threshold_on_validation(known.y.to_numpy(),known.p.to_numpy())
    val.to_csv(out/'validation_predictions.csv',index=False)
    write_json(out/'validation_metrics.json',summarize(val,data.meta['task'],threshold))
    write_json(out/'config.json',dict(seed=seed,refresh=refresh,threshold=threshold,manifest=data.meta))
    if evaluate_test:
        test_rows = indices>=b
        test = frame[test_rows].copy()
        test['p'] = model.predict_proba(x[test_rows])[:,1]
        test.to_csv(out/'predictions.csv',index=False)
        write_json(out/'metrics.json',summarize(test,data.meta['task'],threshold))


def replay_xgboost(data_path,model_dir,out,rate=100.,seed=42):
    """Replay the frozen tree model using its predeclared graph-feature refresh interval."""
    import json
    import time
    from .streaming import fifo
    if rate<=0:
        raise ValueError('rate must be positive')
    out,model_dir = Path(out),Path(model_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    cfg = json.loads((model_dir/'config.json').read_text())
    data = Dataset(data_path)
    if cfg['manifest']!=data.meta:
        raise ValueError('Tree model and dataset manifests differ')
    model = XGBClassifier()
    model.load_model(model_dir/'model.json')
    state = StreamState(data.meta['nodes'],1,10)
    start = data.meta['val_end']
    pending,records,logs = [],[],[]
    last_commit = float(data.events[0]['t'])
    for group in data.groups():
        for i in group:
            tic = time.perf_counter()
            e = data.events[i]
            u,v,t = int(e['src']),int(e['dst']),float(e['t'])
            if i>=start:
                su,sv = state.node_features(u,t,data),state.node_features(v,t,data)
                if data.meta['task']=='edge':
                    x = np.array([np.r_[data.features(e),su,sv]])
                    targets = [(int(e['y']),-1)]
                else:
                    x = np.array([su,sv])
                    targets = [(int(e['ys']),u),(int(e['yd']),v)]
                probabilities = model.predict_proba(x)[:,1]
                cost = time.perf_counter()-tic
                for p,(y,entity) in zip(probabilities,targets):
                    records.append([int(e['id']),t,entity,y,float(p),int(e['typology']),int(state.count[u].sum()+state.count[v].sum()),0])
                logs.append(dict(event_id=int(e['id']),t=t,service_s=cost,score_s=cost,
                                 pending_events=len(pending),staleness_s=max(0,t-last_commit)))
            pending.append(i)
        if len(pending)>=cfg['refresh'] or group.stop==len(data.events):
            tic = time.perf_counter()
            for j in pending:
                state.insert(j,data.events[j])
            last_commit = float(data.events[pending[-1]]['t'])
            pending.clear()
            if group.stop>start:
                logs[-1]['service_s'] += time.perf_counter()-tic
    frame,log = pd.DataFrame(records,columns=COLUMNS),pd.DataFrame(logs)
    arrivals = np.cumsum(np.random.default_rng(seed).exponential(1/rate,len(log)))
    arrivals -= arrivals[0]
    delays,ends = fifo(arrivals,log.service_s.to_numpy())
    log['arrival_s'],log['queue_s'],log['completion_s'] = arrivals,delays,ends
    log['end_to_end_s'] = ends-arrivals
    log['queue_depth'] = np.arange(len(log))-np.searchsorted(ends,arrivals,side='right')
    frame.to_csv(out/'predictions.csv',index=False)
    log.to_csv(out/'latency.csv',index=False)
    result = summarize(frame,data.meta['task'],cfg['threshold'])
    result.update(mode='xgboost_refresh',refresh=cfg['refresh'],offered_rate=rate,
                  measured_service_capacity=len(log)/log.service_s.sum(),simulated_throughput=len(log)/ends[-1],
                  max_queue=int(log.queue_depth.max()),mean_staleness_s=float(log.staleness_s.mean()),
                  queue_model='single-server FIFO discrete-event simulation; Poisson arrivals')
    for name in ('score_s','service_s','queue_s','end_to_end_s'):
        for q in (50,95,99):
            result[f'{name}_p{q}'] = float(np.percentile(log[name],q))
    write_json(out/'metrics.json',result)
    return result
