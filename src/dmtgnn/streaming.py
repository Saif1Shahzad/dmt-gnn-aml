"""Open-loop discrete-event queue driven by measured service times (no wall-clock sleeps)."""
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .engine import COLUMNS, load_checkpoint, make_state, predictions_for, run_range
from .metrics import summarize
from .utils import sync, write_json


def fifo(arrivals,service):
    finish = 0.
    delays,ends = [],[]
    for arrival,cost in zip(arrivals,service):
        begin = max(float(arrival),finish)
        delays.append(begin-arrival)
        finish = begin+cost
        ends.append(finish)
    return np.array(delays),np.array(ends)


def replay(data_path,checkpoint,out,mode='event',batch_size=50,bucket_seconds=600.,rate=100.,seed=42):
    if rate<=0 or batch_size<1 or bucket_seconds<=0 or mode not in ('event','batch','bucket'):
        raise ValueError('Invalid streaming configuration')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    data,model,ck = load_checkpoint(data_path,checkpoint)
    state = make_state(data,model,ck['config'])
    start = data.meta['val_end']
    run_range(model,data,state,0,start,cfg=ck['config'],collect=False)
    pending_tokens,pending_ids,rows,logs = [],[],[],[]
    bucket = None
    last_commit = float(data.events[start-1]['t'])
    def commit():
        nonlocal last_commit
        if pending_ids:
            model.update(pending_tokens,pending_ids,data,state)
            last_commit = float(data.events[pending_ids[-1]]['t'])
            pending_tokens.clear()
            pending_ids.clear()
            state.detach()
    if str(model.device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for group in data.groups(start):
            t = float(data.events[group.start]['t'])
            current_bucket = int((t-float(data.events[start]['t']))//bucket_seconds)
            sync(model.device)
            before = time.perf_counter()
            if mode=='bucket' and bucket is not None and bucket!=current_bucket:
                commit()
            bucket = current_bucket
            sync(model.device)
            pre_cost = time.perf_counter()-before
            for i in group:
                sync(model.device)
                tic = time.perf_counter()
                logits,labels,entities,token = model.score(i,data,state)
                sync(model.device)
                cost = time.perf_counter()-tic
                rows.extend(predictions_for(i,logits,labels,entities,data,state))
                logs.append(dict(event_id=int(data.events[i]['id']),t=t,service_s=cost+(pre_cost if i==group.start else 0),
                                 score_s=cost,pending_events=len(pending_ids),staleness_s=max(0,t-last_commit)))
                pending_tokens.append(token)
                pending_ids.append(i)
            if mode=='event' or (mode=='batch' and len(pending_ids)>=batch_size) or group.stop==len(data.events):
                sync(model.device)
                tic = time.perf_counter()
                commit()
                sync(model.device)
                logs[-1]['service_s'] += time.perf_counter()-tic
    log = pd.DataFrame(logs)
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1/rate,len(log)))
    arrivals -= arrivals[0]
    delays,ends = fifo(arrivals,log.service_s.to_numpy())
    log['arrival_s'],log['queue_s'],log['completion_s'] = arrivals,delays,ends
    log['end_to_end_s'] = ends-arrivals
    # Outstanding events at each arrival, using monotone service completion timestamps.
    log['queue_depth'] = np.arange(len(log))-np.searchsorted(ends,arrivals,side='right')
    log['rss_peak_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    frame = pd.DataFrame(rows,columns=COLUMNS)
    frame.to_csv(out/'predictions.csv',index=False)
    log.to_csv(out/'latency.csv',index=False)
    metrics = summarize(frame,data.meta['task'],ck['threshold'])
    metrics.update(mode=mode,batch_size=batch_size,bucket_seconds=bucket_seconds,offered_rate=rate,
                   measured_service_capacity=len(log)/log.service_s.sum(),
                   simulated_throughput=len(log)/ends[-1],max_queue=int(log.queue_depth.max()),
                   mean_staleness_s=float(log.staleness_s.mean()),
                   gpu_peak_bytes=int(torch.cuda.max_memory_allocated()) if str(model.device).startswith('cuda') else None,
                   rss_peak_bytes=int(log.rss_peak_bytes.max()),
                   timing_scope='measured feature extraction, model score and state commit; excludes disk IO and metric logging',
                   queue_model='single-server FIFO discrete-event simulation; Poisson arrivals',
                   detection_delay=None,detection_delay_reason='Requires incident onset IDs/times, absent from canonical inputs')
    for name in ('score_s','service_s','queue_s','end_to_end_s'):
        for q in (50,95,99):
            metrics[f'{name}_p{q}'] = float(np.percentile(log[name],q))
    write_json(out/'metrics.json',metrics)
    return metrics
