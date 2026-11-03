import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from dmtgnn.data import Dataset, prepare, IBM_HEADER
from dmtgnn.engine import train, evaluate, make_model, make_state, run_range
from dmtgnn.model import focal_loss
from dmtgnn.synthetic import generate
from dmtgnn.streaming import fifo, replay
from dmtgnn.analysis import holm, compare
from dmtgnn.metrics import metric_rows


@pytest.fixture
def data(tmp_path):
    raw = generate(tmp_path/'raw',60)
    prepare(raw,tmp_path/'data','csv',time_unit='s')
    return Dataset(tmp_path/'data')


def config():
    return dict(seed=42,epochs=1,batch_size=12,sampling_budget=5,threads=1,
                model=dict(hidden=8,time_dim=4,heads=2,dropout=0.))


def test_split_and_scaler(data):
    a,b = data.meta['train_end'],data.meta['val_end']
    assert data.events[a-1]['t']<data.events[a]['t']
    assert data.events[b-1]['t']<data.events[b]['t']
    assert np.allclose(data.mean[0],np.log1p(data.events[:a]['amount']).mean())
    assert len(data.events)==60
    assert data.std[1]==1.0  # Constant fee must not amplify merged-edge amounts.


def test_duplicate_ibm_accounts_and_parallel_edges(tmp_path):
    import csv
    path = tmp_path/'ibm.csv'
    with path.open('w',newline='') as f:
        w = csv.writer(f)
        w.writerow(IBM_HEADER)
        for i in range(10):
            w.writerow([f'2022/09/01 00:{i:02d}',1,'same',2,'same',10,'USD',10,'USD','ACH',i%2])
    prepare(path,tmp_path/'prepared')
    data = Dataset(tmp_path/'prepared')
    assert data.meta['nodes']==2
    assert len(data.events)==10
    assert all(data.events['src']!=data.events['dst'])


def test_score_does_not_update_and_no_equal_time_neighbors(data):
    model = make_model(data,config()).eval()
    state = make_state(data,model,config())
    before = state.memory.clone()
    _,_,_,token = model.score(0,data,state)
    assert state.version==0
    assert torch.equal(before,state.memory)
    model.update([token],[0],data,state)
    e = data.events[0]
    assert state.neighbors(int(e['src']),1,e['t'],data)==[]
    assert state.neighbors(int(e['src']),1,e['t']+1,data)==[0]


def test_gru_receives_gradients(data):
    model = make_model(data,config())
    state = make_state(data,model,config())
    group = next(data.groups())
    tokens = [model.score(i,data,state)[3] for i in group]
    model.update(tokens,group,data,state)
    # Score a synthetic later query of a touched endpoint to ensure recurrent path is used.
    e = data.events[0]
    z = model.encode(int(e['src']),float(e['t'])+120,data,state)
    model.node_head(z).sum().backward()
    assert model.gru.weight_ih.grad is not None
    assert float(model.gru.weight_ih.grad.abs().sum())>0


def test_future_features_and_labels_do_not_change_past(data,tmp_path):
    model = make_model(data,config()).eval()
    state = make_state(data,model,config())
    run_range(model,data,state,0,10,collect=False)
    original = model.score(10,data,state)[0].detach()
    # Change every future amount and every label, preserving the train-fitted transform.
    altered = deepcopy(data)
    altered.events = np.array(data.events)
    altered.events[11:]['amount'] = 1e12
    altered.events['y'] = 1-altered.events['y']
    changed = model.score(10,altered,state)[0].detach()
    assert torch.equal(original,changed)


@pytest.mark.parametrize('variant',['dmt','merged','no_direction','no_memory','single_stage','tgn_reference','multignn_reference'])
def test_variants_backward(data,variant):
    cfg = config()
    cfg['model']['variant'] = variant
    model = make_model(data,cfg)
    state = make_state(data,model,cfg)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    _,loss = run_range(model,data,state,0,20,True,optimizer,cfg,False)
    assert np.isfinite(loss)


def test_train_evaluate_replay_roundtrip(data,tmp_path):
    cfg = config()
    train(data.path,cfg,tmp_path/'model')
    ck = tmp_path/'model/checkpoint.pt'
    metrics = evaluate(data.path,ck,tmp_path/'test')
    assert metrics['n']==len(data.events)-data.meta['val_end']
    with pytest.raises(FileExistsError):
        evaluate(data.path,ck,tmp_path/'test')
    for mode in ('event','batch','bucket'):
        r = replay(data.path,ck,tmp_path/mode,mode,batch_size=5,bucket_seconds=120)
        assert r['end_to_end_s_p99']>=r['score_s_p50']
        assert r['n']==metrics['n']
    event_predictions = pd.read_csv(tmp_path/'event/predictions.csv')
    exact_predictions = pd.read_csv(tmp_path/'test/predictions.csv')
    assert np.allclose(event_predictions.p,exact_predictions.p)


def test_node_task_and_unknowns(tmp_path):
    raw = generate(tmp_path/'raw',60)
    prepare(raw,tmp_path/'data','xblock',time_unit='s',labels=tmp_path/'raw/labels.csv')
    data = Dataset(tmp_path/'data')
    assert data.meta['task']=='node'
    train(data.path,config(),tmp_path/'model')
    result = evaluate(data.path,tmp_path/'model/checkpoint.pt',tmp_path/'test')
    assert result['n']<=data.meta['nodes']
    rows = metric_rows(pd.read_csv(tmp_path/'test/predictions.csv'),'node')
    assert rows.entity.is_unique


def test_fifo_and_holm():
    delay,end = fifo([0,.1,.2],[.2,.2,.2])
    assert np.allclose(delay,[0,.1,.2])
    assert np.allclose(end,[.2,.4,.6])
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])


def test_paired_bootstrap(tmp_path):
    a = pd.DataFrame(dict(event_id=range(20),entity=-1,t=range(20),y=[0,1]*10,p=[.1,.9]*10))
    b = a.copy()
    b.p = .2
    a.to_csv(tmp_path/'a.csv',index=False)
    b.to_csv(tmp_path/'b.csv',index=False)
    pair = dict(a=str(tmp_path/'a.csv'),b=str(tmp_path/'b.csv'),threshold_a=.5,threshold_b=.5)
    r = compare([pair]*5,tmp_path/'comparison.json',resamples=10,blocks=2)
    assert r['delta_f1']==1
    assert r['delta_f1_ci95']==[1.,1.]


@pytest.mark.parametrize('refresh',[1,5])
def test_xgboost(data,tmp_path,refresh):
    from dmtgnn.tabular import train_xgboost, replay_xgboost
    train_xgboost(data.path,tmp_path/'xgb',refresh=refresh,evaluate_test=True)
    assert (tmp_path/'xgb/metrics.json').exists()
    replay_xgboost(data.path,tmp_path/'xgb',tmp_path/'xgb_replay')
    a = pd.read_csv(tmp_path/'xgb/predictions.csv')
    b = pd.read_csv(tmp_path/'xgb_replay/predictions.csv')
    assert np.allclose(a.p,b.p)
