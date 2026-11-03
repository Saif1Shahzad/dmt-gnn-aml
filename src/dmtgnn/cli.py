import argparse
import json
from pathlib import Path

from .utils import load_config


def main():
    p = argparse.ArgumentParser(description='DMT-GNN reproducible AML research runner')
    sub = p.add_subparsers(dest='command',required=True)
    s = sub.add_parser('synthetic',help='Generate toy data for pipeline verification')
    s.add_argument('--out',required=True)
    s.add_argument('--events',type=int,default=200)
    s.add_argument('--seed',type=int,default=42)
    s = sub.add_parser('prepare')
    s.add_argument('--input',required=True)
    s.add_argument('--out',required=True)
    s.add_argument('--format',choices=['ibm','csv','xblock'],default='ibm')
    s.add_argument('--mapping')
    s.add_argument('--time-unit',choices=['s','ms','us','ns'])
    s.add_argument('--labels')
    for name in ('train','tune','suite'):
        s = sub.add_parser(name)
        s.add_argument('--data',required=True)
        s.add_argument('--config',required=True)
        s.add_argument('--out',required=True)
        if name=='tune': s.add_argument('--trials',type=int,default=20)
        if name=='suite': s.add_argument('--streaming',action='store_true')
    for name in ('evaluate','replay','explain'):
        s = sub.add_parser(name)
        s.add_argument('--data',required=True)
        s.add_argument('--checkpoint',required=True)
        s.add_argument('--out',required=True)
        if name=='replay':
            s.add_argument('--mode',choices=['event','batch','bucket'],default='event')
            s.add_argument('--batch-size',type=int,default=50)
            s.add_argument('--bucket-seconds',type=float,default=600.)
            s.add_argument('--rate',type=float,default=100.)
            s.add_argument('--seed',type=int,default=42)
        if name=='explain':
            s.add_argument('--event-id',type=int,required=True)
            s.add_argument('--entity',type=int)
            s.add_argument('--max-candidates',type=int,default=20)
    s = sub.add_parser('xgboost')
    s.add_argument('--data',required=True)
    s.add_argument('--out',required=True)
    s.add_argument('--refresh',type=int,default=1)
    s.add_argument('--seed',type=int,default=42)
    s.add_argument('--evaluate-test',action='store_true')
    s = sub.add_parser('replay-xgboost')
    s.add_argument('--data',required=True)
    s.add_argument('--model-dir',required=True)
    s.add_argument('--out',required=True)
    s.add_argument('--rate',type=float,default=100.)
    s.add_argument('--seed',type=int,default=42)
    s = sub.add_parser('compare')
    s.add_argument('--pairs',required=True,help='JSON list of paired prediction paths and frozen thresholds')
    s.add_argument('--out',required=True)
    s.add_argument('--task',choices=['edge','node'],default='edge')
    s.add_argument('--resamples',type=int,default=1000)
    s.add_argument('--blocks',type=int,default=10)
    s = sub.add_parser('holm')
    s.add_argument('--pvalues',type=float,nargs='+',required=True)
    s = sub.add_parser('diagnostics')
    s.add_argument('--predictions',required=True)
    s.add_argument('--out',required=True)
    s.add_argument('--threshold',type=float,required=True)
    s.add_argument('--task',choices=['edge','node'],default='edge')
    s = sub.add_parser('report')
    s.add_argument('--runs',required=True)
    s.add_argument('--out',required=True)
    a = p.parse_args()
    result = None
    if a.command=='synthetic':
        from .synthetic import generate
        result = str(generate(a.out,a.events,a.seed))
    elif a.command=='prepare':
        from .data import prepare
        result = prepare(a.input,a.out,a.format,load_config(a.mapping) if a.mapping else None,a.time_unit,a.labels)
    elif a.command=='train':
        from .engine import train
        result = train(a.data,load_config(a.config),a.out)
    elif a.command=='tune':
        from .experiments import tune
        result = tune(a.data,load_config(a.config),a.out,a.trials)
    elif a.command=='suite':
        from .experiments import suite
        suite(a.data,load_config(a.config),a.out,a.streaming)
    elif a.command=='evaluate':
        from .engine import evaluate
        result = evaluate(a.data,a.checkpoint,a.out)
    elif a.command=='replay':
        from .streaming import replay
        result = replay(a.data,a.checkpoint,a.out,a.mode,a.batch_size,a.bucket_seconds,a.rate,a.seed)
    elif a.command=='xgboost':
        from .tabular import train_xgboost
        train_xgboost(a.data,a.out,a.seed,a.refresh,a.evaluate_test)
    elif a.command=='compare':
        from .analysis import compare
        result = compare(json.loads(Path(a.pairs).read_text()),a.out,a.task,a.resamples,a.blocks)
    elif a.command=='replay-xgboost':
        from .tabular import replay_xgboost
        result = replay_xgboost(a.data,a.model_dir,a.out,a.rate,a.seed)
    elif a.command=='holm':
        from .analysis import holm
        result = holm(a.pvalues)
    elif a.command=='diagnostics':
        from .analysis import diagnostics
        diagnostics(a.predictions,a.out,a.threshold,a.task)
    elif a.command=='report':
        from .analysis import report
        report(a.runs,a.out)
    elif a.command=='explain':
        from .explain import explain
        result = explain(a.data,a.checkpoint,a.event_id,a.out,a.max_candidates,a.entity)
    if result is not None:
        print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
