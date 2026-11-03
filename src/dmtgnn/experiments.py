from copy import deepcopy
from pathlib import Path

import optuna

from .engine import evaluate, train
from .streaming import replay
from .utils import write_json


def tune(data,cfg,out,trials=20):
    if not 1<=trials<=20:
        raise ValueError('Protocol limits tuning to 1–20 validation-only trials')
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    def objective(trial):
        c = deepcopy(cfg)
        c['lr'] = trial.suggest_float('lr',1e-4,1e-2,log=True)
        c['gamma'] = trial.suggest_float('gamma',1.,3.)
        c.setdefault('model',{}).update(hidden=trial.suggest_categorical('hidden',[64,128]),
            dropout=trial.suggest_float('dropout',.1,.4),heads=trial.suggest_categorical('heads',[4,8]))
        return train(data,c,out/f'trial_{trial.number:02d}')
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=cfg.get('seed',42)),
                                storage=f'sqlite:///{out.resolve() / "study.sqlite"}',study_name='validation',load_if_exists=False)
    study.optimize(objective,n_trials=trials)
    best = deepcopy(cfg)
    best.update({k:study.best_params[k] for k in ('lr','gamma')})
    best.setdefault('model',{}).update({k:study.best_params[k] for k in ('hidden','dropout','heads')})
    write_json(out/'best_config.json',best)
    return best


def suite(data,cfg,out,with_streaming=False):
    """Frozen-config confirmatory runs. No tuning decisions based on suite test outputs."""
    out = Path(out)
    for variant in cfg.get('variants',['dmt','merged','no_direction','no_memory','tgn_reference','multignn_reference']):
        for seed in cfg.get('seeds',[42,101,202,303,404]):
            c = deepcopy(cfg)
            c['seed'] = seed
            c.setdefault('model',{})['variant'] = variant
            run = out/variant/f'seed_{seed}'
            train(data,c,run)
            evaluate(data,run/'checkpoint.pt',run/'test')
            if with_streaming:
                for rate in cfg.get('offered_rates',[10,100,1000]):
                    specs = [('event',1,60)]+[('batch',b,60) for b in (10,50,100)]+[('bucket',1,d) for d in (60,600,3600)]
                    for mode,b,d in specs:
                        replay(data,run/'checkpoint.pt',run/'stream'/f'{mode}_b{b}_d{d}_r{rate}',mode,b,d,rate,seed)
