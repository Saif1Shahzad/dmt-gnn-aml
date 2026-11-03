#!/usr/bin/env python3
"""Compare a completed neural suite against DMT across identical seeds."""
import argparse
import json
from pathlib import Path

from dmtgnn.analysis import compare, holm, report
from dmtgnn.utils import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--task',choices=['edge','node'],default='edge')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,101,202,303,404])
    p.add_argument('--resamples',type=int,default=1000)
    a = p.parse_args()
    root,out = Path(a.suite),Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    results = []
    for variant in sorted(root.iterdir()):
        if not variant.is_dir() or variant.name=='dmt' or not (variant/f'seed_{a.seeds[0]}'/'test/predictions.csv').exists():
            continue
        pairs = []
        for seed in a.seeds:
            ra,rb = root/'dmt'/f'seed_{seed}',variant/f'seed_{seed}'
            ta = json.loads((ra/'validation_metrics.json').read_text())['threshold']
            tb = json.loads((rb/'validation_metrics.json').read_text())['threshold']
            pairs.append(dict(a=str(ra/'test/predictions.csv'),b=str(rb/'test/predictions.csv'),threshold_a=ta,threshold_b=tb))
        result = compare(pairs,out/f'{variant.name}.json',a.task,a.resamples)
        result['baseline'] = variant.name
        results.append(result)
    if not results:
        raise ValueError('No complete paired suite runs found')
    for result,pvalue in zip(results,holm([r['wilcoxon_p'] for r in results])):
        result['holm_p'] = pvalue
    write_json(out/'comparisons.json',results)
    report(root,out)


if __name__=='__main__':
    main()
