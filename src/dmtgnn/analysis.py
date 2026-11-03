"""Paired statistical comparisons and descriptive failure analysis."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import f1_score

from .metrics import classification, metric_rows
from .utils import write_json


def holm(pvalues):
    p = np.asarray(pvalues,dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]).clip(0,1)
    result = np.empty(len(p))
    result[order] = adjusted
    return result.tolist()


def compare(pairs,out,task='edge',resamples=1000,blocks=10,seed=42):
    """pairs = [{a: predictions.csv, b: predictions.csv, threshold_a, threshold_b}, ...]."""
    if not pairs or blocks<1 or resamples<1:
        raise ValueError('Need paired seeds and positive blocks/resamples')
    aligned,fa,fb = [],[],[]
    for pair in pairs:
        a,b = (metric_rows(pd.read_csv(pair[k]),task) for k in ('a','b'))
        keys = ['event_id','entity']
        if a.duplicated(keys).any() or b.duplicated(keys).any():
            raise ValueError('Prediction keys must be unique after node reduction')
        m = a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
        if len(m)!=len(a) or len(m)!=len(b) or not np.array_equal(m.y_a,m.y_b) or not np.array_equal(m.t_a,m.t_b):
            raise ValueError('Paired comparisons require identical targets, labels and timestamps')
        m['pa'] = m.p_a>=pair['threshold_a']
        m['pb'] = m.p_b>=pair['threshold_b']
        # Equal-duration temporal blocks shared between paired models.
        span = max(float(m.t_a.max()-m.t_a.min()),1.)
        m['block'] = np.minimum(((m.t_a-m.t_a.min())/span*blocks).astype(int),blocks-1)
        aligned.append(m)
        fa.append(f1_score(m.y_a,m.pa,zero_division=0))
        fb.append(f1_score(m.y_a,m.pb,zero_division=0))
    reference_keys = set(zip(aligned[0].event_id,aligned[0].entity))
    if any(set(zip(m.event_id,m.entity))!=reference_keys for m in aligned):
        raise ValueError('All seeds must cover the same targets')
    rng = np.random.default_rng(seed)
    boot,boot_a,boot_b = [],[],[]
    for _ in range(resamples):
        da,db = [],[]
        for s in rng.integers(len(aligned),size=len(aligned)):
            m = aligned[s]
            existing = m.block.unique()
            sampled = pd.concat([m[m.block==b] for b in rng.choice(existing,len(existing),replace=True)])
            da.append(f1_score(sampled.y_a,sampled.pa,zero_division=0))
            db.append(f1_score(sampled.y_a,sampled.pb,zero_division=0))
        boot_a.append(np.mean(da))
        boot_b.append(np.mean(db))
        boot.append(np.mean(da)-np.mean(db))
    a,b = np.array(fa),np.array(fb)
    delta = a-b
    nonzero = delta[delta!=0]
    ranks = rankdata(abs(nonzero))
    rb = float((ranks[nonzero>0].sum()-ranks[nonzero<0].sum())/ranks.sum()) if len(ranks) else 0.
    result = dict(seeds=len(pairs),f1_a_mean=float(a.mean()),f1_b_mean=float(b.mean()),
                  f1_a_std=float(a.std(ddof=1)) if len(a)>1 else 0.,f1_b_std=float(b.std(ddof=1)) if len(b)>1 else 0.,
                  f1_a_ci95=np.quantile(boot_a,[.025,.975]).tolist(),f1_b_ci95=np.quantile(boot_b,[.025,.975]).tolist(),
                  delta_f1=float(delta.mean()),delta_f1_ci95=np.quantile(boot,[.025,.975]).tolist(),
                  wilcoxon_p=float(wilcoxon(a,b,alternative='two-sided').pvalue) if len(nonzero) else 1.,
                  cliffs_delta=float(np.sign(a[:,None]-b).mean()),rank_biserial=rb,
                  note='Paired seed test; five seeds have low power. CI resamples seeds and time blocks.')
    write_json(out,result)
    return result


def diagnostics(predictions,out,threshold=.5,task='edge'):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    frame = metric_rows(pd.read_csv(predictions),task)
    frame[(frame.y==0)&(frame.p>=threshold)].nlargest(50,'p').to_csv(out/'false_positives.csv',index=False)
    frame[(frame.y==1)&(frame.p<threshold)].nsmallest(50,'p').to_csv(out/'false_negatives.csv',index=False)
    tables = []
    groups = {
        'multiplicity': pd.cut(frame.multiplicity,bins=[-1,0,1,5,20,np.inf],labels=['0','1','2-5','6-20','>20']),
        'degree':pd.cut(frame.degree,bins=[-1,10,100,1000,np.inf]),
        'time':pd.cut(frame.t,bins=min(10,max(1,frame.t.nunique()))),
        'typology':frame.typology,
    }
    for name,group in groups.items():
        for value,part in frame.groupby(group,observed=True):
            tables.append(dict(stratum=name,bucket=str(value),**classification(part.y,part.p,threshold)))
    pd.DataFrame(tables).to_csv(out/'strata.csv',index=False)
    return tables


def report(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    for path in sorted(Path(root).rglob('metrics.json')):
        r = json.loads(path.read_text())
        r['run'] = str(path.parent.relative_to(root))
        rows.append(r)
    if not rows:
        raise ValueError('No metrics.json files found')
    table = pd.DataFrame(rows)
    table.to_csv(out/'results.csv',index=False)
    if 'end_to_end_s_p99' in table:
        fig,ax = plt.subplots(figsize=(8,5))
        for mode,group in table.dropna(subset=['end_to_end_s_p99']).groupby('mode'):
            ax.scatter(group.end_to_end_s_p99*1000,group.f1,label=mode)
        ax.set(xlabel='Simulated end-to-end p99 latency (ms)',ylabel='Minority-class F1')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out/'quality_latency.png',dpi=180)
        plt.close(fig)
    return table
