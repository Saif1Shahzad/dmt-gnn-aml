import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, matthews_corrcoef)


def metric_rows(frame, task):
    known = frame[frame.y>=0].copy()
    # One latest PRE-EVENT prediction per address per split, not degree-weighted edge labels.
    if task=='node':
        known = known.sort_values(['t','event_id']).drop_duplicates('entity',keep='last')
    return known


def threshold_on_validation(y, p):
    if len(np.unique(y)) != 2:
        raise ValueError('Validation must contain positive and negative labels to calibrate threshold')
    precision,recall,thresholds = precision_recall_curve(y,p)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(thresholds[np.argmax(f1)])


def classification(y,p,threshold=.5):
    y,p = np.asarray(y),np.asarray(p)
    if len(y)==0:
        return {'n':0,'positive':0,'pr_auc':None,'f1':None}
    pred = p>=threshold
    result = dict(n=len(y),positive=int(y.sum()),pr_auc=float(average_precision_score(y,p)) if y.sum()>0 else None,
                  precision=float(precision_score(y,pred,zero_division=0)),
                  recall=float(recall_score(y,pred,zero_division=0)),
                  f1=float(f1_score(y,pred,zero_division=0)),mcc=float(matthews_corrcoef(y,pred)),threshold=float(threshold))
    for k in (100,500):
        result[f'precision_at_{k}'] = float(y[np.argsort(-p,kind='stable')[:k]].mean())
        result[f'precision_at_{k}_effective_k'] = min(k,len(y))
    return result


def summarize(frame,task,threshold):
    rows = metric_rows(frame,task)
    result = classification(rows.y.to_numpy(),rows.p.to_numpy(),threshold)
    result['pr_auc_definition'] = 'average precision (step integral)'
    return result
