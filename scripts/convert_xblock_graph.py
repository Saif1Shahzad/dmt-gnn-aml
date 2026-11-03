#!/usr/bin/env python3
"""Convert a trusted NetworkX directed graph pickle to the canonical CSV inputs.

Pickle executes Python code when opened. Only use an archive obtained from a trusted
publisher; this script requires an explicit --trust-pickle flag. CSV ingestion does not.
"""
import argparse
import csv
import pickle
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--trust-pickle',action='store_true')
    p.add_argument('--time-key',default='timestamp')
    p.add_argument('--amount-key',default='amount')
    p.add_argument('--label-key',default='isp')
    p.add_argument('--amount-divisor',type=float,default=1.,help='Use 1e18 only if source values are wei')
    a = p.parse_args()
    if not a.trust_pickle:
        p.error('Only trusted publisher archives may be loaded; pass --trust-pickle to acknowledge pickle execution')
    if a.amount_divisor<=0:
        p.error('amount-divisor must be positive')
    with open(a.input,'rb') as f:
        graph = pickle.load(f)
    if not graph.is_directed():
        raise ValueError('An undirected graph cannot recover original transfer directions')
    if not graph.is_multigraph():
        raise ValueError('A simple graph has lost parallel edges; obtain the transaction multigraph')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'transactions.csv').open('w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['src','dst','timestamp','amount'])
        for u,v,_,attrs in graph.edges(keys=True,data=True):
            if a.time_key not in attrs or a.amount_key not in attrs:
                raise ValueError(f'Missing edge attribute; available keys: {list(attrs)}. Supply --time-key / --amount-key')
            w.writerow([u,v,attrs[a.time_key],float(attrs[a.amount_key])/a.amount_divisor])
    with (out/'labels.csv').open('w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['address','label'])
        for node,attrs in graph.nodes(data=True):
            y = attrs.get(a.label_key,-1)
            if int(y) not in (-1,0,1):
                raise ValueError(f'Invalid node label {y}')
            w.writerow([node,int(y)])
    print(f'Exported {graph.number_of_edges()} transfers and {graph.number_of_nodes()} addresses to {out}')


if __name__=='__main__':
    main()
