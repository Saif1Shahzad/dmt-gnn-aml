"""Independent toy fixture, not IBM/XBlock data and not a scientific benchmark."""
import csv
from pathlib import Path
import numpy as np


def generate(out,events=200,seed=42):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(seed)
    with (out/'transactions.csv').open('w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['src','dst','timestamp','amount','fee','category','label','typology'])
        for i in range(events):
            # Repeat counterparties and timestamp ties, with both classes in every split.
            bad = int(i%7==0)
            u,v = (0,1) if bad else tuple(rng.integers(0,12,2))
            w.writerow([f'a{u}',f'a{v}',1700000000+(i//2)*60,round(float(rng.lognormal(4,1)),2),.1,'transfer',bad,'cycle' if bad else ''])
    with (out/'labels.csv').open('w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['address','label'])
        for i in range(12):
            w.writerow([f'a{i}',int(i==0)])
    return out/'transactions.csv'
