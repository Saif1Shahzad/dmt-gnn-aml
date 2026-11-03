"""Disk-backed ingestion. IDs and labels are never model input features."""
import csv
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import sha256, write_json

DTYPE = np.dtype([('id', '<i8'), ('src', '<i8'), ('dst', '<i8'), ('t', '<f8'),
                  ('amount', '<f8'), ('fee', '<f8'), ('category', '<i4'),
                  ('y', 'i1'), ('ys', 'i1'), ('yd', 'i1'), ('typology', '<i4')])
IBM_HEADER = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account',
              'Amount Received', 'Receiving Currency', 'Amount Paid',
              'Payment Currency', 'Payment Format', 'Is Laundering']


def timestamp(value, unit=None):
    if unit:
        factors = {'s': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}
        return float(value) * factors[unit]
    return pd.Timestamp(value, tz='UTC').timestamp()


def label(value):
    v = int(value)
    if v not in (-1, 0, 1):
        raise ValueError(f'Expected label -1, 0 or 1, got {value}')
    return v


def rows(path, fmt, mapping, time_unit):
    """Preserve duplicate transfers; IBM's repeated Account header is positional."""
    with open(path, newline='') as f:
        if fmt == 'ibm':
            reader = csv.reader(f)
            if next(reader) != IBM_HEADER:
                raise ValueError('IBM CSV header mismatch; expected original HI-Small_Trans.csv schema')
            for i, r in enumerate(reader):
                yield (i, json.dumps([r[1], r[2]]), json.dumps([r[3], r[4]]),
                       timestamp(r[0]), float(r[7]), 0., r[8] + '|' + r[9], label(r[10]), '')
        else:
            reader = csv.DictReader(f)
            for key in ('src', 'dst', 'timestamp', 'amount'):
                if mapping[key] not in (reader.fieldnames or []):
                    raise ValueError(f'Missing column {mapping[key]!r}; provide --mapping YAML')
            for i, r in enumerate(reader):
                def get(k, default):
                    return r.get(mapping.get(k, ''), default)
                yield (i, r[mapping['src']], r[mapping['dst']],
                       timestamp(r[mapping['timestamp']], time_unit),
                       float(r[mapping['amount']]), float(get('fee', 0)),
                       str(get('category', 'unknown')), label(get('label', -1)), str(get('typology', '')))


def prepare(raw, out, fmt='ibm', mapping=None, time_unit=None, labels=None,
            train_fraction=.6, val_fraction=.2):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} must be empty; datasets are immutable')
    out.mkdir(parents=True, exist_ok=True)
    if not 0 < train_fraction < train_fraction + val_fraction < 1:
        raise ValueError('Invalid split fractions')
    mapping = mapping or dict(src='src', dst='dst', timestamp='timestamp', amount='amount',
                              fee='fee', category='category', label='label', typology='typology')
    db = sqlite3.connect(out / 'index.sqlite')
    db.execute('CREATE TABLE nodes (name TEXT PRIMARY KEY, id INTEGER UNIQUE, y INTEGER DEFAULT -1)')
    db.execute('CREATE TABLE events (id INTEGER, src INTEGER, dst INTEGER, t REAL, amount REAL, fee REAL, cat TEXT, y INTEGER, typo TEXT)')
    node_cache = {}  # IDs only: ~O(number of accounts), never holds the full event table.
    def node(name):
        if not name:
            raise ValueError('Empty account identifier')
        if name not in node_cache:
            node_cache[name] = len(node_cache)
            db.execute('INSERT INTO nodes(name,id) VALUES (?,?)', (name, node_cache[name]))
        return node_cache[name]
    n = 0
    for i, u, v, t, a, fee, cat, y, typo in rows(raw, fmt, mapping, time_unit):
        if not all(math.isfinite(x) for x in (t, a, fee)) or a < 0 or fee < 0:
            raise ValueError(f'Nonfinite timestamp or invalid amount/fee at source row {i}')
        db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)', (i, node(u), node(v), t, a, fee, cat, y, typo))
        n += 1
        if n % 100000 == 0:
            db.commit()
    if n < 5:
        raise ValueError('Need at least five events and three distinct timestamps')
    if labels:
        with open(labels, newline='') as f:
            seen = set()
            for r in csv.DictReader(f):
                name, y = r['address'], label(r['label'])
                if name in seen:
                    raise ValueError(f'Duplicate address label: {name}')
                seen.add(name)
                db.execute('UPDATE nodes SET y=? WHERE name=?', (y, name))
    if fmt == 'xblock' and not labels:
        raise ValueError('XBlock requires --labels CSV with address,label; unknown addresses remain -1')
    db.commit()
    db.execute('CREATE INDEX chronological ON events(t,id)')
    # Cut after entire timestamp groups; ratios are approximate, strict causality is exact.
    t1 = db.execute('SELECT t FROM events ORDER BY t,id LIMIT 1 OFFSET ?', (int(n*train_fraction)-1,)).fetchone()[0]
    t2 = db.execute('SELECT t FROM events ORDER BY t,id LIMIT 1 OFFSET ?', (int(n*(train_fraction+val_fraction))-1,)).fetchone()[0]
    ntrain = db.execute('SELECT count(*) FROM events WHERE t<=?', (t1,)).fetchone()[0]
    nvalend = db.execute('SELECT count(*) FROM events WHERE t<=?', (t2,)).fetchone()[0]
    if not 0 < ntrain < nvalend < n:
        raise ValueError('Timestamp ties make a split empty; use a longer dataset')
    cats = {r[0]: j+1 for j, r in enumerate(db.execute('SELECT DISTINCT cat FROM events WHERE t<=? ORDER BY cat', (t1,)))}
    typos = {r[0]: j for j, r in enumerate(db.execute('SELECT DISTINCT typo FROM events ORDER BY typo'))}
    node_y = np.full(len(node_cache), -1, dtype=np.int8)
    for idx, y in db.execute('SELECT id,y FROM nodes'):
        node_y[idx] = y
    arr = np.lib.format.open_memmap(out/'events.npy', mode='w+', dtype=DTYPE, shape=(n,))
    for j, r in enumerate(db.execute('SELECT * FROM events ORDER BY t,id')):
        i,u,v,t,a,f,c,y,ty = r
        arr[j] = (i,u,v,t,a,f,cats.get(c,0),y,node_y[u],node_y[v],typos[ty])
    arr.flush()
    # Bounded-memory train-only moments of log1p amount and fee.
    total, squared = np.zeros(2), np.zeros(2)
    for start in range(0,ntrain,100000):
        block = arr[start:min(start+100000,ntrain)]
        x = np.column_stack([np.log1p(block['amount']),np.log1p(block['fee'])])
        total += x.sum(0)
        squared += (x*x).sum(0)
    mean = total/ntrain
    variance = np.maximum(squared/ntrain-mean**2,0.)
    # Constant features get unit scale, avoiding huge values for merged fees.
    std = np.where(variance>1e-12,np.sqrt(variance),1.)
    meta = dict(format=fmt, task='node' if fmt=='xblock' else 'edge', count=n,
                nodes=len(node_cache), train_end=ntrain, val_end=nvalend,
                train_max_time=t1, val_max_time=t2, categories=cats, typologies=typos,
                mean=mean.tolist(), std=std.tolist(), source_sha256=sha256(raw),
                labels_sha256=sha256(labels) if labels else None, mapping=mapping,
                timestamp_unit=time_unit, split_policy='60/20/20 by rows, ties kept together')
    write_json(out/'manifest.json',meta)
    db.close()
    return meta


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path/'manifest.json').read_text())
        self.events = np.load(self.path/'events.npy', mmap_mode='r')
        self.feature_dim = 3 + len(self.meta['categories']) + 1
        self.mean, self.std = np.array(self.meta['mean']), np.array(self.meta['std'])
        a,b = self.meta['train_end'],self.meta['val_end']
        if not (self.events[a-1]['t'] < self.events[a]['t'] and self.events[b-1]['t'] < self.events[b]['t']):
            raise ValueError('Temporal leakage at split boundary')

    def features(self, event):
        x = np.zeros(self.feature_dim, dtype=np.float32)
        x[:2] = (np.log1p([event['amount'],event['fee']])-self.mean)/self.std
        x[2] = np.sign(event['amount'])
        x[3+int(event['category'])] = 1
        return x

    def bounds(self, split):
        a,b = self.meta['train_end'],self.meta['val_end']
        return {'train':(0,a),'val':(a,b),'test':(b,len(self.events))}[split]

    def groups(self, start=0, end=None):
        end = len(self.events) if end is None else end
        i = start
        while i < end:
            j = i+1
            while j < end and self.events[j]['t'] == self.events[i]['t']:
                j += 1
            yield range(i,j)
            i = j
