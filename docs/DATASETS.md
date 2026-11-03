# Dataset contracts

## IBM AML

Primary source: https://github.com/IBM/AML-Data . The original source distinguishes its code license from the data's CDLA-Sharing-1.0 license. Download HI-Small from the publisher-linked distribution; do not confuse it with AMLSim or a resampled mirror.

The expected header is:

```text
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
```

No laundering typology is inferred from `Is Laundering`. The optional publisher Patterns.txt requires format-aware annotation matching; this repository does not guess its row numbering or reconstruct patterns from labels. To evaluate typologies, create a canonical CSV export including an authoritative `typology` column (one row per original transaction, unchanged order), and ingest via `--format csv`. Keep the provenance of the join. Multiple incidents/typologies per transaction require an explicit evaluation policy. Without annotations, the empty typology group is not a laundering-pattern result.

## XBlock Ethereum

Publisher: https://www.xblock.pro/ . The XBlock Ethereum Phishing Transaction Network is also distributed by the XBlock account on Kaggle. The exact download schema must be checked after acquisition; different XBlock releases are not interchangeable. Do not substitute ETGraph or XBlock-ETH bulk chain data for the benchmark without documenting it.

The code accepts a directed transaction CSV and a separate label registry. It does not auto-download an arbitrary archive or execute remote code.

Canonical transaction CSV:

```csv
src,dst,timestamp,amount,fee,category
0xabc,0xdef,1609459200,1.25,0.002,transfer
```

- Required: `src,dst,timestamp,amount`.
- Optional: `fee` (0), `category` (`unknown`), edge `label` (-1), `typology` (empty).
- Node labels must **not** be transaction feature columns.
- Timestamps must be UTC strings or numeric with an explicit `--time-unit s|ms|us|ns`.
- Amount units must be consistent. Convert wei to ETH if intended and record the conversion; the converter has an explicit divisor. Never mix unconverted token units without an appropriate category/model policy.
- Source rows are retained as zero-based `event_id`; source ID hashes may be recorded externally, but this implementation uses row identity to retain duplicate events.

Labels CSV:

```csv
address,label
0xabc,1
0xdef,0
0xunknown,-1
```

Unlisted addresses remain -1. Explicit 0 means a benchmark negative, not proof that the address is benign. The train and validation portions need usable labels of both classes for meaningful model selection. Keep any label-discovery timestamps in the research data card; this implementation uses retrospective static labels, not a label-availability simulation.

Canonical CSV preparation:

```bash
dmtgnn prepare --format xblock --time-unit s \
  --input data/raw/xblock/transactions.csv \
  --labels data/raw/xblock/labels.csv --out data/processed/xblock
```

If the CSV uses names such as `from_address,to_address,value`, edit `configs/xblock_columns.yaml`, then add `--mapping configs/xblock_columns.yaml`.

For a **trusted publisher NetworkX multigraph pickle**, the included converter preserves every edge:

```bash
python scripts/convert_xblock_graph.py \
  --input data/raw/xblock/graph.pickle --out data/raw/xblock/export \
  --trust-pickle --time-key timestamp --amount-key amount --label-key isp
```

Use actual attribute names from the archive documentation; missing keys raise an actionable error. `--amount-divisor 1e18` is appropriate only for wei input. The converter rejects simple or undirected graphs because missing multiplicity/direction cannot be reconstructed. Loading pickle executes code: the trust flag is intentionally explicit. NetworkX is installed with PyTorch in the tested environment.

Then run the same `train`, `evaluate`, `suite`, `replay`, and `xgboost` commands using `data/processed/xblock`. The prepared manifest selects the node head automatically. Add `--task node` to diagnostics/statistical comparisons.

## Prepared artifact

- `events.npy`: structured, memory-mapped events ordered by timestamp and source row.
- `manifest.json`: raw/label SHA-256 hashes, feature transforms, category and typology dictionaries, split boundaries, task type and dimensions.
- `index.sqlite`: original account-name mapping and disk-sorted staging tables. For identity lookup: `SELECT name FROM nodes WHERE id = ?`.

Treat a prepared directory as immutable. Use a new directory to change raw data, labels, mapping or preprocessing. Checkpoints reject a different manifest. Preserve raw source files and archive version metadata separately. Data and checkpoints remain gitignored.

## Toy fixture

`dmtgnn synthetic` generates independent data solely for software verification. It has no empirical relationship to IBM AML or Ethereum and must never be used for the manuscript's claims.
