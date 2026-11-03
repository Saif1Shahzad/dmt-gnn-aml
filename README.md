# DMT-GNN AML

**Directed Multigraph Temporal Graph Neural Networks for streaming anti-money-laundering research.**

An executable reference implementation of the Phase 2 research plan by Saif Shahzad and Shahoud Shahid. It supports IBM AML transaction classification and XBlock Ethereum address classification, with strict temporal causality, repeatable experiments and measured streaming simulation.

## Included

- Disk-backed CSV preprocessing, immutable manifests and SHA-256 source provenance.
- Train-only feature transforms, chronological 60/20/20 splits, preserved parallel transfers, unknown-label masks.
- Hierarchical directional attention, harmonic time encoding, GRU memory and focal loss.
- Merged-edge, no-direction, no-memory, single-stage and BCE controls; configurable sampling budgets.
- XGBoost with causal graph features; explicitly named internal TGN/Multi-GNN reference controls.
- Validation-only Optuna search, five-seed suites, checkpointing and validation-calibrated thresholds.
- Event/micro-batch/time-bucket replay with Poisson offered load, FIFO queue simulation and latency percentiles.
- Minority F1, precision/recall, average precision, MCC, P@k, paired block bootstrap, Wilcoxon, Holm and effect sizes.
- Error/stratum exports, quality–latency plots, and causal event-deletion case studies.
- Causality/gradient/integration tests, CI, Docker, and toy data for checking the pipeline before downloading anything.

**Read [the protocol](docs/PROTOCOL.md) before interpreting results.** The internal TGN/Multi-GNN controls are not official reproductions; optional MEGA-GNN is not integrated. The event loop is a correctness-first reference and has not been benchmarked on full IBM/XBlock datasets. Real-data training results are not bundled.

## 1. Install and verify

Use Python 3.12+ (3.12 is the CI/Docker target). On macOS, install the XGBoost runtime once with `brew install libomp`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
source scripts/activate.sh
python -m pytest -q
bash scripts/smoke.sh
```

In the original development checkout, a `.venv` is already installed; fresh clones should follow the installation commands above. `source scripts/activate.sh` activates it, keeps plotting caches local, and selects PyTorch's bundled OpenMP on macOS to avoid mixing incompatible runtimes. The CLI applies the same selection before importing numerical libraries. Use `python -m pytest`, not a shell-launched pytest executable, on macOS. A system runtime can also be installed with `brew install libomp`. Linux needs its OpenMP runtime (included in the Docker image).

The smoke workflow creates independent toy transfers, trains, evaluates, replays, and writes `runs/smoke/report/results.csv` and `quality_latency.png`. Toy metrics are only a software test. Output directories must be new to avoid silently overwriting experiments; use `bash scripts/smoke.sh runs/smoke2` to repeat.

`requirements-tested.txt` records the exact dependencies tested on this Mac, not a universal cross-platform lock. `pyproject.toml` defines supported dependency ranges; every run records its configuration and key runtime versions. CUDA PyTorch builds should be installed for the target GPU environment before installing this project.

## 2. Download and prepare IBM HI-Small

Download the original **HI-Small_Trans.csv** from the [IBM AML data release](https://github.com/IBM/AML-Data), which links the distribution. Keep it under `data/raw/ibm/`. No data are committed; the dataset has its own license, independent of code.

```bash
dmtgnn prepare --format ibm \
  --input data/raw/ibm/HI-Small_Trans.csv \
  --out data/processed/ibm
```

The parser understands the two identically named `Account` columns by position and uses `(bank, account)` identifiers. It uses Amount Paid, Payment Currency and Payment Format; the dataset supplies no transfer fee, so fee is zero. The label is `Is Laundering`. Parallel edges and negative examples are retained. Sorting happens in SQLite, followed by memory-mapped arrays.

See [DATASETS.md](docs/DATASETS.md) for XBlock, schema mappings, typology annotations, label policies and provenance.

## 3. Train, then evaluate

```bash
dmtgnn train --data data/processed/ibm \
  --config configs/default.yaml --out runs/ibm/dmt/seed_42

dmtgnn evaluate --data data/processed/ibm \
  --checkpoint runs/ibm/dmt/seed_42/checkpoint.pt \
  --out runs/ibm/dmt/seed_42/test
```

Set `device: cuda` in a copied config on an NVIDIA machine. Start with a representative pilot to estimate runtime and memory. The default has 10 epochs, early stopping, S=10, H=64, four attention heads and seed 42. The supplied smoke config is for software verification only.

Training writes `checkpoint.pt`, validation predictions/metrics, epoch history, config and environment. Evaluation reconstructs causal memory from the preceding train+validation stream without using its labels, then writes frozen-threshold test predictions/metrics. Only load checkpoints you created or trust.

## 4. Tune and run the planned comparisons

```bash
dmtgnn tune --data data/processed/ibm --config configs/default.yaml \
  --trials 20 --out runs/ibm/tuning

dmtgnn suite --data data/processed/ibm \
  --config runs/ibm/tuning/best_config.json --out runs/ibm/suite
```

The suite trains every configured variant at seeds 42, 101, 202, 303 and 404, then evaluates frozen checkpoints. Ablations share the chosen configuration. For fair independently tuned baseline comparisons, tune each model separately by setting `model.variant` in its config; do not claim identical tuning budgets if only DMT was tuned. No test results enter Optuna.

Add `--streaming` to run the predeclared event/batch/bucket/rate matrix for every seed and model. This is a large workload; run a pilot first. To run one condition:

```bash
dmtgnn replay --data data/processed/ibm \
  --checkpoint runs/ibm/dmt/seed_42/checkpoint.pt \
  --mode batch --batch-size 50 --rate 100 \
  --out runs/ibm/dmt/seed_42/stream_batch50_rate100
```

For time buckets use `--mode bucket --bucket-seconds 600`; event mode is `--mode event`. Equal-timestamp events are always scored together before updates. Inference latency and simulated end-to-end latency are distinct outputs. `detection_delay` is null without incident onset annotations.

XGBoost uses the same prepared inputs:

```bash
dmtgnn xgboost --data data/processed/ibm --refresh 100 --seed 42 \
  --evaluate-test --out runs/ibm/xgboost_r100/seed_42
```

Repeat at refresh 100/1000/10000 and the five seeds as predeclared. Use `dmtgnn replay-xgboost --data data/processed/ibm --model-dir runs/ibm/xgboost_r100/seed_42 --rate 100 --out runs/ibm/xgboost_r100/seed_42/replay` for queued systems replay with the same frozen refresh policy. The XGBoost feature cache is disk-backed, but training still needs RAM.

## 5. Analyse and export

For a completed suite, `python scripts/analyze_suite.py --suite runs/ibm/suite --out runs/ibm/comparisons` automatically pairs the five seeds, reads their validation thresholds, computes bootstrap comparisons, and applies Holm correction across baselines.

```bash
dmtgnn report --runs runs/ibm --out runs/ibm/report

dmtgnn diagnostics --predictions runs/ibm/dmt/seed_42/test/predictions.csv \
  --threshold 0.5 --out runs/ibm/dmt/seed_42/diagnostics
```

Replace `0.5` with the frozen threshold in that run's `validation_metrics.json`. For address classification add `--task node`.

Create a JSON list with one pair per seed:

```json
[
  {"a": "runs/dmt/seed_42/test/predictions.csv",
   "b": "runs/merged/seed_42/test/predictions.csv",
   "threshold_a": 0.47, "threshold_b": 0.38}
]
```

The paths and thresholds above illustrate the schema; use actual paths and validation thresholds, and include all five seeds.

```bash
dmtgnn compare --pairs pairs.json --resamples 1000 --blocks 10 --out comparison.json
dmtgnn holm --pvalues 0.031 0.12 0.25
```

Use actual paired-test p-values in the Holm command. Comparisons reject mismatched target IDs, timestamps and labels. Reports contain CSV tables and a quality/latency scatter plot if replay runs exist.

For a true-positive test alert, `dmtgnn explain --data ... --checkpoint ... --event-id SOURCE_ROW_ID --out explanation.json` replays the prefix with each sampled candidate event removed. Node tasks also require `--entity INTEGER_NODE_ID`. This is an expensive qualitative diagnostic, not GNNExplainer.

## Repository map

```text
src/dmtgnn/
  data.py          CSV adapters, train-only transforms, disk-backed graph
  model.py         causal state, hierarchical attention, GRU, ablations
  engine.py        train/validate/test lifecycle and checkpoints
  streaming.py     measured-service FIFO simulation and freshness policies
  tabular.py       XGBoost graph-feature control
  metrics.py       label masks, address reduction, classification metrics
  experiments.py   Optuna and fixed-seed experiment matrix
  analysis.py      bootstrap, significance tests, diagnostics, plots
  explain.py       causal deletion-replay case studies
  synthetic.py     independent toy fixtures
  cli.py           command-line interface
configs/           default, smoke and XBlock mappings
scripts/           smoke workflow, local activation, XBlock graph conversion
tests/            causal and end-to-end regression checks
docs/             protocol, dataset contract, requirement traceability
```

Public repository: [Saif1Shahzad/dmt-gnn-aml](https://github.com/Saif1Shahzad/dmt-gnn-aml). Dataset files, virtual environments, model artifacts and results are ignored by Git.
