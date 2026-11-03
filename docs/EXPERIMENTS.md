# Experiment guide

[← Back to the project](../README.md)

Run commands from the repository root after following the installation instructions. Use new output directories for each experiment and freeze validation-selected settings before evaluating the test set.

## Prepare IBM HI-Small

Download the original **HI-Small_Trans.csv** from the [IBM AML data release](https://github.com/IBM/AML-Data), which links the distribution. Keep it under `data/raw/ibm/`. No data are committed; the dataset has its own license, independent of code.

```bash
dmtgnn prepare --format ibm \
  --input data/raw/ibm/HI-Small_Trans.csv \
  --out data/processed/ibm
```

The parser understands the two identically named `Account` columns by position and uses `(bank, account)` identifiers. It uses Amount Paid, Payment Currency and Payment Format; the dataset supplies no transfer fee, so fee is zero. The label is `Is Laundering`. Parallel edges and negative examples are retained. Sorting happens in SQLite, followed by memory-mapped arrays.

See [DATASETS.md](DATASETS.md) for XBlock, schema mappings, typology annotations, label policies and provenance.

## Train and evaluate

```bash
dmtgnn train --data data/processed/ibm \
  --config configs/default.yaml --out runs/ibm/dmt/seed_42

dmtgnn evaluate --data data/processed/ibm \
  --checkpoint runs/ibm/dmt/seed_42/checkpoint.pt \
  --out runs/ibm/dmt/seed_42/test
```

Set `device: cuda` in a copied config on an NVIDIA machine. Start with a representative pilot to estimate runtime and memory. The default has 10 epochs, early stopping, S=10, H=64, four attention heads and seed 42. The supplied smoke config is for software verification only.

Training writes `checkpoint.pt`, validation predictions/metrics, epoch history, config and environment. Evaluation reconstructs causal memory from the preceding train+validation stream without using its labels, then writes frozen-threshold test predictions/metrics. Only load checkpoints you created or trust.

## Tune and compare

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

## Analyse and export

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

