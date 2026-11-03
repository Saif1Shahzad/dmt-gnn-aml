# Experimental protocol and implementation decisions

The source is the Phase 2 Extended Literature Review in the parent directory, especially §§10.1–10.12. This repository implements an executable research reference. A runnable pipeline does not establish scientific novelty, reported performance, or deployment readiness.

## Research questions

- RQ1: `dmt` versus `merged`, with identical seeds, hyperparameters, sampled histories and splits.
- RQ2: the same frozen checkpoint under event, batch 10/50/100, and bucket 60/600/3600 replay, at predeclared offered rates.
- RQ3: independent training and evaluation in the banking and Ethereum domains. This is two-domain validation, **not** zero-shot transfer or cross-domain fine-tuning.
- RQ4: bounded qualitative, causal deletion-replay case studies. Do not describe these as GNNExplainer or claim certified attribution fidelity.

## Causality and timestamps

All events are sorted by `(timestamp, original zero-based source row)`. Every row survives, including exact duplicates. Split cuts approximate 60/20/20 by rows and move to the end of timestamp groups. Actual counts and cutoffs are in `manifest.json`. The source row is provenance only and never a learned feature.

All events at the same timestamp are scored from the same prior state. Memory updates and insertion happen after the complete group, including in event mode. This deliberately refines the document's per-event wording to satisfy its strict `neighbor_time < query_time` invariant when IBM timestamps have minute resolution. Events sharing a timestamp cannot see one another through memory, neighbors, or counters. Large timestamp groups require memory proportional to group size during training.

Feature scaling uses exact streaming train-only moments of log1p(amount) and log1p(fee); category vocabularies are train-only, with an unknown category. This is z-score scaling, not a claim to implement quantile scaling. Full-dataset integer node IDs are identity indexes only; there is no learned ID embedding.

Memory is rebuilt with frozen epoch weights before validation and again from train+validation before test. Validation/test labels never enter memory. Validation determines early stopping, hyperparameters and threshold. The test suite uses frozen configurations; its predeclared operating conditions are repeated measurements, not opportunities to retune. An exclusive marker blocks repeated evaluation in the same output directory; users can still deliberately create a new directory, so this is an audit guard rather than a security mechanism.

## Model choices

Pure PyTorch implements the proposed operators without PyG compiled extensions. Each direction samples the latest S historical transfers (default 10), retains individual transfers, pools transfers per counterparty with learned multihead attention, then pools counterparties. Harmonic time features use learned frequencies with time deltas expressed in hours. The encoder fuses incoming/outgoing embeddings, recurrent memory, and causal counters. Edge heads consume the arriving event's attributes; node heads use only prior-history embeddings.

Current-event messages are explicitly supplied to the post-score GRU update so memory can learn transfer attributes. Multiple simultaneous messages are averaged. Truncated backpropagation spans multiple timestamp groups; memory is detached only at optimization boundaries. A gradient regression test ensures the GRU actually learns. Inference state is kept on CPU and only active vectors are moved to the chosen device.

Counters include lifetime directional counts and volumes, time since last transfer, and a **sample-bounded** recent-event count within one hour. The recent count is not an exact high-degree rolling-window count. The full prefix is not held in Python objects: events use memory-mapped arrays, history is an integer ring array, and memory is a dense CPU tensor. Model execution is a correctness-first event loop, not an optimized million-event GPU kernel.

The merged control sums **raw** sampled parallel amounts and fees before applying the shared scaler; the most recent timestamp and category represent the merged edge. The original stream, supervision targets and sampling budget are retained, so this tests loss of multiplicity within sampled neighborhoods. It does not merge transactions across prediction targets or sum amounts across the entire ledger.

## Baseline identities

| CLI variant | What it actually implements |
|---|---|
| `dmt` | Proposed hierarchical directional temporal model |
| `merged` | Same model with sampled parallel histories merged |
| `no_direction` | Shared directional weights and symmetric incoming/outgoing pooling |
| `no_memory` | No recurrent node memory; causal local history encoder remains |
| `single_stage` | Individual edges pooled without per-counterparty grouping |
| `tgn_reference` | Internal simple-history temporal memory control, latest transfer per counterparty; **not official TGN** |
| `multignn_reference` | Internal directional edge-attribute model with causal port/order scalar and no recurrent memory; **not an exact Multi-GNN reproduction** |
| `xgboost` command | XGBoost with current event attributes and causal graph counters; node task uses node counters |

The two internal controls support initial experiments but cannot substantiate superiority over the named official methods. Official [Multi-GNN](https://github.com/IBM/Multi-GNN) and [TGN](https://github.com/twitter-research/tgn) use different training/data conventions and require separate reproduction work. Optional MEGA-GNN is not bundled or misrepresented as complete. Use the common prediction schema for externally reproduced methods and match targets, splits and tuning budgets before comparison. This is a deliberate scope limitation, not a silent substitution.

## Node labels and evaluation units

XBlock requires explicit `address,label` input with labels 0, 1, or -1 (unknown). Unlisted addresses remain unknown and are excluded from supervised loss and metrics; they still carry graph messages. If the benchmark defines unflagged addresses as assumed benign, materialize that decision explicitly in the label file and disclose it in the data card. Unknown does not automatically mean benign.

Address labels are retrospective benchmark labels unless discovery timestamps exist. Results are therefore retrospective address classification, not proof of prospective detection. Training supervises labeled endpoints on each event (degree-weighted exposure). Primary evaluation takes the latest pre-event prediction per known address within each split, avoiding degree-weighted reported metrics. Addresses may recur across chronological splits; this is a transductive temporal protocol, not an unseen-address split. Error-stratum degree/multiplicity columns are event-context statistics.

## Streaming measurement

Replay uses a single-server discrete-event FIFO queue with seeded Poisson arrivals independent of model execution. Service durations are measured with device synchronization, including feature extraction, inference and state commit. File IO and diagnostic bookkeeping are excluded. Queue delay plus service duration gives simulated end-to-end latency; inference latency is separately reported. Results must be called a simulation, not wall-clock sustained production throughput.

Micro-batch mode scores each event against the currently committed state and delays updates until the batch boundary, rounded up to complete timestamp groups. Bucket mode flushes when ledger-time bucket changes. This is a model-freshness policy: it does not model Ethereum consensus/network propagation or hold predictions until block closure. Pending updates are flushed at the end. No adaptive policy silently changes a requested condition; overload appears as queue growth. The provided harness reports final peak RSS and GPU allocation; it does not claim a full resident-memory time series.

The paper requests incident detection delay, but transfer labels and typology names alone do not identify an incident onset. Output is explicitly null until incident IDs and onset times are supplied and a corresponding evaluator is added.

## Metrics and statistics

PR-AUC is explicitly average precision, not trapezoidal integration. Thresholds maximize validation F1, never test F1. P@100/500 uses available rows when fewer than k exist and reports effective k. No-positive AP is null. A two-level paired bootstrap resamples seeds then nonempty equal-duration temporal blocks; signed-rank tests operate on paired seed F1 values. Run `holm` over the entire predeclared family of baseline p-values. Cliff's delta and paired rank-biserial are both reported. Five seeds give low-powered significance tests: confidence intervals and effect sizes matter.

Diagnostics export top false positives/negatives and temporal/degree/multiplicity/typology strata. Human investigators must assign semantic error categories; the code does not invent exchange identities, laundering typologies or causal narratives. Case studies replay the entire prefix with one candidate transfer removed, so deleted history cannot survive in hidden state. They can be expensive and do not claim globally minimal explanations.

## Compute limitations and preflight

State memory is approximately `4*N*H + 16*N*S + 16*N + 40*N` bytes, before model activations, events and Python/SQLite ID maps. At 3 million nodes, H=64, S=10 this is roughly 1.42 GB for state alone. H=128 adds another 0.77 GB. Ingest is disk-backed but the identifier map remains O(nodes). Exact metrics, tabular fitting, and exported predictions require additional RAM proportional to evaluation or training rows. XGBoost's memmapped features do not eliminate its own training memory use.

Run a representative pilot and extrapolate measured runtime before starting 20 trials or 5 seeds on millions of events. This repository does not promise the document's GPU-hour budget or latency targets. The default configuration runs on CPU; set `device: cuda` for an NVIDIA environment. The Python event loop may not benefit from a GPU for tiny operations. Full IBM/XBlock runs and CUDA/MPS validation remain to be performed on the downloaded data/hardware.
