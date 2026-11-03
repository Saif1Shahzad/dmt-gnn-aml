# Local verification

Date: November 3, 2026. Host: macOS ARM64, CPU. Real IBM/XBlock datasets were not present.

- `source scripts/activate.sh; python -m pytest -q`: **18 passed**.
- End-to-end smoke script: CSV preparation, two training epochs, checkpoint reload, held-out evaluation, micro-batch replay, and CSV/PNG report completed.
- Integration tests cover edge and node tasks, strict chronological boundaries, constant-feature scaling, IBM duplicate column names, multigraph retention, no score mutation, strict past neighbors, future-feature/label isolation, recurrent gradient propagation, all seven neural variants, replay equivalence in event mode, FIFO timing, Holm correction and paired bootstrap.
- XGBoost fit and replay tested at refresh intervals 1 and 5; replay predictions matched the corresponding frozen-model test predictions. Actual console commands also completed after the macOS OpenMP launcher fix.
- One Optuna trial completed with validation-only optimization and a saved best configuration.
- A two-seed, two-variant suite completed, followed by automatic paired statistics/report generation.
- A true-positive synthetic case study completed with two causal prefix-deletion replays.
- A generated quality/latency plot was visually checked. It is a toy software-verification artifact, not a research result.

All generated verification artifacts are under gitignored `runs/`. The exact installed Python packages are recorded in `requirements-tested.txt`.

GitHub Actions subsequently passed on published commit `218d5be748de343a5e39fadee9691cfd8db59f11`: [causal-pipeline-tests run](https://github.com/Saif1Shahzad/dmt-gnn-aml/actions/runs/41573751287). The workflow installs dependencies on Ubuntu with Python 3.12, runs the regression suite, and executes the smoke workflow.

Not verified here: full-scale datasets, official baseline reproductions, GPU/MPS execution, Docker build, or any claimed scientific improvement. See PROTOCOL.md for explicit departures from the proposed methodology.
