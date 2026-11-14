# What is not in this release yet

This repo is a partial release. The following parts of the experiment code are held back for now and will be published with or after the paper's acceptance:

- Federated training loop (FedAvg orchestration, client sampling and aggregation).
- Unlearning procedures: FedEraser calibration, capped gradient ascent with validation guard and rollback, and retain fine-tuning.
- Canary plant used for the memorization audit.
- Run orchestration (queue and lane scripts, per-run drivers).

Everything needed to check the reported numbers is shipped: the raw data, every per-seed result and run metadata, and the analysis that turns them into the paper's tables and figures.

`results/protocol_v2/FREEZE_sha256.txt` lists SHA-256 hashes of the withheld scripts as they were frozen before the confirmatory runs, so the code released later can be checked against what actually produced these results.
