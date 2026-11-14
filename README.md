# Federated Unlearning for Neural Equalizers in 6G Wireless Communication Systems

Companion repository for the paper by Shreyas Salian, Rishabh Shukla and Om Jee Pandey (under review).

This is a **partial code release**. It ships the full dataset, every per-seed result reported in the paper, the data pipeline, the classical receivers, the three equalizer architectures, the evaluation metrics, the statistical analysis and the figure scripts. The federated training and unlearning procedures are not included yet. See [WITHHELD.md](WITHHELD.md).

## What you can reproduce with this repo

- Load any of the 9 client datasets and build the windowed equalizer inputs with the paper's frame partitions.
- Run the classical MMSE, unbiased MMSE and zero-forcing references and the utility gate (Table "classical reference").
- Build the three neural equalizers (Exp2, Exp4, Exp6) used in the paper. Exp2 has 175,296 parameters.
- Recompute every statistic in the results section, including the paired TOST equivalence tests against the Gold model, from the shipped per-seed results.
- Regenerate the results figures.

## Layout

```
src/fu_equalizer/   config, data loading, IQ imbalance, classical receivers, metrics, model architectures
analysis/           paired TOST analysis for Protocol v2
figures/            figure scripts (write PDFs to figures/out/)
data/               9 client datasets (MATLAB v7.3 / HDF5), see data/README.md
results/
  original_runs/    per-seed results of the original protocol (canary, IQ, Gold gap) and recomputed statistics
  protocol_v2/      15 confirmatory runs (summary.csv and run_meta.json per run), per-seed table, TOST margin grid
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Tested with Python 3.13 and TensorFlow 2.21 on CPU.

## Quick start

```python
from fu_equalizer import data, metrics, config, models

clients = [data.load_client(c, eval_part=config.ORIGINAL_TEST) for c in config.CLIENT_SNR_DB]
const = metrics.build_constellation(clients)
for d in clients:
    best, res = metrics.classical_reference(d, const)
    print(d["client"], best, res[best]["acc"])

model = models.build_exp2()
```

```
python analysis/tost_protocol_v2.py      # reproduces results/protocol_v2/tost_margin_grid.csv
python figures/make_result_plots.py      # method, IQ and pooling figures
python figures/make_confirm_plot.py      # Protocol v2 confirmatory figure
```

## Experimental setup in brief

- 9 clients, each a different channel condition. The forget client is `snr20`.
- OFDM with 1024 subcarriers and 256-QAM. Each input is a 9-subcarrier window of 6 features.
- Frame partitions (Protocol v2): train frames 0 to 149, validation 190 to 229, pilots 230 to 309 (seed 52 only), test 2960 to 2999. The original runs used frames 150 to 189 as their test split.
- IQ imbalance plant: 1.5 dB amplitude and 8 degrees phase.
- Utility gate: a model must beat the best classical receiver on every retained client.

## Citation

Citation details will be added once the paper is accepted.

## License

No license is granted at this time. All rights reserved by the authors. Please contact the authors before reuse.
