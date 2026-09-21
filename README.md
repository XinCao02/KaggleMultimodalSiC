# ResMLP-Ridge for CITE-seq RNA-to-protein prediction

Predict 140 protein measurements from 22,050 RNA features in the
[Kaggle Open Problems multimodal single-cell competition](https://www.kaggle.com/competitions/open-problems-multimodal).
The implementation combines a compact residual MLP ensemble with ridge regression,
and compares it with an adaptation of the public first-place SS-OPM architecture.

## Code map

| File | Purpose |
|---|---|
| `src/citebench/models.py` | ResMLP, residual blocks, SS-OPM adaptation and losses |
| `src/citebench/features.py` | Training-only PCA, supervised gene selection and RNA summaries |
| `src/citebench/data.py` | HDF5 streaming, identifier alignment and evaluation splits |
| `src/citebench/train.py` | MPS training, checkpoint selection, inference and diagnostics |
| `src/citebench/metrics.py` | Cell/protein correlations and paired cluster bootstrap |
| `scripts/run_benchmark.py` | Five holdouts, baselines and validation-selected blending |
| `scripts/run_ablations.py` | PCA-only and selected-gene-only ablations |
| `scripts/refit_predict.py`, `scripts/predict.py` | Full-data refit and checkpoint-based inference |
| `scripts/profile_memory.py` | Fresh-process MPS tensor/driver memory measurements |
| `scripts/make_figures.py` | Reproduce all four report figures from experiment outputs |
| `tests/` | Metric, split and feature-leakage checks |

## Reproduce

Tested with Python 3.9 and PyTorch 2.8 on Apple Silicon. MPS is required by the
commands below; unavailable GPU access raises an error instead of falling back
to CPU. PCA QR/SVD and ridge fitting use CPU; neural training uses MPS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
make test
make prepare DATA_DIR="/path/to/original-kaggle-files"
make audit benchmark ablations final memory verify figures
```

Download `train_cite_inputs.h5`, `train_cite_targets.h5`, `test_cite_inputs.h5`
and `metadata.csv` after accepting the competition’s access terms. Place them
in `DATA_DIR`; source data, checkpoints and predictions are generated locally
and are not distributed here. `evaluation_ids.csv` is additionally needed only
for the optional `scripts/export_cite_rows.py` export.

For a smaller initial run:

```bash
python scripts/run_benchmark.py --device mps --splits temporal
```

All learned features and target transformations use the training partition.
The validation partition selects checkpoints, ridge regularization and blend
weights; the test partition is used only for evaluation. The temporal protocol
fits day 2, validates on day 3 and tests on day 4. Three donor rotations and a
stratified random reference probe different generalization settings.

## Results

Mean cell-wise Pearson correlation on local held-out cells:

| Test partition | Ridge | ResMLP-Ridge | SS-OPM adapted |
|---|---:|---:|---:|
| Day 4 | 0.8597 | 0.8671 | 0.8688 |
| Donor 13176 | 0.8746 | 0.8793 | 0.8819 |
| Donor 31800 | 0.8707 | 0.8756 | 0.8767 |
| Donor 32606 | 0.8814 | 0.8858 | 0.8877 |
| Random | 0.8893 | 0.8976 | 0.8994 |

Each ResMLP has 1.31M parameters, versus 35.72M for SS-OPM adapted.
ResMLP-Ridge averages three independently trained ResMLPs and blends them with
ridge. Its mean protein-wise correlation on day 4 is 0.273: a high within-cell
profile score does not imply uniformly accurate variation across cells for each
protein. Predictions are relative profiles, not calibrated dsb abundances.

Memory probes use real temporal training arrays, float32, batch 256, ten optimizer
steps and a 1,024-cell evaluation batch. The maximum synchronized stage-sampled
MPS tensor allocation was 67.5 MiB for ResMLP and 594.8 MiB for SS-OPM adapted
in each of three fresh processes. These exclude driver caches and are not hardware
peak measurements. Full tensor/driver samples are written to `results/memory/`.

See [comparator scope](docs/public_models.md) for differences from the original
winner. The comparison uses shared RNA features and does not reproduce the full
winning competition system. No official test-set score is claimed.
