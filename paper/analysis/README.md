# Analysis scripts

Run scripts from the repository root after following the [reproduction guide](../README.md). Paths and model selection are defined in `paper/config.py`.

| Stage | Scripts | Output or purpose |
|---|---|---|
| MPRA preprocessing | `01_prepare_gosai_data.py` | Activity tables; requires raw ENCODE files listed in `data/Gosai_MPRA/metadata.csv` |
| Track metadata | `01_parse_model_track_metadata.py` | Standardized track annotations |
| VEF extraction | `02_extract_sei_vef.py`, `02_extract_ag_vef.py`, `02_extract_castillo_ag_vef.py` | VEF matrices from raw model predictions |
| Normalization | `03_normalize_vef.py` | Enformer/Borzoi feature transformations |
| Feature associations | `04_vef_activity_specificity.py` | Correlations printed to standard output |
| VEF-only baselines | `05_train_vef_only_models.py` | Baseline predictions under `results/vef_only/` |
| Checkpoint inference | `06_infer_trained_model.py` | Predictions from a saved model; GPU recommended |
| Gosai evaluation | `07_eval_regression.py`, `08_eval_classification.py`, `09_eval_retrieval.py` | Regression, classification and retrieval tables |
| Feature relationships | `11_vef_pairwise_correlation.py`, `11_vef_partial_correlation.py` | Pairwise/partial correlations and regression coefficients |
| Prediction export | `14_export_prediction_tables.py` | Per-sequence tables under `results/predictions/` |
| Figure metrics | `15_export_figure_metrics.py` | Aggregated tables for Figures 2 and 3 |
| External validation | `12_eval_castillo.py` | Castillo metrics; requires prediction export first |
| Assay selection | `18_vef_assay_selection.py` | Coverage and activity associations; requires raw AlphaGenome predictions |

The data archive supplies the main activity and VEF tables, so stages 01–03 are only needed when regenerating inputs. It does not supply raw track predictions. Generate those with `scripts/predict_sei.py` or `scripts/predict_alphagenome.py`.

The extraction scripts use fixed dataset paths. For a new dataset, adapt these paths and retain the checkpoint's preprocessing and feature-column order. AlphaGenome extraction defaults to `--variant b`. Sei extraction writes raw-logit and standardized tables; verify the selected output against the path expected by the checkpoint.

To evaluate another AlphaGenome run, set `EPICAST_AG_CONFIG` to its configuration name before running an analysis script. Its predictions must already exist under `saved/`.
