"""Shared paths, cell types and model registry for the gosai_0722 analysis bundle."""

import os
from pathlib import Path

bundle_root = Path(__file__).resolve().parent
project_root = bundle_root.parent
data_dir = project_root / "data/gosai_mpra"
saved_dir = project_root / "saved"
results_dir = bundle_root / "results"
figures_dir = bundle_root / "figures"
# Two derived layers that the rest of the bundle is built on, kept apart from the
# per-analysis output directories because everything downstream reads them:
#   predictions/     one self-describing table per model, measured activity next to
#                    the predictions, aligned per sequence. Written by analysis/14.
#   figure_metrics/  one aggregated table per figure panel, the only input plot/
#                    reads, so a figure never depends on saved/ or on row order.
# These hold derived tables of this analysis; the raw and shared datasets stay in
# the project-level data/ directory.
predictions_dir = results_dir / "predictions"
figure_metrics_dir = results_dir / "figure_metrics"

train_cell_types = ["K562", "HepG2", "SK-N-SH"]
test_cell_types = ["HCT116", "A549"]
cell_types = train_cell_types + test_cell_types

assays = ["DNase", "H3K4me3", "H3K27ac", "CTCF"]

mpra_raw_path = data_dir / "gosai_mpra_760679_raw.tsv"
mpra_path = data_dir / "gosai_mpra_760679_zscore.tsv"

vef_paths = {
    "sei": data_dir / "gosai_mpra_760679_sei_vef_logit.tsv",
    "enformer": data_dir / "gosai_mpra_760679_enformer_vef_log1p.tsv",
    "borzoi": data_dir / "gosai_mpra_760679_borzoi_vef_log1p.tsv",
    # CTCF used the wrong h5 column before the track indexing was fixed; see
    # analysis/02_extract_ag_vef.py. The pre-fix matrix is kept for comparison.
    "alphagenome": data_dir / "gosai_mpra_760679_ag_vef_x10_log1p_dnase1.tsv",
    # pre-fix matrix, kept only for before/after comparisons
    "alphagenome_prefix": data_dir / "gosai_mpra_760679_ag_vef_x10_log1p.tsv",
}

# name -> (prediction path, kind); kind is consumed by utils.load_pred_dfs
def latest_run(config_name):
    """Newest run directory of a training config that has predictions saved."""
    runs = sorted((saved_dir / config_name).glob("*/preds.npy"))
    return runs[-1] if runs else saved_dir / config_name / "pending/preds.npy"


# EpiCast-AlphaGenome has to be retrained on the corrected VEF matrix (the old
# CTCF column read a neighbouring transcription factor). Until a run of
# configs/0820_gosai_ag_vef_fixctcf_256.yaml has produced predictions, the model
# is dropped from the evaluation lists so no table mixes corrected VEF-only
# results with stale EpiCast predictions.
# Which EpiCast-AlphaGenome training run the analyses should score. Override with
# the EPICAST_AG_CONFIG environment variable to evaluate a different VEF
# preprocessing variant without editing this file, e.g.
#   EPICAST_AG_CONFIG=0820_gosai_ag_vef_log1p128_256 python analysis/07_eval_regression.py
epicast_ag_config = os.environ.get(
    "EPICAST_AG_CONFIG", "0821_gosai_ag_vef_x10_log1p_dnase1_256"
)
epicast_ag_run = latest_run(epicast_ag_config)
epicast_ag_vef_ready = epicast_ag_run.exists()

epicast_sei_run = saved_dir / "0722_gosai_sei_vef_log1p_256/0723_031345/preds.npy"

# Castillo predictions live next to the Gosai predictions of the same run; they
# are produced by analysis/06_infer_trained_model.py with the matching dataset
# config (configs/0821_castillo_dataset_N_dnase1.yaml for AlphaGenome variant B).
castillo_pred_name = "castillo_preds_pad_N.npy"
epicast_ag_castillo_pred = epicast_ag_run.parent / castillo_pred_name
# No Castillo dataset config carries a Sei VEF matrix, so the Sei checkpoint can only
# be run on AlphaGenome VEF, which is not what it was trained on. Nothing consumes
# this path any more; it needs a Sei-VEF inference config before it means anything.
# epicast_sei_castillo_pred = epicast_sei_run.parent / castillo_pred_name

model_registry = {
    "linear_sei_dnase": (results_dir / "vef_only/sei_dnase/linear_pred.npy", "vef-only"),
    "linear_enformer_dnase": (results_dir / "vef_only/enformer_dnase/linear_pred.npy", "vef-only"),
    "linear_borzoi_dnase": (results_dir / "vef_only/borzoi_dnase/linear_pred.npy", "vef-only"),
    "linear_ag_dnase": (results_dir / "vef_only/ag_dnase/linear_pred.npy", "vef-only"),

    "linear_sei_vef": (results_dir / "vef_only/sei_vef/linear_pred.npy", "vef-only"),
    "mlp_sei_vef": (results_dir / "vef_only/sei_vef/mlp_pred.npy", "vef-only"),
    "xgb_sei_vef": (results_dir / "vef_only/sei_vef/xgb_pred.npy", "vef-only"),
    "linear_ag_vef": (results_dir / "vef_only/ag_vef/linear_pred.npy", "vef-only"),
    "mlp_ag_vef": (results_dir / "vef_only/ag_vef/mlp_pred.npy", "vef-only"),
    "xgb_ag_vef": (results_dir / "vef_only/ag_vef/xgb_pred.npy", "vef-only"),
    
    "epicast_sei_vef": (epicast_sei_run, "seq-vef"),
    "epicast_ag_vef": (epicast_ag_run, "seq-vef"),

    "seq_only_3": (saved_dir / "0722_gosai_seq_only_256/0722_160527/preds.npy", "seq-only"),
    "seq_only_5": (saved_dir / "0722_gosai_seq_only_256_5/0723_051843/preds.npy", "seq-only-all-train"),

    # --- previously evaluated runs, kept here so they can be re-enabled in one place ---
    # "lgbm_sei_vef": (results_dir / "vef_only/sei_vef/lgbm_pred.npy", "vef-only"),
    # "lgbm_ag_vef": (results_dir / "vef_only/ag_vef/lgbm_pred.npy", "vef-only"),
    # "0206_gosai_ag_vef_trans0": (saved_dir / "0206_gosai_ag_vef/0206_025223/preds.npy", "seq-vef"),
    # "0206_gosai_ag_vef_trans3": (saved_dir / "0206_gosai_ag_vef/0206_025400/preds.npy", "seq-vef"),
    # "0206_gosai_sei_vef_trans3": (saved_dir / "0206_gosai_sei_vef/0206_103219/preds.npy", "seq-vef"),
    # "0405_gosai_ag_vef_log1p_trans0": (saved_dir / "0405_gosai_ag_vef_log1p/0404_065142/preds.npy", "seq-vef"),
    # "0406_gosai_only_seq_3": (saved_dir / "0406_gosai_only_seq_3/0405_112636/preds.npy", "seq-only"),
    # "0407_gosai_ag_vef_log1p": (saved_dir / "0407_gosai_ag_vef_log1p/0406_042513/preds.npy", "seq-vef"),
    # "0722_gosai_sei_vef_log1p": (saved_dir / "0722_gosai_sei_vef_log1p/0722_083548/preds.npy", "seq-vef"),
    # "0722_gosai_ag_vef_log1p": (saved_dir / "0722_gosai_ag_vef_log1p/0722_083528/preds.npy", "seq-vef"),
    # "0722_gosai_seq_only": (saved_dir / "0722_gosai_seq_only/0722_130620/preds.npy", "seq-only"),
    # "0722_gosai_seq_only_malinois": (saved_dir / "0722_gosai_seq_only_malinois/0722_171043/preds.npy", "seq-only"),
    # "0722_gosai_ag_vef_log1p_256_trans3": (saved_dir / "0722_gosai_ag_vef_log1p_256_trans3/0723_031448/preds.npy", "seq-vef"),
}

# name -> (display name, bar/line color); one hue family per model kind
model_styles = {
    # vef-only DNase linear: teal -> blue
    "linear_sei_dnase": ("Sei DNase Linear", "#B7D9D3"),
    "linear_enformer_dnase": ("Enformer DNase Linear", "#8FC2C8"),
    "linear_borzoi_dnase": ("Borzoi DNase Linear", "#6FA8C6"),
    "linear_ag_dnase": ("AlphaGenome DNase Linear", "#4F7FA8"),
    # vef-only four-feature: Sei red (light -> dark), AlphaGenome yellow (light -> dark)
    "linear_sei_vef": ("Sei Linear", "#F4B4AE"),
    "mlp_sei_vef": ("Sei MLP", "#E87870"),
    "xgb_sei_vef": ("Sei XGBoost", "#C93C35"),
    "linear_ag_vef": ("AlphaGenome Linear", "#F6E9A8"),
    "mlp_ag_vef": ("AlphaGenome MLP", "#EDD056"),
    "xgb_ag_vef": ("AlphaGenome XGBoost", "#D4AD00"),
    # seq-only: grey
    "seq_only_3": ("Seq-only model", "#A8A8A8"),
    "seq_only_5": ("Seq-only model (all train)", "#767676"),
    # seq-vef: purple
    "epicast_sei_vef": ("EpiCast (Sei)", "#B6A9CC"),
    "epicast_ag_vef": ("EpiCast (AlphaGenome)", "#8E7FAF"),
}

# every model scored by analysis/07, 08, 09
eval_model_names = [
    "linear_sei_dnase",
    "linear_enformer_dnase",
    "linear_borzoi_dnase",
    "linear_ag_dnase",
    "linear_sei_vef",
    "mlp_sei_vef",
    "xgb_sei_vef",
    "linear_ag_vef",
    "mlp_ag_vef",
    "xgb_ag_vef",
    "seq_only_3",
    "epicast_sei_vef",
    "epicast_ag_vef",
    "seq_only_5",
]

# subset shown in the main figures, in legend order
figure_model_names = [
    "linear_sei_dnase",
    "linear_enformer_dnase",
    "linear_borzoi_dnase",
    "linear_ag_dnase",
    "linear_sei_vef",
    "mlp_sei_vef",
    "xgb_sei_vef",
    "linear_ag_vef",
    "mlp_ag_vef",
    "xgb_ag_vef",
    "seq_only_3",
    "epicast_sei_vef",
    "epicast_ag_vef",
]

if not epicast_ag_vef_ready:
    print(f"[config] {epicast_ag_config} has no predictions yet; excluding epicast_ag_vef")
    eval_model_names = [name for name in eval_model_names if name != "epicast_ag_vef"]
    figure_model_names = [name for name in figure_model_names if name != "epicast_ag_vef"]


# --- Castillo MPRA cross-dataset evaluation (fig5) ---
castillo_dir = project_root / "data/castillo_mpra"
castillo_mpra_path = castillo_dir / "castillo_mpra_data.tsv"
castillo_vef_path = castillo_dir / "castillo_mpra_ag_vef_x10_log1p_dnase1.tsv"
# all 10 MPRA cell types (column order in castillo_mpra_data.tsv); the export in
# analysis/14 carries all of them, the evaluation uses the 7 below
castillo_all_cell_types = [
    "NT2-D1",
    "GM12878",
    "786-O",
    "SK-N-SH",
    "WERI-Rb-1",
    "SJCRH30",
    "HepG2",
    "K562",
    "MCF-7",
    "HeLa-S3",
]
# the 7 cells that could be matched to an AlphaGenome biosample and therefore have
# model predictions; this is also the reference panel of the Castillo residual
castillo_cell_types = ["K562", "HepG2", "SK-N-SH", "GM12878", "WERI-Rb-1", "MCF-7", "HeLa-S3"]

# Registry keys double as the results/predictions/castillo_{key}.tsv file stems.
castillo_model_names = ["dhs64", "linear_ag_dnase", "vef_only", "epicast_ag_vef"]
castillo_model_styles = {
    "dhs64": ("DHS64", "#DEDEDE"),
    "linear_ag_dnase": ("DNase-AG", "#84A87C"),
    "vef_only": ("AG-VEF-only", "#6EC893"),
    "epicast_ag_vef": ("EpiCast-AG", "#4A57CF"),
}
castillo_cell_colors = {
    "K562": "#D73027",
    "HepG2": "#F28E2B",
    "SK-N-SH": "#E6AB02",
    "GM12878": "#E83E8C",
    "WERI-Rb-1": "#8E44AD",
    "MCF-7": "#00A6D6",
    "HeLa-S3": "#6B3E26",
}

# Castillo uses an activity-difference CTS definition rather than the percentile
# tails used on Gosai: a CRE is CTS-high in a cell type when its activity exceeds
# every other evaluated cell type by at least this margin, and symmetrically for
# CTS-low. The library is two orders of magnitude smaller than Gosai, so a
# percentile tail would select a handful of sequences per cell type, and the
# activities are raw (never z-scored), so an absolute margin is meaningful here.
castillo_cts_gap = 1.0
# screening depths for the enrichment factor panels, in percent
castillo_screen_pcts = (2.0, 5.0)
# below this many positives AUROC/AUPRC/EF are too unstable to plot
castillo_min_positives = 2


def build_models(model_names):
    """model names -> (name, path, kind) tuples accepted by utils.load_pred_dfs."""
    return [(name, *model_registry[name]) for name in model_names]


def build_styles(model_names):
    """model names -> (display names, colors), aligned with model_names order."""
    labels = [model_styles[name][0] for name in model_names]
    colors = [model_styles[name][1] for name in model_names]
    return labels, colors
