"""Shared paths, cell types and model registry for the paper analysis bundle."""

import os
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

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

# One colour per Gosai cell type, shared by every figure that draws cell-type-level
# points or curves (fig 1C, 1E, 3F), so a cell keeps its identity across the paper.
# Blue, yellow, green, red, purple in the order above (C.Z.'s palette). The previous
# version put orange between red and yellow, and at scatter-point size the HepG2 orange
# and the SK-N-SH yellow were not tellable apart, so orange is gone. The purple is
# lighter and less saturated than the other four on purpose: a purple of matching
# saturation reads as much darker than they do, and a darker one sits too close to the
# blue at scatter-point size.
cell_colors = {
    "K562": "#3B75AF",
    "HepG2": "#E6AB02",
    "SK-N-SH": "#2E9E5B",
    "HCT116": "#D73027",
    "A549": "#B294CC",
}

mpra_raw_path = data_dir / "gosai_mpra_760679_raw.tsv"
mpra_path = data_dir / "gosai_mpra_760679_zscore.tsv"

vef_paths = {
    "sei": data_dir / "gosai_mpra_760679_sei_vef_logit.tsv",
    "enformer": data_dir / "gosai_mpra_760679_enformer_vef_log1p.tsv",
    "borzoi": data_dir / "gosai_mpra_760679_borzoi_vef_log1p.tsv",
    # variant B, the matrix every result in the paper is built on
    "alphagenome": data_dir / "gosai_mpra_760679_ag_vef_x10_log1p_dnase1.tsv",
    # variant A: CTCF read the wrong h5 column before the track indexing was fixed
    # (see analysis/02_extract_ag_vef.py). Kept only for before/after comparisons;
    # nothing in the pipeline reads it.
    "alphagenome_prefix": data_dir / "gosai_mpra_760679_ag_vef_x10_log1p.tsv",
}

# name -> (prediction path, kind); kind is consumed by utils.load_pred_dfs
def latest_run(config_name):
    """Newest run directory of a training config that has predictions saved."""
    runs = sorted((saved_dir / config_name).glob("*/preds.npy"))
    return runs[-1] if runs else saved_dir / config_name / "pending/preds.npy"


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

# name -> (display name, bar/line color); one hue family per VEF source, running light
# to dark over DNase / VEF-only linear / MLP / XGBoost / EpiCast.
# Every colour here is the one `figure_model_blocks` below derives from its colormaps,
# model for model, so a model keeps its colour in every figure that draws it. The two
# have to be edited together; changing a colormap there silently drifts from this dict.
# Enformer and Borzoi belong to no block, so their entries are the only free choices
# left: they appear in figure 1 only, which colours by cell type, not by model.
# Display names follow the manuscript wording so legends can be read against the
# text without a translation step.
model_styles = {
    # Sei: YlOrRd
    "linear_sei_dnase": ("Sei DNase", "#FED06C"),
    "linear_sei_vef": ("Sei VEF-only (linear)", "#FD9F44"),
    "mlp_sei_vef": ("Sei VEF-only (MLP)", "#FC5B2E"),
    "xgb_sei_vef": ("Sei VEF-only (XGBoost)", "#E0181D"),
    "epicast_sei_vef": ("EpiCast-Sei", "#A60026"),
    # AlphaGenome: GnBu
    "linear_ag_dnase": ("AlphaGenome DNase", "#C4E8C1"),
    "linear_ag_vef": ("AlphaGenome VEF-only (linear)", "#91D4BD"),
    "mlp_ag_vef": ("AlphaGenome VEF-only (MLP)", "#57B8D0"),
    "xgb_ag_vef": ("AlphaGenome VEF-only (XGBoost)", "#2889BC"),
    "epicast_ag_vef": ("EpiCast-AlphaGenome", "#08599C"),
    # no VEF at all: grey
    "seq_only_3": ("Sequence-only", "#8C8C8C"),
    "seq_only_5": ("Sequence-only (all train)", "#C0C0C0"),
    # in no block
    "linear_enformer_dnase": ("Enformer DNase", "#8FC2C8"),
    "linear_borzoi_dnase": ("Borzoi DNase", "#6FA8C6"),
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

# Models shown in figures 2 and 3, as (block label, colormap, [(name, short label)]).
# Bars run light to dark inside a block and the blocks are drawn with a gap, so the
# Sei and AlphaGenome families read as two groups and the block label carries the
# VEF source that the short labels leave out. The colours these colormaps produce,
# `plt.get_cmap(cmap)(np.linspace(0.28, 0.92, len(models)))`, are the ones written out
# by hand in `model_styles` above; keep the two in sync.
# Enformer and Borzoi appear only in the descriptive figure 1: they lack an H3K27ac
# track in several cell types, which is why EpiCast was built on Sei and AlphaGenome.
seq_only_grey = "#8C8C8C"
seq_only_cmap = LinearSegmentedColormap.from_list("seq_only", [seq_only_grey, seq_only_grey])
figure_model_blocks = [
    (
        "Sei",
        "YlOrRd",
        [
            ("linear_sei_dnase", "DNase"),
            ("linear_sei_vef", "VEF-only (linear)"),
            ("mlp_sei_vef", "VEF-only (MLP)"),
            ("xgb_sei_vef", "VEF-only (XGBoost)"),
            ("epicast_sei_vef", "EpiCast"),
        ],
    ),
    (
        "AlphaGenome",
        "GnBu",
        [
            ("linear_ag_dnase", "DNase"),
            ("linear_ag_vef", "VEF-only (linear)"),
            ("mlp_ag_vef", "VEF-only (MLP)"),
            ("xgb_ag_vef", "VEF-only (XGBoost)"),
            ("epicast_ag_vef", "EpiCast"),
        ],
    ),
    # A one-model block samples a colormap at 0.28, its light end, which on a grey ramp
    # comes out too dark to read as "no VEF"; a flat colormap sets the grey outright.
    ("", seq_only_cmap, [("seq_only_3", "Sequence-only")]),
]

if not epicast_ag_vef_ready:
    print(f"[config] {epicast_ag_config} has no predictions yet; excluding epicast_ag_vef")
    eval_model_names = [name for name in eval_model_names if name != "epicast_ag_vef"]
    figure_model_blocks = [
        (label, cmap, [m for m in models if m[0] != "epicast_ag_vef"])
        for label, cmap, models in figure_model_blocks
    ]

figure_model_names = [name for _, _, models in figure_model_blocks for name, _ in models]
# Residual panels (fig 3A) drop the sequence-only baseline: on a held-out cell type
# its prediction is the mean of the three training cells, so the predicted residual
# is identically zero and carries no variation to correlate.
residual_model_blocks = figure_model_blocks[:2]
residual_model_names = [name for _, _, models in residual_model_blocks for name, _ in models]


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
# Three of the four are the same models Gosai evaluates, so they carry the same colours
# as in `model_styles` (`vef_only` is the AlphaGenome MLP). DHS64 is external and uses
# sequence alone, which is the role grey marks everywhere in this bundle.
# The live fig5 only reads the display names: its boxes are all light grey and colour
# there means cell type. The colours are drawn by the archived `_fig5` script.
castillo_model_styles = {
    "dhs64": ("DHS64", "#A8A8A8"),
    "linear_ag_dnase": ("DNase-AG", "#C4E8C1"),
    "vef_only": ("AG-VEF-only", "#57B8D0"),
    "epicast_ag_vef": ("EpiCast-AG", "#08599C"),
}
# Same ordered palette as `cell_colors` above, extended from five cells to seven. The
# first five entries are that palette verbatim, so the three cells shared with Gosai keep
# their colour; red and purple are reused by GM12878 and WERI-Rb-1, which is safe because
# no figure ever draws Gosai and Castillo cell types together. The last two are brown and
# teal; the pink they replaced sat between the red and the purple and was hard to tell
# from either.
castillo_cell_colors = {
    "K562": "#3B75AF",
    "HepG2": "#E6AB02",
    "SK-N-SH": "#2E9E5B",
    "GM12878": "#D73027",
    "WERI-Rb-1": "#B294CC",
    "MCF-7": "#8C564B",
    "HeLa-S3": "#17A2B8",
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
# A (cell type, task) pair only counts when it has at least this many ground-truth
# positives; below that AUROC/AUPRC/EF are too unstable to compare. On the current
# labels this keeps all seven cell types for CTS-high (the thinnest, MCF-7, has 95)
# and only K562 (84), GM12878 (169) and MCF-7 (281) for CTS-low. CTS-low metrics are
# still computed everywhere, the threshold only governs what is aggregated or drawn.
castillo_min_positives = 20


def build_models(model_names):
    """model names -> (name, path, kind) tuples accepted by utils.load_pred_dfs."""
    return [(name, *model_registry[name]) for name in model_names]


def build_styles(model_names):
    """model names -> (display names, colors), aligned with model_names order."""
    labels = [model_styles[name][0] for name in model_names]
    colors = [model_styles[name][1] for name in model_names]
    return labels, colors
