# plot/ — 绘图脚本逐个参考

12 个脚本，全部输出 PDF 到 `../figures/`。背景见 `../README.md`，数据来源见 `../analysis/README.md`。

## 通用约定

- 用 `sys.path.insert` 定位 `config`，从项目根或 `paper/` 都能跑，无 CLI 参数。
- 统一 `from epicast.utils.plot_utils import set_mpl_params` 设 matplotlib 默认值，多数还会 `sns.set_theme(style="whitegrid", context="notebook")`。
- 图例名和配色**只从 `config.build_styles(figure_model_names)` 取**，不在绘图脚本里硬编码，所以改配色只需改 `config.model_styles`。
- 存图统一 `dpi=400`。

```bash
conda activate torch
cd /home/hxcai/EpiCast
for f in paper/plot/fig*.py; do python "$f"; done
```

## 输入来源分两类（重要）

| 来源 | 脚本 | 说明 |
|---|---|---|
| **`results/figure_metrics/`** | fig2c3b, fig3aa, fig3cd, fig3ef | 一张图一个 tsv。**改了 `results/` 后必须先重跑 `analysis/15_export_figure_metrics.py`** |
| **`results/predictions/`** | fig2b, fig3g | 要逐序列的实测值和预测值。**改了预测后必须先重跑 `analysis/14_export_prediction_tables.py`** |
| **`results/castillo/`** | fig5 | 由 `analysis/12_eval_castillo.py` 写 |
| **`results/` + 原始 VEF 矩阵** | fig1c, fig1d, fig1eg, fig1f, fig4ab | 这些图要用逐序列的 VEF 值，不适合聚合成指标表，所以直连原始数据 |

共同点是**没有一个绘图脚本直接读 `saved/` 里的 npy**：npy 不带 key，全靠行序对齐 MPRA 表，让每个脚本各自重推一遍这个对齐关系迟早出错。

如果 `results/figure_metrics/` 里缺文件，`fig2c3b_metric_comparison.py` 会直接抛 `FileNotFoundError` 并提示先跑 `analysis/15_export_figure_metrics.py`。

---

## Fig 1 — VEF 携带细胞环境信息

### `fig1c_vef_coverage_bars.py`
四个 sequence-to-function 模型各自覆盖了哪些 assay、多少细胞系。

- **读**：`results/model_track_metadata/*_tracks_parsed.csv`
- **写**：`figures/fig1c_{sei,enformer,borzoi,alphagenome}_assay_coverage_bars.pdf`
- 副产物：同时把 `results/fig1c_assay_coverage/{assay_coverage_count,assay_coverage_pct,model_total_cell_types}.csv` 写出来 —— **这是唯一一个往 `results/` 写东西的 plot 脚本**

### `fig1d_vef_activity_correlation.py`
每个 VEF 与对应细胞系实测活性的 Pearson r 热图。

- **读**：`config.mpra_path` + `config.vef_paths`
- **写**：`figures/fig1d_{sei,enformer,borzoi,alphagenome,alphagenome_prefix}_vef_activity_correlation_heatmap.pdf`
- `alphagenome_prefix` 是 track 索引修正**前**的矩阵（变体 A），只用于修正前后对比
- Enformer / Borzoi 在部分细胞系缺 H3K27ac track，热图上是空缺 —— 这也是论文最终只用 Sei 和 AlphaGenome 构建 EpiCast 的原因

### `fig1eg_activity_activity_correlation.py`
细胞系两两之间的实测活性相关性，一次出两张图。

- **读**：`config.mpra_path`
- **写**：`figures/fig1e_total_activity_activity_correlation_heatmap.pdf`（全集）、`figures/fig1g_all_cts_1_99_activity_activity_correlation_heatmap.pdf`（union CTS 集）
- 两张对比就是论文的关键论证：全集上细胞系间高度相关（活性主要是共享的），CTS 集上相关性大幅下降（residual 筛选确实富集了细胞特异元件）

### `fig1f_vef_specificity.py`
residual VEF 与 residual 活性的关联，区分匹配 / 不匹配细胞系。

- **读**：`config.mpra_path` + `config.vef_paths`
- **写**：`figures/fig1f_{sei,alphagenome}_DNase_{total,all_cts_1_99}_vef_specificity_heatmap.pdf` 和 `..._residual_vef_activity_heatmap.pdf`

---

## Fig 2 — 跨细胞系的活性预测

### `fig2b_epicast_scatter.py`
EpiCast 预测值 vs 实测活性的散点。

- **读**：`results/predictions/gosai_epicast_ag_vef.tsv`
- **写**：`figures/fig2b_epicast_ag_vef_scatter_{K562,HepG2,SK-N-SH,HCT116,A549}.pdf`

### `fig2c3b_metric_comparison.py` ★ 一个脚本出四组柱状图
Fig 2C / 2D / 3B / 3C / 3D 的柱状图全在这里，共 6 个 panel 组。

- **读**：`results/figure_metrics/` 下 `fig2c_activity_test.tsv`、`fig2d_activity_cts.tsv`、`fig3b_residual_test.tsv`、`fig3b_residual_cts.tsv`、`fig3c_cts_high.tsv`、`fig3d_cts_low.tsv`
- **写**（文件名由 `{prefix}_{stem}_{metric}_{cell_type}.pdf` 拼出，`test_cell_types` = HCT116 / A549）：
  - `fig2c_{pearson,spearman,mae,rmse}_{HCT116,A549}.pdf`
  - `fig2d_all_cts_1_99_{pearson,spearman,mae,rmse}_{HCT116,A549}.pdf`
  - `fig3b_{...}` 和 `fig3b_all_cts_1_99_{...}`（residual 版）
  - `fig3c_CTS_high_{auroc,auprc}_{HCT116,A549}.pdf`、`fig3d_CTS_low_{auroc,auprc}_{HCT116,A549}.pdf`
  - 图例单独出：`fig2c_legend.pdf`、`fig3b_legend.pdf`、`fig3c_legend.pdf`
- 相关性类指标的 y 轴锁定 `(0, 0.8)`，AUROC 锁定 `(0, 1)`，MAE/RMSE 自适应（`metric_plot_cfg`）
- 要增删 panel，改脚本顶部的 `panels` 列表即可

---

## Fig 3 — 细胞特异元件的排序

### `fig3aa_residual_metric_comparison.py`
把 residual 的多个指标合并成一张组合 panel。

- **读**：`results/figure_metrics/fig3b_residual_cts.tsv`
- **写**：`figures/fig3aa_residual_cts_metrics.pdf`、`fig3aa_legend.pdf`
- 名字里的 `aa` 不是笔误，是为了和 `fig3b_*` 在文件名排序上区分开

### `fig3cd_cts_classification.py`
CTS-high / CTS-low 的 ROC 和 PR 曲线。

- **读**：`results/figure_metrics/fig3{c,d}_cts_{high,low}_{roc,pr}.tsv`
- **写**：`figures/fig3{c,d}_{HCT116,A549}_cts_{high,low}_{roc,prc}.pdf`
- 虚线是随机期望

### `fig3ef_topk_retrival.py`
top-k% 检索曲线，四种指标各一张。（文件名 `retrival` 是既有拼写，别改，会断引用。）

- **读**：`results/figure_metrics/fig3e_retrieval_cts_high.tsv`、`fig3f_retrieval_cts_low.tsv`
- **写**：`figures/fig3{e,f}_{HCT116,A549}_cts_{high,low}_{ef,p,nns,r}_curve.pdf` + `fig3e_legend.pdf`
- 四个后缀：`ef` = enrichment factor，`p` = precision@k，`nns` = number needed to screen，`r` = recall@k

### `fig3g_residual_topk_activity_boxplot.py`
模型判定为 CTS 的 top-k 序列，在各细胞系里的**实测**活性分布。

- **读**：`results/predictions/gosai_epicast_ag_vef.tsv`
- **写**：`figures/fig3g_{HCT116,A549}_cts_{high,low}_{pct1,n100}_activity_boxplot.pdf`
- `pct1` = 取预测排序的前 1%，`n100` = 取前 100 条
- 这是论文用来说明「EpiCast 不是简单地挑出普遍活跃/不活跃的元件，而是真的富集了目标细胞系偏好的序列」的那张图

---

## Fig 4 — VEF 编码了什么

### `fig4ab_ctcf_ablation.py`
CTCF 的边际相关 vs 条件化后的偏相关。

- **读**：`results/ctcf_ablation/ctcf_ablation.csv` + `config.mpra_path` + `config.vef_paths`
- **写**：
  - 散点（以 K562 为例，3 张）：`figures/fig4a_{alphagenome,sei}_K562_activity_vs_ctcf.pdf`、`..._residual_activity_vs_residual_ctcf_given_dnase.pdf`、`..._residual_activity_vs_residual_ctcf_given_other3vef.pdf`
  - 符号翻转 boxplot：`figures/fig4b_{alphagenome,sei}_ctcf_{beta,partial_r}_sign_flip_boxplot.pdf`
- `figures/` 里还有一批 `fig4a_ag_vef_*` / `fig4a_sei_vef_*` 旧文件名（7月27日），是命名从 `ag_vef` 改成 `alphagenome` 之前的产物，可忽略
- 手稿 Fig 4 还描述了 VEF 两两相关性（4A/4C）和多元回归 β（4E），**当前脚本没有出这几个 panel**，`analysis/11_ctcf_ablation.py` 的 csv 里有 `beta_given_all3` 列可以用

---

## Fig 5 — Castillo 外部数据集零样本验证

### `fig5_castillo_metrics.py`
一张 4 行 × 5 列的综合图，源自师兄 C.Z. 的方案，版式照搬。

- **读**：`results/castillo/castillo_{regression,classification}_metrics.csv`、`castillo_cts_counts.csv`（先跑 `analysis/12_eval_castillo.py`）
- **写**：`figures/fig5_castillo_combined_metrics.pdf`
- **列**：All activity / CTS union activity / CTS union residual / CTS-high / CTS-low
- **行**：前三列是 PCC、SCC、MAE、RMSE；后两列是 AUROC、Normalized AUPRC、2% EF、5% EF
- 一个模型一个 boxplot（汇总 7 个细胞系），彩色散点是各个细胞系；模型顺序和配色取 `config.castillo_model_names` / `castillo_model_styles`（DHS64 / DNase-AG / AG-VEF-only / EpiCast-AG）
- CTS-low 只画阳性数 ≥ `config.castillo_min_positives`（=2）的细胞系，排除 HepG2（0 个）和 HeLa-S3（1 个），样本太少时 AUROC/AUPRC/EF 不稳定或无定义。图注可写：*Cell types with ≤1 CTS-low sequence were excluded from visualization (HepG2, n = 0; HeLa-S3, n = 1).*
- 每个指标共用一个 y 轴范围，同一行的 panel 之间可比
- ⚠️ **不含 EpiCast (Sei)**，而且不该补上：它的 Castillo 预测是拿 AlphaGenome VEF 的 dataset config 喂 Sei checkpoint 跑出来的（四个 `*castillo_dataset*` 配置全指向 AG VEF，没有一个用 `data/castillo_mpra/sei_vef.tsv`），VEF 口径与训练时不一致。导出那行已在 `analysis/14` 注释掉，旧产物改名为 `results/predictions/_castillo_epicast_sei_vef_wrong_vef.tsv`。要进图得先做一份 Sei VEF 的推理配置重跑

---

## `figures/` 里的历史文件

以 `_` 开头的是已弃用的产物，当前没有脚本生成，可忽略：`_fig4e_5_seq_61_cell_type_bar.pdf`、`_fig4e_DNase.pdf`、`_fig4e_EpiCast_VEF_only_prediction_heatmap{,_cluster}.pdf`（61 细胞系热图，对应根 README 提到的 web server 场景）。

## `deprecated/plot/`

`_fig1c_assay_coverage_heatmap.py`、`_fig2d_virtual_ccre.py`、`_fig3c_cts_threshold.py`、`_fig3d_epicast_residual_scatter.py`、`_fig3e_topk_retrival.py`、`_fig4d_known_promoters.py` —— 只有这些还在引用 `results/train3test2_*` 那几个已无 writer 的目录。

`_fig4c_castillo_mpra.py`、`_fig4d_castillo_cts_classification.py` —— 我原先的 Castillo 雷达图（分位数 CTS 口径），已被 `fig5_castillo_metrics.py` 取代。
