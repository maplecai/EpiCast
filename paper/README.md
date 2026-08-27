# paper — 论文分析包

EpiCast 论文的**全部分析代码、中间结果和成图**都在这个目录。仓库全局导览见根目录 `AGENTS.md`。

这个目录原名 `analysis_gosai_0722`（主数据集 Gosai MPRA + 这一轮分析的起始日期 7月22日），2026年8月24日改名为 `paper`：日期前缀早已过期，而且外层叫 `analysis` 会和里面的 `analysis/` 子目录撞名。`config.py` 里对应的变量是 `bundle_root`。

⚠️ **`analysis_gosai_0722_backup/` 没有跟着改名**，它是改名之前的旧快照，`analysis/16_eval_vef_variant_b2.py` 仍然按原名读它。

---

## 一、目录结构

```
paper/
├── config.py          全局定义：路径、细胞系、模型注册表、图例名与配色
├── utils.py           全局函数：VEF 提取、预测加载、mask 构造、CTS 定义、residual
├── analysis/          20 个分析脚本，按编号顺序构成 pipeline  → analysis/README.md
├── plot/              绘图脚本，一个脚本一组图 panel           → plot/README.md
├── results/           analysis/ 的全部产物（csv/npy/tsv），gitignore
│   ├── predictions/       ★ 逐序列自描述表：实测值 + 各模型预测值，由 analysis/14 写
│   ├── figure_metrics/    ★ 逐图 panel 的汇总指标表，由 analysis/15 写，plot/ 读这里
│   └── <其余按分析主题分目录>：correlation/ classification/ retrieval/ castillo/ …
├── figures/           plot/ 的产物（pdf），gitignore
├── manuscript/        论文 docx（旧版本，与代码不一致）、逆向生成的画图描述、Castillo 方法说明
└── deprecated/        已弃用脚本，以及 share_20250805/（旧的一次性交接包）
```

**读代码的推荐顺序**：`config.py`（搞清有哪些模型和路径）→ `utils.py`（搞清 CTS/residual 怎么算）→ 具体的 analysis 脚本。

### 1.1 `results/` 的两个特殊子目录

`results/` 下大部分子目录按分析主题命名（`correlation/`、`classification/`…），是各评估脚本自己的完整输出。另外两个不是主题，而是**整个下游都依赖的两层派生数据**，所以单独拎出来：

| 目录 | 粒度 | 写者 | 读者 |
|---|---|---|---|
| `results/predictions/` | 逐序列一行，一个模型一张表，实测值和预测值并排 | `analysis/14` | `plot/fig2a`、`plot/fig3fg`、`analysis/12` |
| `results/figure_metrics/` | 逐 (模型, 细胞系) 聚合，一次评估一张表 | `analysis/15` | `plot/` 的 fig2/fig3 系列 |

命名说明：这两个目录**不叫 `data/` 和 `metrics/`**，不是为了避开根目录 `data/`，而是因为 `predictions` 和 `figure_metrics` 本身更准确——根目录 `data/` 放的是原始与外部数据集，这里放的是本分析派生出来的预测表和指标表。同时 `figure_metrics` 与同级的 `correlation/`、`classification/` 区分得开：后者是评估脚本的完整长表，前者是专供绘图的窄表。

> 历史包袱：这两个目录原来在 `share/data/` 和 `share/metrics/` 下。`share/` 本意是给师兄的一次性交接包，后来分析脚本开始从里面读写，交接包变成了 pipeline 的必经环节，这是不合理的。现已移入 `results/`，旧的 `share/`（只剩 `README.md` 和 `code/` 快照）归档到 `deprecated/share_20250805/`。**注意 `deprecated/share_20250805/README.md` 里的参考数值表已过期**，见 §6.2。

---

## 二、核心定义（不搞清这些，后面所有数字都会看错）

### 2.1 两个正交的 train/test

| 维度 | 划分 | 定义在 |
|---|---|---|
| **序列**，按染色体 | `test` = chr7, chr13（63,698 条）<br>`val` = chr19, chr21, chrX（59,460 条）<br>`train` = 其余（637,521 条） | `utils.build_basic_masks` |
| **细胞系** | `train_cell_types` = K562, HepG2, SK-N-SH<br>`test_cell_types` = HCT116, A549 | `config.py` |

论文所有评估都在 **test 染色体** 上做；跨细胞系泛化看的是 **HCT116 / A549**。这两套 train/test 完全独立，不要混。

### 2.2 缺失值与「子宇宙」原则

| 细胞系 | 有实测值 | 覆盖率 |
|---|---|---|
| K562 / HepG2 / SK-N-SH | 760,679 | 100% |
| HCT116 | 448,103 | 58.9% |
| A549 | 319,496 | 42.0% |

预测值一侧是满的（760,679 行全有）。**处理细胞系 c 时，把「有 c 实测值的那部分」当成完整数据集**，所有百分位、排序、指标都在这个子集里算。在全集上取百分位再套到子集会让预测正例率和真实正例率对不上。

### 2.3 Residual / gap

论文核心是细胞特异性，所以大量指标算在 residual 上，定义是**减去 3 个训练细胞系的均值**（不是全部 5 个）：

```python
gap_true = df[c] - df[["K562", "HepG2", "SK-N-SH"]].mean(axis=1)
gap_pred = df[f"{c}_pred"] - df[[f"{t}_pred" for t in train_cell_types]].mean(axis=1)
```

实现在 `utils.load_residual_eval_dfs`。选这 3 个作参考系的理由：只有它们有全量实测值。

**一个重要推论**：纯序列模型在 HCT116/A549 上的预测是 3 个训练头的均值，所以它的预测 residual 恒等于 0 —— 它在构造上就无法排序细胞特异变体，AUROC 恒为 0.5，precision/recall 恒为 0。`utils.load_train_pred_df` 的 docstring 明确记录了这一点，Fig 3 的部分 panel 因此不画它。

### 2.4 CTS high / low

Gosai 用 residual 的两端 1% / 99%（`utils.build_cts_labels`、`build_cts_tail_masks`）：

```python
vals = gap.dropna()                # 先 dropna，即上面的子宇宙
cts_high = gap > np.percentile(vals, 99)
cts_low  = gap < np.percentile(vals, 1)
```

因此每个细胞系的 CTS 集合恰好是它自己 measured 子集的 1%：K562/HepG2/SK-N-SH 各 7,607 条，HCT116 4,482 条，A549 3,195 条。把各细胞系的 high 和 low 并起来得到 **union CTS set**，作为严格评估集。

Castillo 那边阈值是 5% / 95%，且先把 10 个细胞系各自 z-score、再用 10 个细胞系 z-score 的均值当参考系。真值标签用 10 个细胞系定义，但**只在有预测的 7 个细胞系上评估**（预测侧的参考系只能用这 7 个）。

`utils.py` 里还有两个**备选** CTS 定义，用于敏感性分析，不是论文主口径：

- `build_mingap_labels` / `mingap_scores` — Gosai et al. 原文的定义：`high = A_c - max(其他参考细胞系)`。正的 high 分数保证该元素确实是所比较细胞系中最活跃的，语义上比「减均值」更贴近「selective」。由 `analysis/15_eval_mingap_comparison.py` 使用。
- `build_specific_masks` — 与次高细胞系的差值取 99 分位。

### 2.5 mask 的组合语法

`utils.get_mask` 支持用字符串组合 mask，`&` 和 `|` 都支持，这就是 results 里 `test&all_cts_1_99_pearson.csv` 这类文件名的来源：

```python
get_mask("test&cts_1_99", masks, cell_type="HCT116")
```

---

## 三、被比较的模型

全部注册在 `config.model_registry`。`config.eval_model_names` 是被 `analysis/07`、`08`、`09` 打分的 14 个模型；主图只用其中 11 个，由 `config.figure_model_blocks` 按 VEF 源分块给出，柱图就按这个顺序画、块之间留间隔：

| 块 | 注册名 | 图例 | 类型 | 预测来源 |
|---|---|---|---|---|
| Sei | `linear_sei_dnase` | DNase | VEF-only（1 维） | `results/vef_only/sei_dnase/linear_pred.npy` |
| Sei | `linear_sei_vef` / `mlp_sei_vef` / `xgb_sei_vef` | VEF-only (linear / MLP / XGBoost) | VEF-only（4 维） | `results/vef_only/sei_vef/` |
| Sei | `epicast_sei_vef` | EpiCast | 序列+VEF | `saved/0722_gosai_sei_vef_log1p_256/0723_031345/preds.npy` |
| AlphaGenome | `linear_ag_dnase` | DNase | VEF-only（1 维） | `results/vef_only/ag_dnase/` |
| AlphaGenome | `linear_ag_vef` / `mlp_ag_vef` / `xgb_ag_vef` | VEF-only (linear / MLP / XGBoost) | VEF-only（4 维） | `results/vef_only/ag_vef/` |
| AlphaGenome | `epicast_ag_vef` | EpiCast | 序列+VEF | `config.epicast_ag_run`（见下） |
| — | `seq_only_3` | Sequence-only | 纯序列 | `saved/0722_gosai_seq_only_256/0722_160527/preds.npy` |

不进主图的三个：`linear_enformer_dnase`、`linear_borzoi_dnase`（Enformer / Borzoi 只出现在描述性的 Fig 1，它们在多个细胞系缺 H3K27ac track，所以 EpiCast 只用 Sei 和 AlphaGenome 构建）、`seq_only_5`（纯序列但用全 5 个细胞系训练，作为上界参考）。它们仍然被完整评估，只是被 `analysis/15` 过滤掉，不进 `results/figure_metrics/`。

`config.residual_model_blocks` 是前两块共 10 个模型，Fig 3A 用它：held-out 细胞的 sequence-only 预测被定义为三个训练细胞的均值，predicted residual 恒为 0。

配色约定：Fig 2/3 的柱子颜色由 `figure_model_blocks` 里的 colormap 决定（Sei 用 `YlOrRd`，AlphaGenome 用 `GnBu`，块内由浅到深，sequence-only 单独一个深灰），所以同一个模型在 Fig 2 和 Fig 3 里颜色一致。`config.model_styles` 保留每个模型的显示名和单色，供别处使用。细胞系配色在 `config.cell_colors`（Gosai）和 `config.castillo_cell_colors`：按细胞系顺序取蓝黄绿红紫（2026-08-25 C.Z. 定，Castillo 后面再接棕和青），紫比其余四个更浅、饱和度更低。旧配色的橙（HepG2）和黄（SK-N-SH）在散点尺寸下分不开，所以整条色阶换掉了。

Castillo 那边是另一套（`config.castillo_model_names`）：`dhs64`、`vef_only`（= AlphaGenome MLP）、`linear_ag_dnase`、`epicast`。

### EpiCast-AlphaGenome 的 run 是可切换的

```python
epicast_ag_config = os.environ.get("EPICAST_AG_CONFIG", "0821_gosai_ag_vef_x10_log1p_dnase1_256")
epicast_ag_run = latest_run(epicast_ag_config)     # 该 config 下最新一个有 preds.npy 的 run
```

所以想评估另一个 VEF 预处理变体，不用改代码：

```bash
EPICAST_AG_CONFIG=0820_gosai_ag_vef_log1p128_256 python analysis/07_eval_regression.py
```

如果该 config 还没有预测，`config.py` 会打印提示并**自动把 `epicast_ag_vef` 从 `eval_model_names` 和 `figure_model_names` 里剔除**，避免某张表把修正后的 VEF-only 结果和过期的 EpiCast 预测混在一起。环境变量在 `import config` 时生效。

---

## 四、Pipeline

完整依赖顺序（详细的逐脚本 IO 见 `analysis/README.md`）：

```
01_prepare_gosai_data          原始 ENCODE → 760,679 行 + z-score 标签表
01_parse_model_track_metadata  4 个模型的 track 元数据 → 统一 assay/cell type 命名
        │
        ├── 02_extract_sei_vef / 02_extract_ag_vef / 02_extract_castillo_ag_vef
        │   03_normalize_vef                          → VEF 矩阵
        ▼
05_train_vef_only_models       VEF-only baseline（sklearn/XGBoost，CPU）
06_infer_trained_model         EpiCast checkpoint → preds.npy            [GPU]
10_predict_castillo_mpra.sh    EpiCast → castillo_preds_pad_N.npy        [GPU]
        ▼
07_eval_regression             回归指标（activity + residual）
08_eval_classification         CTS 分类指标 + ROC/PR 曲线
09_eval_retrieval              top-k 检索指标（p@k, EF, NNS）
11_vef_partial_correlation     VEF 边际/偏相关 + 标准化 β
        ▼
14_export_prediction_tables    → results/predictions/*.tsv
        ├── 15_export_figure_metrics  → results/figure_metrics/*.tsv
        └── 12_eval_castillo          → results/castillo/*.csv
        ▼
plot/*.py                      → figures/*.pdf
```

注意 `12_eval_castillo` 排在 `14` **之后**：它读的是 `results/predictions/castillo_*.tsv`，而不是 `saved/` 里的 npy。

四个**旁支**（可选，无 plot 消费，不进论文主图）：`15_eval_mingap_comparison.py`（Gosai 的 CTS 定义敏感性）、`16_eval_vef_variant_b2.py`（VEF 预处理变体敏感性）、`17_eval_castillo_sei.py`（Sei 侧模型的 Castillo 评估）、`18_eval_castillo_ranking_score.py`（Castillo 排序分数 gap vs residual）。

除 `06` / `10` 需要 GPU 外，**其余全部是 CPU-only**。

脚本无 CLI 参数（除 `06`），从项目根或本目录都能跑（都用 `sys.path.insert` 定位 `config`/`utils`）：

```bash
conda activate torch
python paper/analysis/07_eval_regression.py
```

### plot/ 只读派生表，不读 saved/

这是一个刻意的设计（`config.py` 里有注释）：绘图脚本读 `results/predictions/` 和 `results/figure_metrics/`，不直接碰 `saved/` 里的 npy。npy 没有任何 key，全靠行序和列序对齐 MPRA 表，让每个绘图脚本各自重推一遍这个对齐关系迟早出错；派生表把实测值和预测值放在同一行，读者不需要重推。

后果：改了 `results/` 之后，必须重跑 `14_export_prediction_tables.py` 和 `15_export_figure_metrics.py`，图才会更新。fig1 系列是例外，直连原始 MPRA 表和 VEF 矩阵，因为它们要用逐序列的 VEF 值，不适合聚合成指标表。

---

## 五、results/ 产物清单

| 子目录 | 生成脚本 | 内容 |
|---|---|---|
| `correlation/` | `07` | `all_models_correlation.csv`（长表：model, split, cell_type, metric, n_eval, value）+ 20 张宽表 `{split}_{metric}.csv`（5 split × 4 指标） |
| `correlation_residual/` | `07` | 同上结构，算在 residual 上 |
| `classification/` | `08` | `all_models_classification.csv` + `test_CTS_{high,low}_{precision,recall,f1,auroc,auprc}.csv` + `curves/test_{cell}_{task}_{roc,pr}.csv` |
| `retrieval/` | `09` | `all_models_retrieval.csv`（含 p@100/1000/10000, ef@…）+ `curves/test_{cell}_{task}_curve.csv` |
| `vef_only/` | `05` | `{sei,enformer,borzoi,ag}_dnase/` 与 `ag_vef/`、`sei_vef/`，每个模型三件套 `*_pred.npy` / `*.joblib` / `*_params.json` |
| `vef_partial_correlation/` | `11_vef_partial…` | `vef_partial_correlation.csv`，80 行 = 2 个 VEF 源 × {absolute, residual} × 5 个细胞系 × 4 个 assay，列含 `marginal_r, partial_r, beta`。fig4B / fig4D / fig4E 的输入 |
| `ctcf_ablation/` | 已无生成脚本 | `ctcf_ablation.csv`。`11_ctcf_ablation` 推广成 `11_vef_partial_correlation` 之后留下的旧产物，只有归档的 `plot/_fig4bde` 还在读 |
| `vef_pairwise_correlation/` | `11_vef_pairwise…` | `vef_pairwise_correlation.csv`，120 行 = 2 个 VEF 源 × {absolute, residual} × 6 个 assay 对 × 5 个细胞系。fig4A / fig4C 的输入 |
| `predictions/` | `14` | `gosai_{model}.tsv`（13 个主图模型）、`castillo_{dhs64,linear_ag_dnase,vef_only,epicast_ag_vef}.tsv`（4 张，Castillo 侧没有 EpiCast-Sei，见 §6.2 脚注）。逐序列自描述表 |
| `figure_metrics/` | `15_export…` | 12 张 tsv，**名字不带图号**：`activity_{test,cts}`、`residual_{test,cts}`、`cts_{high,low}`、`cts_{high,low}_{roc,pr}`、`retrieval_cts_{high,low}`。与手稿 panel 的对应写在脚本 docstring 里 |
| `castillo/` | `12` | `castillo_{regression,classification}_metrics.csv`、`castillo_cts_counts.csv`。整张 fig5 只依赖这三个文件 |
| `castillo_sei/` | `17_eval_castillo…` | `castillo_sei_{regression,classification}_metrics.csv`。Sei 侧模型的 Castillo 评估，旁支 |
| `castillo_ranking_score/` | `18_eval_castillo…` | `castillo_ranking_score_{metrics,summary}.csv`。同一批 CTS 标签下 gap 排序 vs residual 排序的对比，旁支 |
| `model_track_metadata/` | `01_parse…` | `{sei,enformer,borzoi,alphagenome}_tracks_parsed.csv` |
| `assay_coverage/` | `plot/_fig1c_…` | assay 覆盖度 count / pct / 总细胞系数（**由 plot 脚本写出，是唯一的例外**） |
| `mingap/` | `15_eval_mingap…` | `mingap_vs_mean_all_models.csv` + 8 张宽表。旁支分析 |
| `vef_variant_b2/` | `16_eval_vef…` | `{linear,xgb,mlp}_pred.npy` + `variant_comparison.csv`。旁支分析 |

**2026-08-25 清理**：`train3test2_correlation/`、`train3test2_specific_retrieval/`、`train3test2_standardized_pred_tsv/`、`epicast/`（leave-one-out 预测）、`gene_therapy_promoters/`、`assay_coverage/all_assays/` 共 733 MB 已删除 —— 全仓库没有任何脚本引用它们，也没有 writer 能再生成。

**剩下的两处历史残留**（有 writer 或有归档 reader，先留着）：`ctcf_ablation/`（只被归档的 `plot/_fig4bde` 读，见上表）、`_castillo_percentile_cts_deprecated/`（旧 Castillo 分位数方案的产物，见 §7.3）。

---

## 六、结果：主要结论与可对表的数字

### 6.1 定性结论

1. **VEF 确实携带细胞环境信息**。每个细胞系的 VEF 与该细胞系的实测活性正相关（Fig 1）；但活性本身在细胞系之间高度相关，说明大部分活性是共享的。
2. **减掉共享成分后，匹配细胞系的信号才浮现**。在 union CTS 集上，residual 活性与**匹配**细胞系的 residual DNase 的关联明显强于**不匹配**的细胞系。
3. **全集上 EpiCast ≈ 纯序列模型**；**CTS 集上 EpiCast > 纯序列模型**。这是论文的核心证据：序列编码共享的调控潜力，VEF 编码细胞特异的差异。
4. **CTS-low 比 CTS-high 难**。要区分「在目标细胞系里被特异抑制」和「在所有细胞系里都不活跃」。
5. **检索效率**：CTS-high 富集约 20–80 倍，NNS 降到 10 以下（随机约 100）；CTS-low 富集约 20 倍。
6. **VEF 的条件关联会翻转**。四个 VEF 彼此强相关；控制其余三个后，CTCF 的正相关变成一致的负相关，H3K27ac 大幅衰减。对**绝对活性**，DNase 和 H3K4me3 主导；对 **residual 活性**，H3K27ac 成为最强贡献者 —— 可及性反映共享潜力，增强子特征反映细胞特异差异。
7. **Castillo 零样本可迁移**。活性回归上 EpiCast-AG 全面最强（全集 PCC 中位数 0.585 vs DHS64 0.441）。但在 **CTS-high** 排序上，基于可及性的模型反而更好（2% EF 中位数：DHS64 13.8、DNase-AG 13.8，EpiCast-AG 只有 9.1）—— 该库 78% 是用 DHS64 优化可及性设计出来的合成序列，存在设计偏置，用 DHS64 打分等于部分泄漏了设计目标。**CTS-low** 上这个偏置不起作用，EpiCast-AG 明显最好（AUROC 0.655 vs DHS64 0.569；2% EF 3.9 vs 0.59，后者甚至低于随机）。

### 6.2 可对表的数字

以下数字**于 2026-08-22 直接从 `results/predictions/` 重算并核对通过**。

Gosai `test` 染色体上的 Pearson r（每个细胞系单独算，HCT116/A549 已剔除缺失）：

| 模型 | K562 | HepG2 | SK-N-SH | HCT116 | A549 |
|---|---|---|---|---|---|
| EpiCast (Sei) | 0.8126 | 0.8198 | 0.8115 | 0.7991 | 0.6923 |
| EpiCast (AlphaGenome) | 0.8441 | 0.8435 | 0.8310 | 0.8014 | 0.6980 |

Castillo 全量（8,152 条全用，无划分）的 Pearson r：

| 模型 | K562 | HepG2 | SK-N-SH | GM12878 | WERI-Rb-1 | MCF-7 | HeLa-S3 | 均值 |
|---|---|---|---|---|---|---|---|---|
| EpiCast (AlphaGenome) | 0.7423 | 0.6892 | 0.5848 | 0.4460 | 0.5508 | 0.4380 | 0.6061 | **0.5796** |
| EpiCast (Sei) † | 0.7217 | 0.4627 | 0.5990 | 0.3540 | 0.5622 | 0.4376 | 0.6327 | 0.5386 |
| AlphaGenome DNase Linear | 0.7064 | 0.6903 | 0.1227 | 0.4691 | 0.6796 | 0.5046 | 0.4558 | 0.5183 |
| AlphaGenome MLP | 0.5863 | 0.5917 | 0.3794 | 0.3649 | 0.6609 | 0.4258 | 0.3625 | 0.4816 |
| DHS64 | 0.7565 | 0.5586 | 0.0555 | 0.2932 | 0.6194 | 0.4208 | 0.4409 | 0.4493 |

† **EpiCast (Sei) 这一行不要用**。它的 Castillo 预测是拿 AlphaGenome VEF 的 dataset config 喂 Sei checkpoint 跑出来的（四个 `*castillo_dataset*` 配置全指向 AG VEF），VEF 口径与训练时不一致，数字没有解释力。已于 8月22日处理：`analysis/14` 里的导出、`config.epicast_sei_castillo_pred` 都注释掉了，旧产物改名为 `results/predictions/_castillo_epicast_sei_vef_wrong_vef.tsv`。`config.castillo_model_names` 本来就不含它，fig5 不受影响。要真的比，得先做一份指向 `data/castillo_mpra/sei_vef.tsv` 的推理配置重跑。**注意 Gosai 侧的 EpiCast (Sei) 完全正常**，上面那张 Gosai 表可以用。

此外，fig5 的 Castillo 四模型指标（gap ≥ 1 口径）中位数：

| 模型 | 全集 PCC | CTS-union PCC | CTS-high AUROC | CTS-high 2% EF | CTS-low AUROC | CTS-low 2% EF |
|---|---|---|---|---|---|---|
| DHS64 | 0.4409 | 0.4557 | 0.877 | 13.80 | 0.569 | 0.59 |
| DNase-AG | 0.5046 | 0.5579 | 0.891 | 13.80 | 0.542 | 0.53 |
| AG-VEF-only | 0.4258 | 0.4234 | 0.901 | 8.90 | 0.588 | 2.96 |
| **EpiCast-AG** | **0.5848** | **0.6316** | 0.884 | 9.13 | **0.655** | **3.89** |

CTS 标签数（gap ≥ 1，union = 3,333 / 8,152）：K562 713/84、HepG2 847/0、SK-N-SH 227/6、GM12878 538/169、WERI-Rb-1 400/11、MCF-7 95/281、HeLa-S3 203/1（CTS-high/CTS-low）。CTS-low 图只画阳性数 ≥ 2 的 5 个细胞系（排除 HepG2 的 0 个和 HeLa-S3 的 1 个），共 551 个阳性标签。

> ⚠️ **`deprecated/share_20250805/README.md` §三 里的参考数值表已过期**，所有依赖 AlphaGenome VEF 的模型（EpiCast-AlphaGenome、AlphaGenome DNase Linear、AlphaGenome MLP）数字都对不上；EpiCast (Sei) 和 DHS64 完全一致。原因清楚：8月20日 VEF 矩阵切换到变体 B（修正 CTCF 列 + DNase 从 1bp 头读），预测表重新导出了，但那份 README 的表没跟着改。**模型排序和所有定性结论不受影响**（EpiCast-AG > EpiCast-Sei > AG DNase Linear > AG MLP > DHS64，与旧表一致）。这份交接包已经发给师兄，他那版 Castillo 分析就是基于它算的，所以他给回来的 CSV 数字同样是旧的 —— 详见 §7.3。

重跑了 pipeline 想验证口径是否一致，用这段对第一张表：

```python
import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv("results/predictions/gosai_epicast_ag_vef.tsv", sep="\t")
test = df[df["split"] == "test"]
for c in ["K562", "HepG2", "SK-N-SH", "HCT116", "A549"]:
    v = test[test[c].notna()]
    print(c, round(pearsonr(v[c], v[f"{c}_pred"])[0], 4))
```

---

## 七、已知不一致与坑

### 7.1 代码 vs 手稿

**手稿 `manuscript/epicast_0817-chx.docx` 的正文是旧版本，代码是新版本。** 图的部分已经对齐：`manuscript/epicast_figure_plot_descriptions.md` 是按手稿图片逐 panel 整理的复现说明，`plot/` 下的脚本已按它重命名并重做（映射表见 `plot/README.md`），**Fig 1–5 全部 panel 都有了活跃脚本**。Fig 4 是最后补齐的：4A/4C 出自 `analysis/11_vef_pairwise_correlation.py` + `plot/fig4ac_vef_correlation_heatmap.py`，4B/4D/4E 出自 `analysis/11_vef_partial_correlation.py` + `plot/fig4bde_vef_partial_correlation.py`（2026-08-25，把只算 CTCF 的 `11_ctcf_ablation.py` 推广到 4 个 assay 之后才画得出来）。

#### 要改手稿的实质不一致

- **Castillo 的排序分数**（Methods「Evaluation of cell-type-specific CRE prioritization」段）：手稿写「按 predicted residual activity 排序，CTS-low 取负」，但 `analysis/12_eval_castillo.py` 用的是**预测值之间的 gap**（\(\hat y_c-\max_{j\ne c}\hat y_j\)，CTS-low 对称），让分数和 gap 定义的标签朝向一致。这段 Methods 是通用写法，对 Gosai 成立、对 Castillo 不成立。Fig 5 caption 的「predicted residual activity used in defining CTS CREs」勉强能读成 gap，但也该改直白。权威口径见 `manuscript/castillo_fig5_methods.md` §7。代码保持 gap 不动。
- **Fig 5E 只剩 3 个细胞系**（2026-08-25 恢复画图，等 C.Z. 定去留）：CTS-low 的阳性数是 HepG2 0、HeLa-S3 1、SK-N-SH 6、WERI-Rb-1 11、K562 84、GM12878 169、MCF-7 281，过 `castillo_min_positives = 20` 后只有后三个进图，所以一列只有三个点。手稿 5E 的结论在这三个细胞系上仍然成立（AUROC 中位 EpiCast 0.655 > DHS64 0.569 > AG-VEF-only 0.546 > DNase-AG 0.495；2% EF 中位 1.78 / 0.59 / 1.18 / 0.53），但 caption 必须写出被排除的细胞系和各自的 n。
- **Fig 2A 的相关系数**：手稿图片标 0.799 / 0.694，现在是 HCT116 `0.801`、A549 `0.698`（正文里没有出现这两个数，只在图上）。样本量 36,345 / 27,973 完全一致，差异只来自 VEF 变体 B 的 CTCF 列修正。
- **VEF-only baseline 的重训范围**：手稿 Methods 说 Sei 和 AlphaGenome 各训 3 个模型（共 6 个，与 `model_registry` 一致），但 `05_train_vef_only_models.py` 现在**只重训 AlphaGenome**，`results/vef_only/sei_vef/` 是旧 run 的产物（脚本注释解释了理由：Sei VEF 矩阵没被 track 索引修正影响，拟合在固定 seed 下确定，旧预测仍然有效）。数字没问题，但复现说明要补一句。
- **Fig 4A/4C 的两套矩阵**：手稿的 4A/4C 是一个 4×4 方块、上下三角各放一套 VEF 源，caption 没写哪个三角对应谁。两套数值挤在一格里读不出来，所以代码改成**横向两张下三角热图，左 Sei、右 AlphaGenome**（仍然一个 panel 一个 PDF）。手稿图要换掉，caption 补一句 *Left and right heatmaps show Sei and AlphaGenome VEFs, respectively.*
- **小样本摘要最终采用 mean ± SD**：手稿 Fig 1D、4E、5 的图注写的是 *Horizontal lines and error bars indicate the mean and standard deviation (SD)*，所以 mean ± SD 那一版是定稿版本，已改名占用正式脚本名 `plot/fig1c_...py` / `fig4bde_...py` / `fig5_...py`，legend 也由它输出。三个落选画法归档为 `plot/_figXX..._bar.py`（只有中位数横线）、`_box.py`（有箱体无须线）、`_boxplot.py`（常规箱线图），输出名带 `_` 前缀。选它的理由是这些图只汇总 5 个（Gosai）或 7 个（Castillo）细胞系，n=5 时箱线图的 Q1/Q3 恰好落在第 2、第 4 个点上，1.5×IQR 还会把细胞系判成离群点（fig1c 7/16、fig4bde 22/48、fig5 9/48 个位置触发）。caption 写 *bar, mean; whiskers, ± 1 s.d.; points, individual cell types*。

#### 已按手稿改了代码的

- **Fig 5 的过滤阈值**：caption 写「≤ 5 excluded」，代码原先是 `castillo_min_positives = 2`（排除 ≤1），先对齐成 `6`，2026-08-25 又提到 **`20`**：`>5` 只挡掉 HepG2（0）和 HeLa-S3（1），SK-N-SH 的 6 条、WERI-Rb-1 的 11 条一样支撑不起 AUROC/EF。新阈值下 CTS-high 七个细胞系全留（最薄的 MCF-7 有 95 条），所以 **fig5 的图没有变化**；CTS-low 只剩 K562 / GM12878 / MCF-7 三个，本来就不画，但 `analysis/12` 和 `analysis/18` 照旧算它的指标。caption 改成 *cell type–task combinations with fewer than 20 positive CTS CREs were excluded*。
- **Fig 1E 的点色**：caption 写 matched 是黑点、unmatched 是灰点，代码原先用紫（`#8E7FAF`）+ 灰。已改成黑 + 灰，图已重出。

#### 手稿自身的残缺（不是与代码冲突，但更该先修）

- **Abstract 和 Results 多处掉字**，句子断在半截，例如 Abstract「we introduce virtual epigenomic features (VEFs), sequence- Building on VEFs, we develop EpiCast, integrates DNA sequence with  to predict episomal」。全篇多处 `cell` 后面掉了 `types`。
- **Fig 3A 和 Fig 3B 在正文里没有任何引用段落**。正文的图引用只有 1A–1G、2A–2C、3C–3F、4A–4E、5A–5E；讲 CTS-high prioritization 的那段（`[39]`）只剩「A major goal of cell-type-specific regulatory design is」一句残句，Fig 3A 的 residual 回归也没人提。
- **Discussion 与 Results 口径有张力**：Results 说 VEF-only 明显不如 EpiCast，Discussion 说「VEF-only models retained predictive performance comparable to EpiCast in the CTS setting for the prioritization tasks」。当前数字是两边各占一半——CTS-high AUROC 在 HCT116 上 `mlp_sei_vef` 0.723 ≈ `epicast_ag_vef` 0.722，但在 A549 上 EpiCast 0.725/0.746 明显高于 VEF-only 的 0.647 上限。Discussion 那句要限定到具体细胞系。

#### 已经对齐、不用改的（核对过）

- 数据集数字全对：760,679 条；HCT116 448,103（58.9%）/ A549 319,496（42.0%）；test 63,698 / val 59,460 / train 637,521；Castillo 8,152 = 1,836 + 6,316；145bp 用 27+28 个 N 补到 200bp。
- 模型结构全对：`ConvFiLMNet` 参数量正好 **2,442,761**，conv 输出 feature map 正好 **256 × 4**，FiLM 线性层零初始化，无 cell-type embedding 无 per-cell 输出头。训练超参（AdamW 1e-3 / 1e-4、batch 4096、warmup 10 + T0 10 + Tmult 2、clip 1.0、patience 40、seed 0、按验证染色体上 3 个训练细胞系平均 PCC 选模型）逐项对上。
- **AlphaGenome VEF 的口径就是变体 B**：Methods 写「DNase 从 1bp 分辨率输出读、histone/TF 从 128bp 输出读并除以 128」「变换为 log(1+10x)」，与 `utils.ag_variants["b"]` 完全一致。Sei VEF 也确实做了逐维零均值单位方差（实测 mean ≈ −0.01、std ≈ 0.995）。
- **Enformer/Borzoi 被排除的理由是准确的**：实测确实只有 H3K27ac 缺，且缺的正好是 K562、HepG2、A549 三个细胞系，与 Methods 和 Fig 1 caption 的说法一字不差。
- Castillo 的 CTS 定义（gap ≥ 1）、residual 参考系（全 7 细胞）、Fig 5A–C 只比 PCC/SCC 不比 MAE/RMSE、Fig 3F 画「target + reference cell types」共 4 条曲线、Fig 3A 排除 sequence-only（预测 residual 恒为 0）、Fig 2/3 共 11 个模型（Enformer/Borzoi 的 DNase 只出现在 Fig 1），这些手稿和代码都一致。
- 正文的定性描述基本站得住：CTS-high 的 EF 峰值 84.5(HCT116) / 72.8(A549)，随机 NNS = 1/0.0092 ≈ 108，与「20–80 倍」「随机约 100」对得上。只有「EpiCast consistently achieved strong enrichment」偏乐观——最浅的筛查深度上 `epicast_ag_vef` 在 A549 两个任务的 EF 都是 0，且 NNS「低于 10」只在浅端成立，10% 深度会升到 20–53。

无法从仓库核对的：Sei 的 AUROC > 0.95 筛选（18,889 / 21,907 profiles）、每个 cell–assay 由 3–87 个 Sei profile 贡献、DHS64 用 split 0/1/3 三个 released model。这些在早期 notebook 和 `enhancer-design/` 里，没有留下可复算的中间产物。

### 7.2 config 路径 vs extract 脚本

`config.vef_paths` 指向的文件和 `analysis/02_*` 产出的文件名**不一致**，这不是 bug，而是分工：

| config 变量 | 指向 | 谁生成 |
|---|---|---|
| `vef_paths["alphagenome"]` | `*_ag_vef_x10_log1p_dnase1.tsv`（变体 B） | `02_extract_ag_vef.py`（变体由 `utils.ag_variants` 选） |
| `castillo_vef_path` | `castillo_mpra_ag_vef_x10_log1p_dnase1.tsv` | `02_extract_castillo_ag_vef.py`（同上） |
| `vef_paths["sei"]` | `*_sei_vef_logit.tsv` | 早期 notebook；`02_extract_sei_vef.py` 写的是 `*_logit_raw.tsv` / `*_logit_zscore.tsv`，不覆盖它 |
| `vef_paths["enformer"]`, `["borzoi"]` | `*_vef_log1p.tsv` | `03_normalize_vef.py` |

变体 B/B2/C 的定义集中在 `utils.ag_variants` 里（`ag_default_variant = "b"` 即论文用的那个），两个 extract 脚本共用同一份定义和同一套 track 索引推导，所以不会各自漂移。

### 7.3 Castillo 分析已换成师兄的方案

Castillo（fig5）现在用的是师兄 C.Z. 的方案。方法说明是 `manuscript/castillo_fig5_methods.md`，**动 fig5 之前先读它**：它在排序分数和 normalized AUPRC 公式这两点上比 `manuscript/epicast_figure_plot_descriptions.md` 准确（前者把排序分数写成了 residual，后者把公式列为待确认）。整合方式：

- 计算部分 → `analysis/12_eval_castillo.py`，绘图部分 → `plot/fig5_castillo_metrics.py`（按本仓库「analysis 算指标、plot 只画」的惯例拆开，指标定义一字未改）。
- 已验证：用同一份输入跑师兄原脚本和拆开后的版本，回归 84 行、分类 112 行、CTS 计数**全列 max|diff| = 0**。
- 我原先的 Castillo 方案（分位数口径的 `12_eval_castillo_mpra.py`、`13_eval_castillo_classification.py`、`plot/fig4c_*`、`plot/fig4d_*`）已移入 `deprecated/`，产物移到 `results/_castillo_percentile_cts_deprecated/`。
- 两套方案的关键差别：**CTS 口径**。旧方案用 gap 的 1%/99% 分位（跟 Gosai 一致），新方案用绝对差值 gap ≥ 1。Castillo 只有 8,152 条序列，分位数口径每个细胞系只挑出几十条；且它的活性从没做 z-score，绝对差值本身有意义，还正好对上手稿 Methods 的描述。
- 师兄的原始单文件目录 `castillo_final_analysis/` 已于 2026-08-24 删除：脚本在 git 历史 `a581ac5` 里，方法说明移到了 `manuscript/`，三个 CSV 和那张合并大图是 **8月5日交接包**（变体 B 修正 CTCF 列之前）算的，DHS64 与现在完全一致、三个 AG 系模型都对不上，已被 `results/castillo/` 取代。

### 7.4 其他

- **两个脚本都编号 15**（`15_eval_mingap_comparison.py` 和 `15_export_figure_metrics.py`），职责不同，只是编号撞了。pipeline 里真正在 14 之后的是 `15_export_figure_metrics.py`。
- `06_infer_trained_model.sh` 里当前激活的那行跑的是 `0722_gosai_ag_vef_log1p_256_not_x10`，和 `config.epicast_ag_config` 的默认值不一致；`10_predict_castillo_mpra.sh` 只推理了 Sei。两个 shell 脚本都是「用完随手改」的状态，**要照着注释确认参数再跑，别直接执行**。
- `05_train_vef_only_models.py` 会写出 `ag_vef/{ridge,lasso}_pred.npy`，但这两个不在 `model_registry` 里，不参与评估。
- `04_vef_activity_specificity.py` 只打印到 stdout，不落盘。
- **`deprecated/share_20250805/README.md` 里的参考数值表已过期**，AlphaGenome 系模型的数字对不上，详见 §6.2。同目录下的 `code/` 是 7 个 analysis + 6 个 plot 脚本的快照，给合作者看定义用的，**不是可运行副本**。

---

## 八、常用操作

```bash
conda activate torch
cd /home/hxcai/EpiCast

# 重算全部评估指标（CPU，几分钟）
python paper/analysis/07_eval_regression.py
python paper/analysis/08_eval_classification.py
python paper/analysis/09_eval_retrieval.py

# 同步派生表，然后重画所有图
python paper/analysis/14_export_prediction_tables.py
python paper/analysis/15_export_figure_metrics.py
python paper/analysis/12_eval_castillo.py
for f in paper/plot/fig*.py; do python "$f"; done

# 换一个 VEF 预处理变体重新评估，无需改代码
EPICAST_AG_CONFIG=0820_gosai_ag_vef_log1p128_256 \
  python paper/analysis/07_eval_regression.py
```
