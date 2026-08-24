# analysis_gosai_0722 — 论文分析包

EpiCast 论文的**全部分析代码、中间结果和成图**都在这个目录。仓库全局导览见根目录 `AGENTS.md`。

目录名里的 `gosai_0722` = 主数据集（Gosai MPRA）+ 这一轮分析的起始日期（7月22日）。

---

## 一、目录结构

```
analysis_gosai_0722/
├── config.py          全局定义：路径、细胞系、模型注册表、图例名与配色
├── utils.py           全局函数：VEF 提取、预测加载、mask 构造、CTS 定义、residual
├── analysis/          20 个分析脚本，按编号顺序构成 pipeline  → analysis/README.md
├── plot/              12 个绘图脚本，一个脚本一组图 panel     → plot/README.md
├── results/           analysis/ 的全部产物（csv/npy/tsv），gitignore
│   ├── predictions/       ★ 逐序列自描述表：实测值 + 各模型预测值，由 analysis/14 写
│   ├── figure_metrics/    ★ 逐图 panel 的汇总指标表，由 analysis/15 写，plot/ 读这里
│   └── <其余按分析主题分目录>：correlation/ classification/ retrieval/ castillo/ …
├── figures/           plot/ 的产物（pdf），gitignore
├── manuscript/        论文 docx（旧版本，与代码不一致）
├── deprecated/        已弃用脚本，以及 share_20250805/（旧的一次性交接包）
└── castillo_final_analysis/   师兄 Castillo 方案的原始版本，已整合进主线，见 §7.3
```

**读代码的推荐顺序**：`config.py`（搞清有哪些模型和路径）→ `utils.py`（搞清 CTS/residual 怎么算）→ 具体的 analysis 脚本。

### 1.1 `results/` 的两个特殊子目录

`results/` 下大部分子目录按分析主题命名（`correlation/`、`classification/`…），是各评估脚本自己的完整输出。另外两个不是主题，而是**整个下游都依赖的两层派生数据**，所以单独拎出来：

| 目录 | 粒度 | 写者 | 读者 |
|---|---|---|---|
| `results/predictions/` | 逐序列一行，一个模型一张表，实测值和预测值并排 | `analysis/14` | `plot/fig2b`、`plot/fig3g`、`analysis/12` |
| `results/figure_metrics/` | 逐 (模型, 细胞系) 聚合，一个图 panel 一张表 | `analysis/15` | `plot/` 的 fig2/fig3 系列 |

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

全部注册在 `config.model_registry`，`config.figure_model_names` 是主图里的 13 个，按图例顺序：

| 注册名 | 图例 | 类型 | 预测来源 |
|---|---|---|---|
| `linear_sei_dnase` | Sei DNase Linear | VEF-only | `results/vef_only/sei_dnase/linear_pred.npy` |
| `linear_enformer_dnase` | Enformer DNase Linear | VEF-only | `results/vef_only/enformer_dnase/` |
| `linear_borzoi_dnase` | Borzoi DNase Linear | VEF-only | `results/vef_only/borzoi_dnase/` |
| `linear_ag_dnase` | AlphaGenome DNase Linear | VEF-only | `results/vef_only/ag_dnase/` |
| `linear_sei_vef` / `mlp_sei_vef` / `xgb_sei_vef` | Sei Linear / MLP / XGBoost | VEF-only（4 维） | `results/vef_only/sei_vef/` |
| `linear_ag_vef` / `mlp_ag_vef` / `xgb_ag_vef` | AlphaGenome Linear / MLP / XGBoost | VEF-only（4 维） | `results/vef_only/ag_vef/` |
| `seq_only_3` | Seq-only model | 纯序列 | `saved/0722_gosai_seq_only_256/0722_160527/preds.npy` |
| `epicast_sei_vef` | EpiCast (Sei) | 序列+VEF | `saved/0722_gosai_sei_vef_log1p_256/0723_031345/preds.npy` |
| `epicast_ag_vef` | EpiCast (AlphaGenome) | 序列+VEF | `config.epicast_ag_run`（见下） |

`eval_model_names` 比 `figure_model_names` 多一个 `seq_only_5`（纯序列但用全 5 个细胞系训练，作为上界参考，不进主图）。

配色约定（`config.model_styles`）：单 assay DNase linear 是青→蓝渐变，Sei 四维是红系，AlphaGenome 四维是黄系，纯序列是灰，EpiCast 是紫。

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
11_ctcf_ablation               VEF 边际/偏相关 + 标准化 β
        ▼
14_export_prediction_tables    → results/predictions/*.tsv
        ├── 15_export_figure_metrics  → results/figure_metrics/*.tsv
        └── 12_eval_castillo          → results/castillo/*.csv
        ▼
plot/*.py                      → figures/*.pdf
```

注意 `12_eval_castillo` 排在 `14` **之后**：它读的是 `results/predictions/castillo_*.tsv`，而不是 `saved/` 里的 npy。

两个**旁支**（可选，无 plot 消费，不进论文主图）：`15_eval_mingap_comparison.py`（CTS 定义敏感性）、`16_eval_vef_variant_b2.py`（VEF 预处理变体敏感性）。

除 `06` / `10` 需要 GPU 外，**其余全部是 CPU-only**。

脚本无 CLI 参数（除 `06`），从项目根或本目录都能跑（都用 `sys.path.insert` 定位 `config`/`utils`）：

```bash
conda activate torch
python analysis_gosai_0722/analysis/07_eval_regression.py
```

### plot/ 只读派生表，不读 saved/

这是一个刻意的设计（`config.py` 里有注释）：绘图脚本读 `results/predictions/` 和 `results/figure_metrics/`，不直接碰 `saved/` 里的 npy。npy 没有任何 key，全靠行序和列序对齐 MPRA 表，让每个绘图脚本各自重推一遍这个对齐关系迟早出错；派生表把实测值和预测值放在同一行，读者不需要重推。

后果：改了 `results/` 之后，必须重跑 `14_export_prediction_tables.py` 和 `15_export_figure_metrics.py`，图才会更新。少数几个 plot 脚本（fig1 系列、fig4ab）仍直连 `results/` 和原始 VEF 矩阵，因为它们要用逐序列的 VEF 值，不适合聚合成指标表。

---

## 五、results/ 产物清单

| 子目录 | 生成脚本 | 内容 |
|---|---|---|
| `correlation/` | `07` | `all_models_correlation.csv`（长表：model, split, cell_type, metric, n_eval, value）+ 20 张宽表 `{split}_{metric}.csv`（5 split × 4 指标） |
| `correlation_residual/` | `07` | 同上结构，算在 residual 上 |
| `classification/` | `08` | `all_models_classification.csv` + `test_CTS_{high,low}_{precision,recall,f1,auroc,auprc}.csv` + `curves/test_{cell}_{task}_{roc,pr}.csv` |
| `retrieval/` | `09` | `all_models_retrieval.csv`（含 p@100/1000/10000, ef@…）+ `curves/test_{cell}_{task}_curve.csv` |
| `vef_only/` | `05` | `{sei,enformer,borzoi,ag}_dnase/` 与 `ag_vef/`、`sei_vef/`，每个模型三件套 `*_pred.npy` / `*.joblib` / `*_params.json` |
| `ctcf_ablation/` | `11` | `ctcf_ablation.csv`（marginal_r, partial_r_given_*, beta_given_all3） |
| `predictions/` | `14` | `gosai_{model}.tsv`（13 个主图模型）、`castillo_{dhs64,linear_ag_dnase,vef_only,epicast_ag_vef}.tsv`（4 张，Castillo 侧没有 EpiCast-Sei，见 §6.2 脚注）。逐序列自描述表 |
| `figure_metrics/` | `15_export…` | 12 张 tsv：`fig2c_activity_test`、`fig2d_activity_cts`、`fig3b_residual_{test,cts}`、`fig3{c,d}_cts_{high,low}`、`fig3{c,d}_cts_{high,low}_{roc,pr}`、`fig3{e,f}_retrieval_cts_{high,low}` |
| `castillo/` | `12` | `castillo_{regression,classification}_metrics.csv`、`castillo_cts_counts.csv`。整张 fig5 只依赖这三个文件 |
| `model_track_metadata/` | `01_parse…` | `{sei,enformer,borzoi,alphagenome}_tracks_parsed.csv` |
| `fig1c_assay_coverage/` | `plot/fig1c_…` | assay 覆盖度 count / pct / 总细胞系数（**由 plot 脚本写出，是唯一的例外**） |
| `mingap/` | `15_eval_mingap…` | `mingap_vs_mean_all_models.csv` + 8 张宽表。旁支分析 |
| `vef_variant_b2/` | `16_eval_vef…` | `{linear,xgb,mlp}_pred.npy` + `variant_comparison.csv`。旁支分析 |

**已无 writer 的历史残留**（只被 `deprecated/plot/` 引用，可以忽略）：`train3test2_correlation/`、`train3test2_specific_retrieval/`、`train3test2_standardized_pred_tsv/`、`epicast/leave_one_out_pred*.npy`、`gene_therapy_promoters/`、`fig1c_assay_coverage/all_assays/`、`_castillo_percentile_cts_deprecated/`（旧 Castillo 方案的产物，见 §7.3）。

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

> ⚠️ **`deprecated/share_20250805/README.md` §三 里的参考数值表已过期**，所有依赖 AlphaGenome VEF 的模型（EpiCast-AlphaGenome、AlphaGenome DNase Linear、AlphaGenome MLP）数字都对不上；EpiCast (Sei) 和 DHS64 完全一致。原因清楚：8月20日 VEF 矩阵切换到变体 B（修正 CTCF 列 + DNase 从 1bp 头读），预测表重新导出了，但那份 README 的表没跟着改。**模型排序和所有定性结论不受影响**（EpiCast-AG > EpiCast-Sei > AG DNase Linear > AG MLP > DHS64，与旧表一致）。这份交接包已经发给师兄，他的 `castillo_final_analysis/` 就是基于它算的，所以那里的 CSV 数字同样是旧的 —— 详见 §7.3。

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

**手稿 `manuscript/epicast_0817-chx.docx` 是旧版本，代码是新版本。** 最显眼的是**图编号已经重排，但代码里的文件名没跟着改**。当前推断的对应关系（**待与作者一起核对，不要当成定论**）：

| 代码前缀 | 内容 | 手稿 panel |
|---|---|---|
| `fig1c` | 4 个模型的 assay 覆盖度柱状图 | Fig 1B 附近 / 补充图 |
| `fig1d` | VEF × 活性相关性热图 | Fig 1C |
| `fig1e` | 全集活性-活性相关性热图 | Fig 1D |
| `fig1f` | residual VEF × residual 活性 / VEF specificity | Fig 1E |
| `fig1g` | CTS 集活性-活性相关性热图 | Fig 1F |
| `fig2b` | EpiCast 预测 vs 实测散点 | Fig 2A |
| `fig2c` | 全集回归指标柱状图 | Fig 2B |
| `fig2d` | CTS 集回归指标柱状图 | Fig 2C |
| `fig3aa` / `fig3b` | residual 回归指标 | Fig 3A |
| `fig3c` / `fig3d` | CTS-high / CTS-low 分类（柱 + ROC/PR） | Fig 3B / 3C |
| `fig3e` / `fig3f` | CTS-high / CTS-low 检索曲线 | Fig 3D / 3E |
| `fig3g` | top-k 候选的实测活性 boxplot | Fig 3F |
| `fig4a` / `fig4b` | CTCF 边际 vs 偏相关、符号翻转 | Fig 4（A–E 的一部分） |
| `fig5` | Castillo 回归 + CTS 分类综合图 | **Fig 5** |

手稿把 CTCF/VEF 解耦分析独立成了 Fig 4，Castillo 变成 Fig 5；代码已按手稿编号命名（`plot/fig5_castillo_metrics.py`）。另外手稿 Fig 4 描述了 VEF 两两相关性和多元回归 β（Fig 4A/4C/4E），而代码 `fig4ab_ctcf_ablation.py` 目前只出散点和符号翻转 boxplot。

其他不一致：
- 手稿 Methods 说 Castillo 的 CTS 定义是「超过其他细胞系最大值至少 1」—— **代码现在就是这个口径**（`castillo_cts_gap = 1.0`），见 §7.3。Gosai 那边仍用 1%/99% 分位，两个数据集口径不同是刻意的。
- 手稿说 VEF-only 对 Sei 和 AlphaGenome 各训 3 个模型；代码 `05_train_vef_only_models.py` 现在**只重训 AlphaGenome**，`results/vef_only/sei_vef/` 是旧 run 的产物（脚本注释解释了理由：Sei VEF 矩阵没被 track 索引修正影响，且拟合在固定 seed 下确定，所以旧预测仍然有效）。
- 手稿提到的 `fig4e` 系列（61 细胞系热图）在 `figures/` 里是 `_fig4e_*.pdf`（下划线前缀 = 已弃用），当前没有脚本生成。

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

Castillo（fig5）现在用的是师兄 C.Z. 的方案，原始版本留在 `castillo_final_analysis/`，其中 `数据处理与分析流程.md` 是写得最细的方法说明，**值得先读**。整合方式：

- 计算部分 → `analysis/12_eval_castillo.py`，绘图部分 → `plot/fig5_castillo_metrics.py`（按本仓库「analysis 算指标、plot 只画」的惯例拆开，指标定义一字未改）。
- 已验证：用同一份输入跑师兄原脚本和拆开后的版本，回归 84 行、分类 112 行、CTS 计数**全列 max|diff| = 0**。
- 我原先的 Castillo 方案（分位数口径的 `12_eval_castillo_mpra.py`、`13_eval_castillo_classification.py`、`plot/fig4c_*`、`plot/fig4d_*`）已移入 `deprecated/`，产物移到 `results/_castillo_percentile_cts_deprecated/`。
- 两套方案的关键差别：**CTS 口径**。旧方案用 gap 的 1%/99% 分位（跟 Gosai 一致），新方案用绝对差值 gap ≥ 1。Castillo 只有 8,152 条序列，分位数口径每个细胞系只挑出几十条；且它的活性从没做 z-score，绝对差值本身有意义，还正好对上手稿 Methods 的描述。
- ⚠️ `castillo_final_analysis/` 里的三个 CSV 和那张 PDF 是师兄基于 **8月5日交接包**算的，即变体 B 修正之前的预测。DHS64 的数字和现在完全一致（它不依赖 AlphaGenome VEF），三个 AG 系模型都对不上。**`results/castillo/` 才是当前数字**，`castillo_final_analysis/` 只作为出处和方法说明保留。

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
python analysis_gosai_0722/analysis/07_eval_regression.py
python analysis_gosai_0722/analysis/08_eval_classification.py
python analysis_gosai_0722/analysis/09_eval_retrieval.py

# 同步派生表，然后重画所有图
python analysis_gosai_0722/analysis/14_export_prediction_tables.py
python analysis_gosai_0722/analysis/15_export_figure_metrics.py
python analysis_gosai_0722/analysis/12_eval_castillo.py
for f in analysis_gosai_0722/plot/fig*.py; do python "$f"; done

# 换一个 VEF 预处理变体重新评估，无需改代码
EPICAST_AG_CONFIG=0820_gosai_ag_vef_log1p128_256 \
  python analysis_gosai_0722/analysis/07_eval_regression.py
```
