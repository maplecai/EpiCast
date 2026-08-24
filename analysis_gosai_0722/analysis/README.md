# analysis/ — 脚本逐个参考

论文分析流水线，20 个脚本（18 个 `.py` + 2 个 `.sh`）。编号大致代表依赖顺序，但**不是**严格串行（见下方依赖图）。

背景和核心定义在上一级的 `../README.md`，别跳过。

## 通用约定

- 所有脚本用 `sys.path.insert(0, parents[1])` 定位 `config` 和 `utils`，因此**从项目根或从 `analysis_gosai_0722/` 都能跑**。
- **只有三个脚本带 CLI 参数**：`06_infer_trained_model.py`（换 checkpoint / 数据集），以及两个 AlphaGenome 提取脚本的 `--variant`（`02_extract_ag_vef.py`、`02_extract_castillo_ag_vef.py`，默认 `b`）。其余脚本路径全部来自 `config.py`。
- 唯一的环境变量是 `EPICAST_AG_CONFIG`，在 `import config` 时生效，用来切换 EpiCast-AlphaGenome 评估哪个训练 run。
- **只有 `06` 和 `10` 需要 GPU**，其余全是 CPU。
- 跑之前 `conda activate torch`。

```bash
cd /home/hxcai/EpiCast
python analysis_gosai_0722/analysis/07_eval_regression.py
```

## 依赖图

```
01_prepare_gosai_data ──────────────────────────┐
01_parse_model_track_metadata ──→ plot/fig1c    │
                                                │
02_extract_sei_vef ─┐                           │
02_extract_ag_vef ──┼─→ 03_normalize_vef ───────┤   VEF 矩阵
02_extract_castillo_ag_vef ─┘                   │
                                                ▼
                              04_vef_activity_specificity（只打印）
                              05_train_vef_only_models
                              06_infer_trained_model      [GPU]
                              10_predict_castillo_mpra.sh [GPU]
                                                │
        ┌───────────────────────────────────────┼──────────────────┐
        ▼                    ▼                  ▼                  ▼
07_eval_regression   08_eval_classification  09_eval_retrieval  11_ctcf_ablation
        │                    │                  │                  │
        │                    │                  │                  ├→ plot/fig4ab
        └────────────────────┴──────────────────┘
                             ▼
           14_export_prediction_tables  →  results/predictions/
                             │
        ┌────────────────────┴───────────────────┐
        ▼                                        ▼
15_export_figure_metrics              12_eval_castillo
   → results/figure_metrics/             → results/castillo/
        ▼                                        ▼
   plot/fig2*, fig3*                    plot/fig5_castillo_metrics
                                        （fig2b / fig3g 直读 predictions/）

旁支（无下游消费）：15_eval_mingap_comparison、16_eval_vef_variant_b2
```

`12_eval_castillo` 排在 `14` 之后是刻意的：它读 `results/predictions/castillo_*.tsv`，不碰 `saved/` 的 npy。

---

## 数据准备

### `01_prepare_gosai_data.py`
从 Gosai 原始 ENCODE 文件复现 Malinois 式预处理，产出 760,679 行的标签主表。

- **读**：`data/Gosai_MPRA/metadata.csv` + `data/Gosai_MPRA/raw/`（各 accession 的 lab custom processed data 与 FASTA）
- **写**：`data/gosai_mpra/gosai_mpra_760679_raw.tsv`、`gosai_mpra_760679_zscore.tsv`（= `config.mpra_raw_path` / `mpra_path`）
- 处理链：去掉无法解析基因组序列的 CRE → 去掉 pDNA/RNA 覆盖不足的 → 同一序列在同一 source project 内合并 → 跨 project 按 `UKBB > GTEx > OL15` 优先级去重 → 去极低活性离群点 → 按 200bp 序列 join 五个细胞系的表 → 要求 K562/HepG2/SK-N-SH 三者都有实测值
- z-score：每个细胞系**只用 train 染色体**估计均值和标准差，再应用到全部数据，避免 test 统计量泄漏
- **pipeline 起点**，几乎所有下游脚本都依赖它

### `01_parse_model_track_metadata.py`
解析四个 sequence-to-function 模型的 track 元数据，把 assay 名和细胞系名统一到一套命名。

- **读**：`data/Sei/Sei_tracks_info.csv`、`data/Enformer/model_track_info.tsv`、`data/Borzoi/targets_human.txt`、`data/AlphaGenome/metadata.csv`
- **写**：`results/model_track_metadata/{sei,enformer,borzoi,alphagenome}_tracks_parsed.csv`
- 只被 `plot/fig1c_vef_coverage_bars.py` 消费；与主 pipeline 并行，无依赖

---

## VEF 提取与归一化

### `02_extract_ag_vef.py`
从 AlphaGenome 的 h5 预测里按**修正后的** track 索引提取 Gosai 的四 assay VEF。**`--variant` 默认 `b`，即论文主结果用的那份矩阵。**

```bash
python analysis_gosai_0722/analysis/02_extract_ag_vef.py              # 变体 B（默认）
python analysis_gosai_0722/analysis/02_extract_ag_vef.py --variant c  # 变体 C
```

- **读**：`alphagenome_vef/gosai_ag_pred_760k_pad_0.h5`、`alphagenome_vef/metadata_padded.tsv`，可选读 `mpra_path` 打印相关性诊断
- **写**：`data/AlphaGenome/gosai_ag_track_columns.csv` + 两个 VEF 矩阵，文件名后缀跟着变体走（`utils.ag_variants[variant]["suffix"]`），所以矩阵和它的预处理口径永远对得上：
  - 变体 B → `gosai_mpra_760679_ag_vef_raw_dnase1.tsv`、`..._ag_vef_x10_log1p_dnase1.tsv`（**= `config.vef_paths["alphagenome"]`**）
  - 变体 C → `gosai_mpra_760679_ag_vef_raw_128bp.tsv`、`..._ag_vef_log1p_128bp.tsv`
- **track 索引是这个脚本存在的理由**：AlphaGenome 每个输出头是定宽张量，列对应**补齐（padded）后**的元数据行，真实 track 在前、显式 `Padding` 行填尾。早先的实现从「去掉 padding 的元数据」推列号再减硬编码偏移，导致 TF 头错位两格，"CTCF" 列实际读到了邻近的另一个转录因子。修正后的逻辑在 `utils.ag_head_starts` 里从数据推导每个头的起始列，并用 h5 的实际宽度 assert 校验
- 变体定义集中在 `utils.ag_variants`（`ag_default_variant = "b"`），两个提取脚本共用，不会各自跑偏。含义见 `../../AGENTS.md` §5.3 与 `16_eval_vef_variant_b2.py` 的 docstring

### `02_extract_castillo_ag_vef.py`
对 Castillo MPRA 做同样的提取，复用 `utils` 里的同一套 helper，保证两个数据集的 track 索引不会漂移。

- **读**：`data/castillo_mpra/castillo_mpra_ag_pred.h5`、`alphagenome_vef/metadata_padded.tsv`
- **写**：`data/AlphaGenome/castillo_ag_track_columns.csv` + 两个 VEF 矩阵，同样由 `--variant`（默认 `b`）决定后缀：
  - 变体 B → `castillo_mpra_ag_vef_raw_dnase1.tsv`、`castillo_mpra_ag_vef_x10_log1p_dnase1.tsv`（**= `config.castillo_vef_path`**）
  - 变体 C → `castillo_mpra_ag_vef_{raw,log1p}_128bp.tsv`
- 两个数据集必须用**同一个变体**提取，否则 Gosai 上训出来的模型在 Castillo 上会拿到口径不同的 VEF

### `02_extract_sei_vef.py`
从 Sei 的 h5 预测提取 logit VEF 及逐列 z-score 版本。

- **读**：`predict_CRE_activity/outputs/Gosai_MPRA_Sei_pred.h5`、`data/Sei/Sei_tracks_info.csv`
- **写**：`data/Sei/metadata_pivot_vef.csv`、`data/gosai_mpra/gosai_mpra_760679_sei_vef_logit_raw.tsv`、`..._logit_zscore.tsv`
- Sei 输出的是概率，先转 logit，再在匹配到的 profile（AUROC > 0.95 者）上平均
- ⚠️ **不覆盖** `config.vef_paths["sei"]` 指向的 `gosai_mpra_760679_sei_vef_logit.tsv`（那是更早的 notebook 产物）

### `03_normalize_vef.py`
把 Enformer / Borzoi 的 raw VEF 做 log1p 存盘；AlphaGenome 和 Sei 只打印与活性的相关性（它们在提取阶段已归一化）。

- **读**：`mpra_raw_path`、`data/gosai_mpra/gosai_mpra_760679_{ag,enformer,borzoi}_vef_raw.tsv`
- **写**：`data/gosai_mpra/gosai_mpra_760679_{enformer,borzoi}_vef_log1p.tsv`（= `config.vef_paths` 的 enformer / borzoi 项）
- 需要先有 enformer/borzoi 的 raw 矩阵，那是在本目录外生成的

### `04_vef_activity_specificity.py`
Sei 与 AlphaGenome 的 VEF-活性、VEF-specificity 相关矩阵，逻辑与 `plot/fig1f` 相同。

- **读**：`mpra_path`、`vef_paths["sei"]`、`vef_paths["alphagenome"]`
- **写**：无，**只打印到 stdout**。定位是快速 sanity check

---

## 训练与推理

### `05_train_vef_only_models.py`
在 Gosai 上训练所有 VEF-only baseline（不看序列）。

- **读**：`mpra_path`、`vef_paths[source]`
- **写**：
  - `results/vef_only/{sei,enformer,borzoi,ag}_dnase/linear_{pred.npy,.joblib,_params.json}`
  - `results/vef_only/ag_vef/{linear,ridge,lasso,xgb,mlp}_{pred.npy,.joblib,_params.json}`
- 训练样本构造方式很关键：把每条 CRE 与每个训练细胞系配对成一个样本（`get_X_y` 沿细胞系维度 concat），**不显式编码细胞系身份**——细胞环境信息只从 VEF 输入进来。因此训好的模型可以直接套到 held-out 细胞系上
- 只用 train 染色体 + 3 个训练细胞系拟合，缺失对被 `epicast.utils.remove_nan` 剔除
- ⚠️ **只重训 AlphaGenome 的四 assay 模型**。`results/vef_only/sei_vef/` 是旧 run 的产物，仍被 `model_registry` 使用；脚本注释给出了理由：Sei 矩阵不受 track 索引修正影响，且拟合在固定 seed 下确定
- ⚠️ `ridge` 和 `lasso` 会被写出但不在 `model_registry` 里，不参与评估

### `06_infer_trained_model.py` ★ 主要的带 CLI 的脚本
加载 EpiCast checkpoint 做推理，存 `.npy`。**需要 GPU。**

```bash
python analysis_gosai_0722/analysis/06_infer_trained_model.py \
  -c saved/<config>/<run>/config.yaml \              # 或 -k 直接给 checkpoint
  -dc configs/0821_castillo_dataset_N_dnase1.yaml \  # 换数据集，省略则用 config 自带的
  -o preds.npy \
  -de cuda:0
```

- **写**：`{run_dir}/{output_name}`，默认 `preds.npy`
- 换数据集推理（`-dc`）就是 Castillo 零样本预测的实现方式：同一个 checkpoint，喂进 N 补齐到 200bp 的 Castillo 序列和 Castillo 的 VEF
- ⚠️ dataset config 决定喂哪份 VEF。四个 `*castillo_dataset*` 配置**都带 AlphaGenome VEF**，所以只能配 EpiCast-AlphaGenome 的 checkpoint

### `06_infer_trained_model.sh`
Gosai 批量推理的便捷包装，历史命令行都注释在里面。当前激活的是 `0821_gosai_ag_vef_x10_log1p_dnase1_256/0820_155453`，即 `config.epicast_ag_config` 指向的那个 run。

### `10_predict_castillo_mpra.sh`
对 Castillo 跑 EpiCast 推理，产出 `castillo_preds_pad_N.npy`。**需要 GPU。**

- 调用 `06_infer_trained_model.py -c saved/0821_gosai_ag_vef_x10_log1p_dnase1_256/0820_155453/config.yaml -dc configs/0821_castillo_dataset_N_dnase1.yaml -o castillo_preds_pad_N.npy`，即变体 B 的 EpiCast-AlphaGenome，输出正是 `config.epicast_ag_castillo_pred`
- ⚠️ **EpiCast-Sei 的那行是刻意注释掉的**：没有任何 Castillo 配置指向 Sei VEF（`data/castillo_mpra/sei_vef.tsv` 在，但没有配置用它），拿现成的 config 去配 Sei checkpoint 等于喂错 VEF。旧产物已改名为 `results/predictions/_castillo_epicast_sei_vef_wrong_vef.tsv`（`_` 前缀 = 弃用），`14_export_prediction_tables.py` 和 `config.epicast_sei_castillo_pred` 都已注释掉。注意 **Gosai 侧的 EpiCast-Sei 是正常的**，只有 Castillo 侧有这个问题

---

## 评估

以下三个脚本共享同一套输入（`mpra_path` + `config.build_models(eval_model_names)` 的预测），所以可以任意顺序跑。

### `07_eval_regression.py`
对每个模型 × split × 细胞系算 Pearson / Spearman / MAE / RMSE，**在活性和 residual 上各算一遍**。

- **写**：
  - `results/correlation/all_models_correlation.csv`（长表：`model, split, cell_type, metric, n_eval, value`）
  - `results/correlation/{split}_{metric}.csv`（宽表，5 split × 4 指标 = 20 张；列 `model, model_type, K562, HepG2, SK-N-SH, HCT116, A549`）
  - `results/correlation_residual/` 同样结构
- split 取值：`test`、`test&cts_1_99`、`test&all_cts_1_99`、`test&cts_5_95`、`test&all_cts_5_95`

### `08_eval_classification.py`
CTS-high / CTS-low 的分类指标与 ROC / PR 曲线（只在 test 细胞系上）。

- **写**：
  - `results/classification/all_models_classification.csv`（列含 `precision, recall, f1, auroc, auprc, n_eval, n_pos, prevalence`）
  - `results/classification/test_{HCT116,A549}_CTS_{high,low}_by_model.csv`
  - `results/classification/test_CTS_{high,low}_{precision,recall,f1,auroc,auprc}.csv`
  - `results/classification/curves/test_{cell}_CTS_{high,low}_{roc,pr}.csv`
- 排序分数是**预测 residual**；CTS-low 用其相反数

### `09_eval_retrieval.py`
top-k 检索指标：p@k、enrichment factor、NNS，k = 100 / 1000 / 10000，另出 log 间隔的完整曲线。

- **写**：`results/retrieval/all_models_retrieval.csv`、`test_{cell}_CTS_{high,low}_by_model.csv`、`curves/test_{cell}_{task}_curve.csv`

### `11_ctcf_ablation.py`
CTCF 的信号在控制其他 VEF 后还剩多少。

- **读**：`mpra_path`、`vef_paths["alphagenome"]`、`vef_paths["sei"]`
- **写**：`results/ctcf_ablation/ctcf_ablation.csv`，12 列：`vef_source, cell_type, n` + 相关性 `marginal_r, marginal_p, partial_r_given_dnase, partial_p_given_dnase, partial_r_given_all3, partial_p_given_all3` + 标准化系数 `beta_marginal, beta_given_dnase, beta_given_all3`。「控制其余三个 VEF」的那列叫 `partial_r_given_all3`（不是 `_other3`）
- 用 `statsmodels` 做标准化 OLS；这是论文「CTCF 的正相关在条件化后翻转为负」那个结论的来源
- → `plot/fig4ab_ctcf_ablation.py`

---

## 导出与 Castillo

### `14_export_prediction_tables.py`
把「实测值 + 预测值」拼进同一张自描述 tsv，导出到 `results/predictions/`。

- **写**：`results/predictions/gosai_{model}.tsv`（13 个主图模型）、`castillo_{dhs64,vef_only,linear_ag_dnase,epicast_ag_vef}.tsv`（Castillo 侧 4 张，EpiCast-Sei 已注释掉，理由见下）
- 存在的理由（脚本 docstring）：`.npy` 预测文件不带任何 key，靠**行序**与 MPRA 表对齐，下游极易搞错。这些表把两侧放进一个文件，读者不用再自己推对齐关系
- Gosai 表 14 列：`id, chr, pos, split` + 5 个细胞系实测（z-score 后）+ 5 个 `{细胞系}_pred`
- Castillo 表 21 列：`id, category, source, target` + 10 个细胞系实测（**原始值，未 z-score**）+ 7 个 `{细胞系}_pred`
- 列名统一为 `{细胞系}_pred`，所以换模型只要换文件名，读取代码不用改
- Castillo 的 VEF-only 和 DNase linear 是**直接复用 Gosai 上训好的 joblib**，喂 Castillo 的 VEF，所以也是零样本

### `15_export_figure_metrics.py`
把 Fig 2 / 3 用到的指标表导到 `results/figure_metrics/`，**这些 plot 脚本只读这里**。

- **读**：`results/{correlation,correlation_residual,classification,retrieval}/`
- **写**：`results/figure_metrics/` 下 12 个 tsv —— `fig2c_activity_test`、`fig2d_activity_cts`、`fig3b_residual_{test,cts}`、`fig3{c,d}_cts_{high,low}`、`fig3{c,d}_cts_{high,low}_{roc,pr}`、`fig3{e,f}_retrieval_cts_{high,low}`
- 表里带 `model`（注册名）和 `model_label`（图例名）两列。柱状图数据是「模型 × 细胞系」宽表，5 个细胞系都给全（主图只画 HCT116 / A549）；曲线把两个 held-out 细胞系叠在一张表里用 `cell_type` 区分
- **改了 `results/` 就必须重跑这个脚本，图才会更新**
- 不含 Castillo：fig5 的口径和指标集跟 Gosai 不一样，由 `12_eval_castillo` 自己写表
- ⚠️ `results/figure_metrics/` 里还躺着 `fig4c_castillo_pearson.tsv`、`fig4d_castillo_cts_{high,low}.tsv` 三个文件，是分位数口径时代的残留，**已经没有 writer 也没有 reader**，别拿它们对数字

### `12_eval_castillo.py`
Castillo 零样本评估，**整张 fig5 的唯一计算入口**。源自师兄 C.Z. 的 `castillo_final_analysis/castillo_final_analysis.py`，指标定义一字未改，只是把绘图拆到 `plot/fig5`。

- **读**：`results/predictions/castillo_{dhs64,linear_ag_dnase,vef_only,epicast_ag_vef}.tsv`（所以必须先跑 `14`）
- **写**：`results/castillo/castillo_{regression,classification}_metrics.csv`、`castillo_cts_counts.csv`
- 8,152 条 genomic + synthetic **合起来算**，不分组、不划分 train/test —— 没有模型见过这个数据集，每条都是 held-out
- 实测值和预测值都用**原始值，不做任何 normalization**。Castillo 活性从没 z-score 过，所以 MAE/RMSE 是原始量纲，只能模型之间比，不能跟 Gosai 的数字比
- **CTS 口径**：`CTS-high = 目标 - max(其余6个) ≥ 1`，`CTS-low = min(其余6个) - 目标 ≥ 1`。绝对差值，不是分位数；阈值在 `config.castillo_cts_gap`。union = 3,333 / 8,152
- **residual**：目标 - 全部 7 个细胞系均值。这只用于 residual 回归那一列，**跟 CTS 标签的 max/min gap 不是同一个量**
- 每个模型用**自己预测值之间的 gap** 排序，方向和被评的标签一致
- `normalized_auprc = (auprc - prevalence) / (1 - prevalence)`，因为各细胞系 CTS 数量差很多，原始 AUPRC 的随机基线不一样，不能放同一个 boxplot 里比
- ⚠️ 与 `castillo_final_analysis/` 里的 CSV **数字不同**：那是 8月5日交接包（变体 B 修正前）的结果，DHS64 一致、三个 AG 系模型都变了

---

## 旁支

### `15_eval_mingap_comparison.py`（旁支）
比较两种 CTS 定义：本文的「减训练细胞系均值」vs Gosai et al. 原文的「min-gap」。

- **写**：`results/mingap/mingap_vs_mean_all_models.csv` + 8 张宽表（`pcc_activity`、`pcc_specificity`、`auroc_high`、`auprc_high`、`ef_high`、`auroc_low`、`auprc_low`、`ef_low`）
- min-gap 定义见 `utils.mingap_scores` 的 docstring：`high = A_c - max(其他参考细胞系)`，正值保证该元素确实是所比较细胞系中最活跃的
- ⚠️ 与 `15_export_figure_metrics.py` **编号相同但无关**，且**没有 plot 脚本消费它**

### `16_eval_vef_variant_b2.py`（旁支）
AlphaGenome VEF 预处理变体的敏感性分析。**它的 docstring 是变体 A/B/B2/C 命名的权威来源。**

- **读**：`mpra_path`、`alphagenome_vef/gosai_ag_pred_760k_pad_0.h5`、`analysis_gosai_0722_backup/results/vef_only/ag_vef/`（旧预测）
- **写**：`data/gosai_mpra/gosai_mpra_760679_ag_vef_x10_log1p_128bp.tsv`、`results/vef_variant_b2/{linear,xgb,mlp}_pred.npy`、`variant_comparison.csv`
- B2 单独隔离「DNase 从哪个头读出」这一个因素，因为它对每个 assay 的变换与 B 完全相同。B 自己的数字来自更早一次 run（预测文件已被 C 覆盖），在 summary 里是**引用而非重算**的
- ⚠️ 无 plot 消费

---

## `deprecated/analysis/`

- `_13_predict_known_promoters.py` —— 曾产出 `results/gene_therapy_promoters/`，已无引用
- `_12_eval_castillo_mpra.py`、`_13_eval_castillo_classification.py` —— 我原先的 Castillo 方案，用 z-score + 分位数口径定 CTS（10 个细胞系定标签、7 个做评估）。已被师兄的 `12_eval_castillo.py` 取代，产物在 `results/_castillo_percentile_cts_deprecated/`
