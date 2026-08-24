# EpiCast 项目导览（给新 agent）

这份文档的目的：让一个刚接手的 agent 在 5 分钟内知道**这个项目在做什么、能跑的代码在哪、结果落在哪**，以及**哪些目录是历史包袱，不要碰**。

论文级别的分析细节在 `paper/README.md`，这里只讲仓库全局。

---

## 一、项目在做什么

**一句话**：用预训练基因组序列模型「凭空造出」某个细胞系的表观遗传特征，再拿这些特征去预测同一条 DNA 序列在该细胞系里的**质粒（episomal）报告基因活性**，从而在**没有 MPRA 实测数据的细胞系**上做零样本预测。

核心概念：

- **VEF（virtual epigenomic feature，虚拟表观遗传特征）**：把 Sei / Enformer / Borzoi / AlphaGenome 这类 sequence-to-function 模型的输出，按「细胞系 × assay」聚合成标量。本项目固定用 4 个 assay：`DNase`、`H3K4me3`、`H3K27ac`、`CTCF`，所以每条序列在每个细胞系里是一个 **4 维向量**。
  - VEF 不代表质粒上真实形成的染色质状态，只是「基因组模型认为这个细胞系会怎么解读这段序列」的计算代理。
- **EpiCast**：序列分支（6 层残差 CNN）+ VEF 分支，用 **FiLM**（feature-wise linear modulation）把 VEF 注入序列特征，输出该细胞系的活性标量。
  - 关键设计：模型里**没有** cell-type embedding，也**没有** per-cell-type 输出头。细胞系身份只通过 VEF 向量进入网络，所以换一个训练时没见过的细胞系，不需要新增任何参数。
- **CTS（cell-type-specific）CRE**：本项目真正关心的对象。活性在细胞系之间高度相关，所以要先减掉共享成分（residual / gap），再看两端尾部。

科学结论一句话：**序列决定「共享的调控潜力」，VEF 决定「细胞特异的那部分差异」**。所以在全集上 EpiCast 和纯序列模型打平，但在 CTS 子集上 EpiCast 明显更好。

论文手稿：`paper/manuscript/epicast_0817-chx.docx`（标题 *Virtual epigenomic features enable prediction of cell-context-dependent episomal regulatory activity*）。**手稿正文是旧版本，代码是新版本，两者有不一致，以代码为准**（已知不一致清单见 `paper/README.md` §7.1）。

---

## 二、环境

```bash
conda activate torch          # 当前使用的环境，epicast 已 pip install -e . 装进去
python -c "import epicast"    # 应指向 /home/hxcai/EpiCast/src/epicast/__init__.py
```

跑任何 Python 命令前先激活 `torch` 环境。其他环境（`ag`、`castillo`）是给 AlphaGenome 官方 API 和 Castillo 相关代码单独准备的。

包定义在 `pyproject.toml`（包名 `epicast`，源码目录 `src/`），依赖清单在 `requirements.txt`。

---

## 三、目录地图

### 3.1 活跃 / 重要（会改动、要看）

| 路径 | 是什么 |
|---|---|
| `paper/` | **论文的全部分析代码和结果**。新 agent 90% 的时间在这里。详见其 `README.md`。8月24日由 `analysis_gosai_0722/` 改名而来，8月24日之前的提交和旧文档里都是旧名 |
| `src/epicast/` | 训练/推理的 Python 包：模型、数据集、Trainer、损失、指标、工具 |
| `configs/` | 训练与推理的 YAML 配置（57 个活跃 + `_archive/`） |
| `saved/` | 训练产物：checkpoint、日志、预测 `.npy` |
| `data/` | 所有数据集。本项目只关心 `data/gosai_mpra/` 和 `data/castillo_mpra/` |
| `scripts/train.py`, `scripts/predict.py` | 训练与推理入口 |
| `alphagenome_vef/` | 用 AlphaGenome 跑预测的脚本，产出 `gosai_ag_pred_760k_pad_0.h5` 和 `metadata_padded.tsv` 这两个**上游 h5 与元数据**。VEF 矩阵本身由 `paper/analysis/02_extract_ag_vef.py` 从这里的 h5 提取 |
| `README.md`（根目录） | 面向 GitHub 公开发布的英文 README，讲的是 `predict_CRE_activity/` 那套「给用户用」的三步预测流程，**不是论文分析流程**。内容已略微过时 |

### 3.2 支撑 / 偶尔要看

| 路径 | 是什么 |
|---|---|
| `predict_CRE_activity/` | 面向用户的端到端预测 pipeline（Sei feature → Sei VEF → EpiCast 活性），根 README 的示例就是它 |
| `alphagenome-pytorch/` | AlphaGenome 的 PyTorch 复现，本地推理用 |
| `alphagenome_research/` | Google 官方 AlphaGenome 代码（vendored） |
| `enhancer-design/` | Castillo 的 DHS64 模型与设计数据，提供 `castillo_dhs64_pred_merged.tsv` 这个 baseline |
| `pretrained_models/` | enformer / basenji2 / malinois 权重 |
| `process_raw_data/` | 各 MPRA 数据集的原始下载与预处理 notebook |
| `server.py` | 一个串起 `predict_CRE_activity/` 三步的演示脚本，不是 HTTP 服务 |

### 3.3 不要碰（历史包袱 / 已废弃）

- `analysis_gosai_0722_backup/`、`analyze_gosai_backup/` — 分析目录的旧快照，名字里的 `analysis_gosai_0722` 是 `paper/` 的旧名，**这两个目录刻意没跟着改名**。只有一个用途：`paper/analysis/16_eval_vef_variant_b2.py` 会读 `analysis_gosai_0722_backup/results/vef_only/ag_vef/` 里的旧预测做变体对比。
- `paper/castillo_final_analysis/` — 师兄 Castillo 方案（Fig 5）的原始单文件版本，**已整合进主线**（`analysis/12_eval_castillo.py` + `plot/fig5_castillo_metrics.py`），这里只作为出处保留。其中 `数据处理与分析流程.md` 是方法说明的最佳来源，值得读；但三个 CSV 和 PDF 是 8月5日旧预测算的，**数字已过期**。
- `paper/deprecated/` — 已弃用的 analysis 与 plot 脚本，以及 `share_20250805/`（旧的一次性交接包，README 里的数值表已过期）。
- `_analyze_castillo/`、`_analyze_epi_features/`、`_analyze_martinez_mpra/`、`_analyze_siraj_mpra/`、`_padding_effect/` — 下划线前缀 = 一次性探索，已 gitignore。
- `predict_CRE_activity_new/`、`TFBS_based_models/`、`TFBU/`、`outputs/`、`src/_archive/genoml/`（`epicast` 包的旧名镜像）。

### 3.4 Git 跟踪情况

`.gitignore` 把**所有数据类扩展名**（`*.tsv *.npy *.h5 *.pth *.csv *.pdf *.yaml` …）和**所有产物目录**（`data/ saved/ results/ figures/ manuscript/`）都忽略了，另外 `_*`、`*_backup/`、`deprecated/` 也忽略。

两个后果，写代码时要注意：

1. **`configs/*.yaml` 一个都不被跟踪。** 早期有 16 个 1–2 月的旧配置误入库，已于 8月21日移出索引，现在 `git ls-files '*.yaml'` 为空。所以**改配置不会进 diff，clone 下来也拿不到任何配置**——训练配置必须在文档里写清楚（见 §4.3 的表），发布时再决定要不要 `git add -f`。
2. **仓库里能提交的基本只有 `.py` 和 `.md`**。所以文档必须写清「哪个文件由哪个脚本生成」，否则 clone 下来无从复现。

---

## 四、训练与推理

### 4.1 模型

`src/epicast/models/` 下两个主角：

| 类 | 文件 | 用途 |
|---|---|---|
| `ConvFiLMNet` | `conv_film_net.py` | **EpiCast 本体**。6×`ResConvBlock`(256ch, k=3, `resnet_v2`) + MaxPool(2) + Dropout(0.2) → `FiLM(VEF)` → GAP → Linear(1024) + ReLU + Dropout(0.5) → Linear(1) |
| `ConvNet` | `conv_net.py` | **纯序列 baseline**。同样的 conv 栈，无 FiLM，`output_dim` = 训练细胞系数（3 或 5） |

`ConvFiLMNet.forward` 的关键行为：当 `feature` 是 3 维 `(B, n_cell_types, 4)` 时，**同一份 conv 特征图被复用**，对每个细胞系分别做一次 FiLM + head，`torch.stack` 成 `(B, n_cell_types)`。这就是「一套权重服务任意细胞系」的实现。

`FiLM`（`models/film.py`）的线性层权重和 bias 初始化为 0，所以训练起点 γ=β=0，调制退化为恒等，避免未训练的条件通路一开始就破坏序列特征的尺度。

其他模型（`ConvFusionTransNet`、`ConvTransformer`、`Malinois`、`ConvFiLMNet2` …）是对照实验用的，论文没用。

### 4.2 训练

```bash
conda activate torch
python scripts/train.py -c configs/0821_gosai_ag_vef_x10_log1p_dnase1_256.yaml
```

`utils.load_config` 读 YAML → `utils.process_config` 创建 `saved/<config 文件名>/<MMDD_HHMMSS>/` 并把配置副本和日志写进去 → `utils.init_obj` 按 `{type, args}` 反射实例化每个组件 → `Trainer.train()`。

固定的训练设置（写在每个 config 里）：`MaskedMSELoss`（缺失标签不产生梯度）、`AdamW`(lr 1e-3, wd 1e-4)、`WarmupCosineAnnealingWarmRestarts`(warmup 10, T_0 10, T_mult 2)、batch 4096、grad clip 1.0、`max_epochs 1000`、`EarlyStopping` 监控**验证染色体上 3 个训练细胞系的平均 Pearson**、patience 40、`seed: 0`。

模型选择只用到训练细胞系和 train/val 染色体，所以 held-out 细胞系（HCT116/A549）和 test 染色体（chr7/chr13）不参与任何决策。

### 4.3 config 命名约定

```
<MMDD>_<数据集>_<VEF 源>_<变换>_<宽度>[_<细胞系数>].yaml
```

例：`0821_gosai_ag_vef_x10_log1p_dnase1_256` = 8月21日 / Gosai / AlphaGenome VEF / `log1p(10x)` 且 DNase 取 1bp head / conv 256 通道。

论文用到的 4 个训练 config：

| config | 模型 | 对应论文里的 |
|---|---|---|
| `0821_gosai_ag_vef_x10_log1p_dnase1_256` | `ConvFiLMNet` | EpiCast-AlphaGenome |
| `0722_gosai_sei_vef_log1p_256` | `ConvFiLMNet` | EpiCast-Sei |
| `0722_gosai_seq_only_256` | `ConvNet`(3 头) | Seq-only（3 训练细胞系） |
| `0722_gosai_seq_only_256_5` | `ConvNet`(5 头) | Seq-only（全 5 细胞系，上界参考） |

另有一类**只含 `total_dataset` 块的推理专用 config**，给 Castillo 换数据集用，例如 `0821_castillo_dataset_N_dnase1.yaml`（`pad: true, padded_len: 200`，把 145bp 序列用 N 补到 200bp）。

### 4.4 saved/ 布局

```
saved/<config 名>/<MMDD_HHMMSS>/
├── config.yaml                    # 本次运行的完整配置快照
├── info.log / debug.log
├── checkpoints/{best,last}.pth
├── preds.npy                      # Gosai 全 760,679 行 × 5 细胞系
└── castillo_preds_pad_N.npy       # Castillo 8,152 行 × 7 细胞系
```

`config.py` 里的 `latest_run()` 取某个 config 下**最新一个已经有 `preds.npy` 的 run**，所以重跑训练后分析脚本会自动切到新 run。论文当前用的两个 run：

- `saved/0821_gosai_ag_vef_x10_log1p_dnase1_256/0820_155453/`（EpiCast-AlphaGenome）
- `saved/0722_gosai_sei_vef_log1p_256/0723_031345/`（EpiCast-Sei）

时间戳目录名（`0820_155453`）比 config 文件名里的日期（`0821_`）早一天，是正常的：run 目录名来自实际开跑时间，而 config 文件名的日期前缀是手写的，写的时候多写了一天。config 文件本身的 mtime 是 8月20日 15:54，与 run 目录严格对得上。

### 4.5 推理

分析包里的 `paper/analysis/06_infer_trained_model.py` 是实际用的推理入口（需要 GPU），它可以用 `-dc` 换一个 dataset config 从而对别的数据集推理：

```bash
python paper/analysis/06_infer_trained_model.py \
  -c saved/0821_gosai_ag_vef_x10_log1p_dnase1_256/0820_155453/config.yaml \
  -dc configs/0821_castillo_dataset_N_dnase1.yaml \
  -o castillo_preds_pad_N.npy
```

⚠️ dataset config 里带的是 **AlphaGenome** VEF 矩阵，所以只能配 EpiCast-AlphaGenome 的 checkpoint。没有任何 Castillo 配置指向 Sei VEF（`data/castillo_mpra/sei_vef.tsv` 在，但没有配置用它），拿这个 config 去配 Sei checkpoint 等于喂错 VEF。

`scripts/predict.py` 是更早的通用推理 CLI，功能重叠。

---

## 五、数据

### 5.1 Gosai MPRA（主数据集，`data/gosai_mpra/`）

760,679 条 200bp CRE × 5 个细胞系，来自 Gosai et al. 2024（ENCODE）。

- `gosai_mpra_760679_zscore.tsv` — 训练主表。列：`seq id chr pos ref_allele alt_allele allele OL data_project K562 HepG2 SK-N-SH HCT116 A549`
- `gosai_mpra_760679_ag_vef_x10_log1p_dnase1.tsv` — **论文用的 AlphaGenome VEF**，20 列 = 5 细胞系 × 4 assay
- `gosai_mpra_760679_sei_vef_logit.tsv` — 论文用的 Sei VEF
- 其余 `*_vef_*.tsv` 是各种预处理变体，留作敏感性分析

两个正交的划分（**极易混淆，务必分清**）：

| 维度 | 划分 | 含义 |
|---|---|---|
| 序列（按染色体） | `train` 其余 / `val` chr19,21,X / `test` chr7,13 | 泛化到新序列 |
| 细胞系 | `train` K562, HepG2, SK-N-SH / `test` HCT116, A549 | 泛化到新细胞环境 |

标签是 z-score 后的 MPRA log2 fold change，**均值方差只用 train 染色体估计**。HCT116（58.9%）和 A549（42.0%）有大量缺失，未插补。

### 5.2 Castillo MPRA（外部验证，`data/castillo_mpra/`）

8,152 条 145bp CRE × 10 个细胞系，来自 Castillo-Hair et al. 2025。其中 1,836 条基因组来源 + 6,316 条用 DHS64 模型设计的合成序列。

- `castillo_mpra_data.tsv` — 列：`id seq category source target` + 10 个细胞系名（**原始值，未 z-score**）
- `castillo_mpra_ag_vef_x10_log1p_dnase1.tsv` — 28 列 = 7 细胞系 × 4 assay

10 个细胞系里只有 7 个能匹配到 AlphaGenome biosample，故只在这 7 个上评估：`K562, HepG2, SK-N-SH, GM12878, WERI-Rb-1, MCF-7, HeLa-S3`。其中前 3 个在 Gosai 训练里见过，后 4 个完全没见过。

**这个数据集全程零样本，没有任何训练或调参。**

### 5.3 AlphaGenome VEF 的预处理变体（重要）

权威定义在 `paper/utils.py` 的 `ag_variants` 字典里（`ag_default_variant = "b"`），两个 extract 脚本共用它，所以变体不会被改单个脚本悄悄重定义：

| 变体 | CTCF 列 | DNase 读出 | 变换 |
|---|---|---|---|
| A | **错**（读到了邻近的另一个 TF） | 1bp head | `log1p(10x)`，128bp head 先除 bin 宽 |
| **B** | 已修正 | 1bp head | 同 A ← **论文当前用的就是 B** |
| B2 | 已修正 | 128bp head | `log1p(10x/128)` |
| C | 已修正 | 128bp head | `log1p(x)`，不缩放 |

`config.py` 里 `vef_paths["alphagenome_prefix"]` 指向的 `*_ag_vef_x10_log1p.tsv` 就是变体 A，只留作修正前后的对比。变体 A 已经没有生成路径了（`ag_variants` 里不含它），它就是修正 CTCF 列之前的历史产物。

换变体重跑不需要改代码，只要在 extract 脚本里传 `variant=`；文件名后缀由 `ag_variants[variant]["suffix"]` 决定，所以矩阵和它的预处理口径不会对不上。

---

## 六、给 agent 的几条硬约束

1. **先激活 `conda activate torch`**，再跑任何 Python。
2. **谨慎修改和删除现有文件**，改动要最小化，保持现有代码风格。
3. **脚本里的常量用小写**（模块/包里的常量可以大写）——这是本仓库的既有约定，`config.py`、`utils.py` 都遵循。
4. **代码要简洁直白**：不加多余的类型检查、形状检查、防御性分支和 fallback。信任内部数据契约，让非法数据自然报错。不要为了「更通用」而引入抽象层。
5. **区分两个 train/test**：染色体划分 vs 细胞系划分，正交，写代码和写文档时都不要混。
6. **HCT116 / A549 有缺失**。凡涉及这两个细胞系的指标、百分位、排序，都必须先按真值 `notna()` 过滤，并且**在过滤后的子集内**算百分位（即「子宇宙」原则）。在全 760,679 行上取百分位再套到子集是一个已知的错误来源。
7. **Gosai 的 residual 参考系永远是 3 个训练细胞系的均值**，不是全部 5 个。**Castillo 不一样**：它的 residual 参考系是全部 7 个细胞系，CTS 标签用的是绝对差值 gap ≥ 1 而不是分位数。两个数据集口径不同是刻意的，别统一。
8. **论文手稿有已知错误，暂不处理**。发现代码与手稿不一致时，以代码为准，记录下来，不要擅自改代码去迁就手稿。
9. **改了 `results/` 里的东西，要重跑对应的导出脚本**：`analysis/14_export_prediction_tables.py`（预测表）、`analysis/15_export_figure_metrics.py`（fig2/fig3 指标表）、`analysis/12_eval_castillo.py`（fig5 指标表），否则图不会更新。绘图脚本一律不直接读 `saved/` 里的 npy。
