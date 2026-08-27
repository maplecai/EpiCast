# plot/ — 绘图脚本逐个参考

背景见 `../README.md`，数据来源见 `../analysis/README.md`。

## 命名与前缀约定（2026-08-24 起）

脚本名和输出 PDF 名里的编号**一律指手稿 `manuscript/epicast_0817-chx.docx` 的 panel 编号**。逐 panel 的权威描述在 `manuscript/epicast_figure_plot_descriptions.md`：布局、坐标轴、模型顺序、参考线都以它为准。

`_` 前缀 = **手稿里没有对应 panel，或内容还没按手稿重做**，不进论文：

| 文件 | 为什么带 `_` |
|---|---|
| `_fig1c_vef_coverage_bars.py` | assay 覆盖柱图，手稿 Fig 1B 是示意图，不用它 |
| `_fig3cd_cts_classification.py` | ROC/PR 曲线，手稿 3B/3C 用的是柱状图 |
| `_fig4bde_vef_partial_correlation.py` | 只覆盖 CTCF 的旧 Fig 4B/4D/4E，已被同名不带 `_` 的脚本取代 |
| `_fig5_castillo_metrics.py` | 五 panel 的旧 Fig 5，CTS-low 已由不带 `_` 的脚本重新画出，这份没有用途了 |

`../figures/` 里同理：不带 `_` 的 PDF 是当前定稿产物，带 `_` 的是改名前那一版或上面几个脚本的输出。`_fig5_castillo_metrics.py` 的输出名已经全部加了 `_` 前缀，不会覆盖现行 Fig 5。

`_bar` / `_box` / `_boxplot` 后缀（都带 `_` 前缀）= fig1c / fig4bde / fig5 摘要画法的三个落选版本：中位数横线、无须箱线图、常规箱线图。手稿最终采用的是 mean ± SD 那一版，它已经改名占用了不带后缀的正式脚本名，legend 也由它输出；归档脚本的输出名全部加了 `_` 前缀，不会覆盖定稿产物。

**一个 panel 一个 PDF**，figure 级的拼版在 Illustrator 里做。多 panel 的网格（如 2B 是 4 行 × 2 列）算一个 panel，出一个 PDF。图例和 colorbar 一律单独出文件。

## 通用约定

- 用 `sys.path.insert` 定位 `config`，从项目根或 `paper/` 都能跑，无 CLI 参数。
- **每个脚本的 `main()` 开头必须是这三行**，少一行字体就会不一样：

  ```python
  set_mpl_params()                                # Arial + seaborn talk 主题
  sns.set_theme(style="white", context="talk")    # 这一步会把字体重置回 DejaVu Sans
  plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})   # 所以要再设回来
  ```

  第二行是全仓库唯一控制字号的地方；代价是 seaborn 会顺手把 `font.family` 清掉，所以第三行不能省。fig5 曾经漏掉整段，结果全图是 DejaVu Sans，和其余十张图对不上。
- **字号只由 `context="talk"` 决定，脚本里不写任何 `fontsize=` / `labelsize=`**（2026-08-25 起，此前是逐处写死 13pt 标签 / 11pt 刻度）。理由：每张图在正文里都是某个 figure 的一个 panel，会被缩小，所以字必须从一开始就偏大；而一处一处写死的字号最终必然对不齐。
  想让某张图的字看起来更小更精细，**唯一的手段是把 `figsize` 调大**（图变大，字相对变小）；反过来想让字更显眼就把 `figsize` 调小。
- **`figsize` 取整数，且高度按「单个子图高 6」这个基准来**：拼版时不用缩放就能直接拼，字号也自动对齐。宽度按内容给，一行 k 个子图就是 `k × 单图宽`；上下两排子图就是高 12。当前一览：

  | 高 6（单排） | 高 12（两排） |
  |---|---|
  | fig1c `(16,6)`、fig1e `(24,6)`、fig1df `(6,6)`、fig3bc `(8,6)`、fig3de `(8,6)`、fig3fg `(24,6)`、fig4ac `(12,6)`、fig4bd `(32,6)`、fig5a-c `(8,6)` | fig2a `(6,12)`、fig2bc `(8,12)`、fig3a `(8,12)`、fig4e `(12,12)`、fig5de `(8,12)` |

  ⚠️ **`figsize / 子图个数` 不等于单个子图的长宽比**，差的那一截是 `wspace` / `hspace`。边距（0.15/0.15/0.9/0.9）横竖都是 0.75，会等比缩掉两个方向，不影响比例；但 `wspace` 是按**子图宽度**算的间隙，k 个子图 + (k−1) 个间隙意味着单个子图只拿到 `1 / (k + (k−1)·wspace)` 的宽度，而不是 `1/k`。
  例：fig3fg 是 4 个子图、`wspace=0.3`，所以每个子图拿到 `1/(4+3×0.3) = 1/4.9` 的宽度。想让单图正好 4:6，宽度得是 `4 × 4.9 = 19.6`；现在的 `(24,6)` 实际是 4.9×4.5 英寸。要精确控制就把 `wspace` 一起算进去。

- **colorbar 和 legend 一律单独出文件**，不占正图的版面。colorbar 的画法：一张 `(2,6)` 的空图，里面放一条高 `0.6` 的 axes，于是在「单图高 6」的基准下 colorbar 长度正好是热图边长的 0.6 倍。fig1D/1F 共用一条（`fig1df_colorbar.pdf`），fig4A/4C 共用一条（`fig4ac_colorbar.pdf`）。
  分块 legend（fig2bc / fig3a / fig3bc：一块一个 VEF 来源，块头是来源名）是一张 `(4,6)` 的图、一块一个 axes，**每个 axes 的高按自己的条目数分配**（`height_ratios` 用 `模型数 + 1`，那个 1 是块头）。等高会让条目多的那一块溢出自己的 axes、压到下一块的块头上。
- **边距不用调**：所有脚本的 `left/bottom/right/top` 统一写 `0.15 / 0.15 / 0.9 / 0.9`，因为存图一律带 `bbox_inches="tight"`，画布多出来或者标签戳出画布都会被裁剪框吸收。真正需要按图调的只有 `wspace` / `hspace` —— tight 裁的是外框，管不了 panel 之间的挤压。
- **每个脚本自己组织自己的版式**，不抽公共绘图模块。柱图的位置计算、断轴、图例这些代码在几个脚本里是重复的，这是有意的：改一张图不应该动到别的图。
- 从 `config` 取的只有**内容**，不是版式：模型清单与顺序 `figure_model_blocks`、细胞系配色 `cell_colors` / `castillo_cell_colors`。这样同一个模型/细胞系在不同图里颜色一致。
- **颜色编码「这张图里变化的那个维度」，同一个对象在哪张图里都是同一个 hex**（2026-08-25 对齐）：
  - 一张图里画多个模型 → 颜色表示模型（fig2bc / 3a / 3bc / 3de），从 `figure_model_blocks` 的 colormap 推导：Sei 是 YlOrRd 浅→深，AlphaGenome 是 GnBu 浅→深，不用 VEF 的 sequence-only 是灰（`config.seq_only_grey` = `#8C8C8C`）。sequence-only 这个 block 只有一个模型，按 `linspace(0.28, 0.92, 1)` 会取到色阶的浅端 0.28，而灰阶的 0.28 是 `#5A5A5A` 这种偏深的灰，所以这一 block 用一条**恒定灰**的 colormap（`config.seq_only_cmap`）把颜色直接定死。
  - 一张图里只有一个模型 → 颜色让给细胞系（fig1c / fig2a / fig3fg 用 `cell_colors`，fig5 用 `castillo_cell_colors`）。fig1e 是唯一例外，它按 matched / unmatched 上黑灰，这是手稿 caption 的口径。
  - **细胞系配色按 `config.cell_types` 的顺序取蓝黄绿红紫**（2026-08-25 C.Z. 定）：K562 `#3B75AF`、HepG2 `#E6AB02`、SK-N-SH `#2E9E5B`、HCT116 `#D73027`、A549 `#B294CC`。旧配色是红橙金绿蓝，散点大小下 HepG2 的橙和 SK-N-SH 的黄分不出来，所以橙整条去掉。紫比另外四个**更浅、饱和度更低**：同饱和度的紫看上去比它们暗得多，而深紫在散点尺寸下又会和蓝糊在一起。Castillo 七个细胞系在这五个后面接棕 `#8C564B` 和青 `#17A2B8`（先试过桃红，夹在红和紫之间反而更难认）。
  - `config.model_styles` 里逐模型写死的 hex **和 colormap 那套逐个对得上**（EpiCast-AlphaGenome 两边都是 `#08599C`）。它现在只有归档的 `_fig*` 在读，但改了 colormap 要同步，反过来也一样。
  - `castillo_model_styles` 里三个和 Gosai 同名的模型沿用同一个 hex（`vef_only` = AlphaGenome MLP），DHS64 只用序列所以是灰。活跃的 fig5 只取它的名字，不取颜色。
- 存图统一 `dpi=400` + `bbox_inches="tight"`，建图统一 `dpi=100`。
- **刻度线要显式打开**：seaborn `style="white"` 把 `xtick.bottom` / `ytick.left` 都设成 `False`，所以每个有数值刻度的轴都要写一句 `ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5)`。柱图（fig2bc / fig3a / fig3bc）的 x 轴没有刻度（模型身份靠 legend），所以只开 y。`length=3.5` 是覆盖 talk 主题的 9.0，全仓库一致。
- **非数据的线一律 1pt，只有颜色区分**（2026-08-25 统一）：
  - **框线纯黑** —— 子图边框、刻度线在 `set_mpl_params()` 里由 `axes.edgecolor/linewidth`、`?tick.color`、`?tick.major.width` 统一设定（覆盖 seaborn `style="white"` 的 `.15` 深灰），所以脚本里不用再写。热图的三角外框和 colorbar 外框是手画的 patch，也写 `lw=1.0`。热图内部的格线是数据分隔不是框线，仍是 `lw=0.5` 的灰。
  - **参考线灰色虚线**（`color="gray", lw=1.0, linestyle="--"`）—— y=0、随机基线（AUROC 0.5 / prevalence / EF 1）、fig2a 的 y=x 对角线全都用这一档。fig3de legend 里的 "Random" 图例句柄要跟着改，否则和图里的线不一样粗。
- **小样本的摘要画 mean ± SD**（手稿定稿采用）。汇总细胞系分布的三张图（fig1c / fig4bde / fig5）统一：散点照旧全画，横线是 mean，上下各一条 1 倍样本 SD（`ddof=1`），不画箱体：

  ```python
  mean, sd = np.nanmean(values), np.nanstd(values, ddof=1)
  ax.vlines(x, mean - sd, mean + sd, color="black", lw=1.0, zorder=3)
  ax.hlines(mean, x - width / 2, x + width / 2, color="black", lw=1.5, zorder=3)
  for cap in (mean - sd, mean + sd):
      ax.hlines(cap, x - width / 4, x + width / 4, color="black", lw=1.0, zorder=3)
  ```

  mean 横线宽度沿用箱线图版的箱宽（fig1c / fig4bde 是 0.5，fig5 是 0.56），SD 的 cap 取一半宽。选它而不选箱线图的理由是 mean 和 SD 在 n=5 和 n=500 时定义完全一样；n=5 时箱线图的 Q1、Q3 正好落在第 2 和第 4 个点上，1.5×IQR 还会判出一堆离群细胞系。代价是默认了细胞系对称散开，而这恰恰是图上看不出来的。图注写 *bar, mean; whiskers, ± 1 s.d.; points, individual cell types*。

  三个落选画法归档为 `_figXX..._bar.py`（只有中位数横线）、`_figXX..._box.py`（有箱体无须线）、`_figXX..._boxplot.py`（常规箱线图，带须线和 cap），输出名同样带 `_` 前缀和对应后缀，不会覆盖定稿图。**改数据口径时只改现行脚本，归档那三份不用同步。**
- **两个 VEF 源同时出现时，一律 Sei 在前、AlphaGenome 在后**（2026-08-25 统一）：左右排就是左 Sei，上下排就是上 Sei，图例和柱子分组也是这个次序。手稿没规定这个顺序，是我们自己定的，所以唯一的要求是别自相矛盾。涉及 fig1c 的 panel 顺序、fig1e 的两组、fig2bc/3a/3bc 的 block（来自 `config.figure_model_blocks`）、fig3de 的曲线、fig3fg 的 panel 顺序、fig4ac 的左右热图、fig4bde 的两组 / 两列。fig4ac 原来是 AlphaGenome 在左，这次跟着改了
- **分类轴的刻度标签一律竖排**（`rotation=90`）：assay 名、细胞系名、模型名都是长字符串，斜排（`rotation=38, ha="right"`）在拼版缩放后更难对齐。
- **title 的分工**（2026-08-25 起）：**子子图的身份标签写进代码**，因为它是数据信息的一部分 —— 一格代表哪个细胞系、哪个模型，`ax.set_title` 直接写上（网格版式只写最上面一行）。留给拼版时手加的只有 panel 字母（A/B/C）和 `suptitle`；需要标注的 n、断轴的斜线标记同理，脚本只把 n 打到 stdout。
  分组表头分两种写法：能靠 panel title 的第二行带出来的就重复写（fig3fg 的模型名、fig4e 的 setting，同排两个 panel 重复，拼版时删掉多余的），一个表头要罩住四个 panel 的就画一个**居中组头**——在外层 gridspec 的整格上加一个 `set_axis_off()` 的空 Axes，用它的 `set_title(..., pad=40)` 顶在 panel title 上方（fig4bd 的 `Sei` / `AlphaGenome`）。
- 绘图代码保持简洁：不为了微调多写几十行，不用冷门 API（私有属性、自定义 marker 路径等），细节留给拼版阶段手动改。

```bash
conda activate torch
cd /home/hxcai/EpiCast
for f in paper/plot/fig*.py; do python "$f"; done   # 论文用图，约 2 分钟
```

## 输入来源分三类

| 来源 | 脚本 | 说明 |
|---|---|---|
| **`results/figure_metrics/`** | fig2bc, fig3a, fig3bc, fig3de | 一次评估一个 tsv。**改了 `results/` 后必须先重跑 `analysis/15_export_figure_metrics.py`** |
| **`results/predictions/`** | fig2a, fig3fg | 要逐序列的实测值和预测值。**改了预测后必须先重跑 `analysis/14_export_prediction_tables.py`** |
| **`results/castillo/`** | fig5 | 由 `analysis/12_eval_castillo.py` 写 |
| **原始 MPRA 表 + VEF 矩阵** | fig1c, fig1df, fig1e | 这几张要逐序列的 VEF 值，聚合成指标表反而丢信息，所以直连原始数据 |

`results/figure_metrics/` 里的文件名**不带图号**（`activity_test.tsv` 而不是 `fig2c_...`），因为手稿重新编号时不该牵动中间数据。对应关系写在 `analysis/15_export_figure_metrics.py` 的 docstring 里。

共同点是**没有一个绘图脚本直接读 `saved/` 里的 npy**：npy 不带 key，全靠行序对齐 MPRA 表，让每个脚本各自重推一遍这个对齐关系迟早出错。

---

## Fig 1 — VEF 携带细胞环境信息

Fig 1A、1B、1G 是示意图，不由脚本生成。

### `fig1c_vef_activity_correlation.py` → 1C
4 个模型 panel 并排（Enformer / Borzoi / Sei / AlphaGenome），共用 y 轴，panel title 就是模型名；每个 panel 横轴是 4 个 assay，每个 x 位置上 5 个细胞系的散点 + 一个常规箱线图。

- **读**：`config.mpra_path` + `config.vef_paths`
- **写**：`figures/fig1c_vef_activity_correlation.pdf`、`fig1c_legend.pdf`
- 每个点是 `PCC(该细胞该 assay 的 VEF, 该细胞的实测活性)`，成对完备（用 `Series.corr`，自动跳过缺失）
- 汇总的是**细胞系之间**的 PCC 分布，不是单条 CRE 的分布
- `figsize=(16, 6)`，单个 panel 4:6
- y 轴固定 0–0.7，只标 0 / 0.2 / 0.4 / 0.6，ylabel 是 `PCC`。数据最低 0.16，所以不画 y=0 的参考线；assay 名竖排（旋转 90°）
- 5 个点**不做横向抖动**，全落在中线上，所以互相遮挡就说明数值接近；靠白色描边区分重叠的点
- Enformer / Borzoi 在 K562、HepG2、A549 没有 H3K27ac track（列全是 NaN），这三个点直接缺失，不插补，中位数只用剩下 2 个点 —— 这也是论文最终只用 Sei 和 AlphaGenome 建 EpiCast 的原因
- `vef_paths` 里的 `alphagenome_prefix`（CTCF 修正前的变体 A）不进图

### `fig1df_activity_correlation_heatmap.py` → 1D + 1F
细胞系两两之间的实测活性相关性，全集和 CTS 集各一张 5×5 热图。

- **读**：`config.mpra_path`
- **写**：`figures/fig1d_activity_correlation_whole.pdf`、`fig1f_activity_correlation_cts.pdf`、`fig1df_colorbar.pdf`
- 两张热图都是 `(6,6)`，共用 0–1 色阶，所以 colorbar 只出一份、单独一个文件
- **热图样式（fig4ac 跟这套一致，但两个脚本各写一份，不抽公共函数）**：只画下三角、保留对角线的 1.00；格子里的字是黑色；colorbar 长 3.6 = 热图边长的 0.6 倍，只标 0 / 0.5 / 1；热图和 colorbar 都有黑色外框
- 格子边框是**手画的**（`linewidths=0` + 逐格 `Rectangle`）：seaborn 的 `mask` 会连空格子一起画灰色格线，右上角就脏了。黑外框是沿三角形阶梯边界的一条 `Polygon`
- 数字保留两位小数：三位的话在 talk 字号下会连成一片
- 相关性成对完备（`DataFrame.corr`），HCT116/A549 各自用它们与对方共有的 CRE
- CTS 子集用 `utils.build_masks` 的 `all_cts_1_99`，即 5 个细胞系 1%/99% 尾部的**并集**，42,703 条
- 两张对比是论文的关键论证：全集上 0.72–0.88（活性主要是共享的），CTS 集上掉到 0.48–0.79

### `fig1e_dnase_residual_specificity.py` → 1E
虚拟 DNase residual 与活性 residual 的相关性，看它是否认得自己的细胞。

- **读**：`config.mpra_path` + `config.vef_paths`
- **写**：`figures/fig1e_dnase_residual_specificity.pdf`、`fig1e_legend.pdf`
- **横排 10 个 panel、5 个一组共两组**（左 Sei、右 AlphaGenome），组间留空隙。`figsize=(24, 6)`，单个 panel 约 2.4:6
- 10 个 panel 共用同一条 y 轴，所以 y 刻度和 ylabel 只画在最左边那一格；第二组的开头如果也写 ylabel，会戳到第一组 A549 的头上
- x 轴两个刻度（Whole / CTS）离得近，所以显式打开 x 的 tick 短线（seaborn `style="white"` 默认把刻度线关掉了），靠它对应标签和位置
- panel title 是目标细胞系，两个组表头（Sei / AlphaGenome）拼版时手加
- 每个 panel 固定目标细胞 `t`，对 5 个 VEF 细胞环境 `s` 各画一条从 Whole 到 CTS 的连线；`s == t` 是 matched，黑色加粗，其余淡灰（caption 的 black/gray 口径）
- residual 一律以 K562/HepG2/SK-N-SH 均值为参考系，活性和 VEF 都是
- 看的是**每条线的走向**：CTS 集上 matched 线拉开，说明 VEF 带的是细胞环境信息而不只是序列强度

---

## Fig 2 — 跨细胞系的活性预测

### `fig2a_epicast_scatter.py` → 2A
EpiCast-AlphaGenome 预测值 vs 实测活性。

- **读**：`results/predictions/gosai_epicast_ag_vef.tsv`
- **写**：`figures/fig2a_epicast_ag_scatter.pdf`
- 上下两联，HCT116 在上、A549 在下，panel title 是细胞系名（细胞系名以前挂在 xlabel 上，现在按新约定挪到 title），只取 test 染色体且该细胞系有实测值的序列
- 只画一个模型，所以散点颜色让给细胞系（`cell_colors`，HCT116 红 / A549 紫，和 fig1c、fig3fg 里这两个细胞系同色）
- y 轴写 `Predicted activity`，模型名不进坐标轴（一张图只有一个模型，模型身份是 panel 级信息，拼版时手加）
- 共用坐标范围和 1:1 aspect，对角线是 `y = x`，右下角标注 PCC 和 n，**都从数据现算，不硬编码**
- 当前值：HCT116 `PCC=0.801, n=36,345`；A549 `PCC=0.698, n=27,973`。手稿写的是 0.799 / 0.694，差异来自 VEF 变体 B 的 CTCF 列修正，n 完全一致

### `fig2bc_activity_metrics.py` → 2B + 2C
全集和 CTS 集上的模型对比柱图。

- **读**：`results/figure_metrics/activity_test.tsv`、`activity_cts.tsv`
- **写**：`figures/fig2b_activity_whole.pdf`、`fig2c_activity_cts.pdf`、`fig2bc_legend.pdf`
- 每张 4 行（PCC / SCC / MAE / RMSE）× 2 列（HCT116 / A549），最上面一行的 title 是列对应的细胞系
- 11 个模型分三段：Sei 块 5 个、AlphaGenome 块 5 个、sequence-only 1 个，块之间留间隔
- **2B 和 2C 同一行共用 y 轴上限**，两张可以直接对比
- 手稿这两张没画随机基线，脚本也不画

---

## Fig 3 — 细胞特异元件的排序

### `fig3a_residual_metrics.py` → 3A
CTS 集内 residual 活性的模型对比。

- **读**：`results/figure_metrics/residual_cts.tsv`
- **写**：`figures/fig3a_residual_cts.pdf`、`fig3a_legend.pdf`
- 4 行 × 2 列，同 2B 的版式（含首行的细胞系 title），但**只有 10 个模型，不画 sequence-only**：held-out 细胞的 seq-only 预测被定义为三个训练细胞的均值，predicted residual 恒为 0，没有可相关的变化
- 模型清单取 `config.residual_model_blocks`

### `fig3bc_cts_prioritization.py` → 3B + 3C
CTS-high / CTS-low 的排序指标柱图。

- **读**：`results/figure_metrics/cts_high.tsv`、`cts_low.tsv`
- **写**：`figures/fig3b_cts_high.pdf`、`fig3c_cts_low.pdf`、`fig3bc_legend.pdf`
- 每张 2 行（AUROC / AUPRC）× 2 列，11 个模型，首行 title 是细胞系（断轴那格的 title 挂在上半段）
- AUROC 固定 0–1 轴、随机线 0.5；AUPRC **按细胞系各自缩放**，随机线是该细胞系的 positive prevalence（约 0.9%–1.2%，不是硬编码的 1%）
- CTS-high 的 A549 panel 用**断轴**：EpiCast-AlphaGenome 的 AUPRC 是 0.206，其余模型都 ≤0.076，不断轴会把所有柱子压平。断点写在脚本顶部 `panels` 的第三个元素里，改数据后要检查断点是否还落在空隙上（不能切到任何一根柱子）

### `fig3de_topk_retrieval.py` → 3D + 3E
top-k 检索曲线。

- **读**：`results/figure_metrics/retrieval_cts_high.tsv`、`retrieval_cts_low.tsv`
- **写**：`figures/fig3d_retrieval_cts_high.pdf`、`fig3e_retrieval_cts_low.pdf`、`fig3de_legend.pdf`
- 每张 2 行（Enrichment fold / NNS）× 2 列，首行 title 是细胞系
- x 仍然是对数轴（1e-4 到 1e-1），但**刻度写成指数本身**（−4/−3/−2/−1），xlabel 是 `log10(top-ranked fraction)`；比 `10^-4` 这种上标窄得多，也不用担心 mathtext 的字体和 Arial 对不上（所以 `log10` 是纯文本，不写 `$\log_{10}$`）
- **只画 EpiCast-Sei 和 EpiCast-AlphaGenome**，取各自 block 色阶最深的那一档，和 Fig 2/3 的柱子颜色对得上
- 随机期望：EF = 1，NNS = `1 / prevalence`（HCT116 约 105.7），从表里的 prevalence 现算
- 曲线不做平滑，小 k 处的抖动是离散筛选的真实表现

### `fig3fg_topk_activity_profile.py` → 3F / 3G
被排在最前面的 CRE，在各细胞系里的**实测**活性随筛选深度的变化。

- **读**：`results/predictions/gosai_epicast_{ag,sei}_vef.tsv`
- **写**：`figures/fig3f_topk_activity_cts_high.pdf`、`fig3g_topk_activity_cts_low.pdf`、`fig3fg_legend.pdf`
- **一个任务一张图**（原来是 8 个 panel 挤在一张，拆成 3F = CTS-high、3G = CTS-low），每张 4 个 panel，顺序是 Sei→HCT116、Sei→A549、AlphaGenome→HCT116、AlphaGenome→A549
- panel title 写两行「模型 + 目标细胞系」，所以哪两个 panel 属于同一个模型不依赖拼版时手加的模型头
- 两张图**共用一个 y 范围**（取两个任务全部曲线 mean ± SEM 的极值再放 0.1），拆成两个文件之后 `sharey` 管不到跨文件，所以显式算出来传进去
- x 轴口径同 3D/3E（指数刻度 + `log10(top-ranked fraction)`）。换成指数之后刻度标签短了很多，四个 decade 都能标开。x 轴标题用一个 `supxlabel` 而不是 4 遍 xlabel
- `figsize=(24, 6)`，`wspace=0.3` 之后单个子图实际是 4.9×4.5 英寸
- 每个 panel **4 条线**（3 个训练细胞系 + 该 panel 的目标细胞系），目标那条加粗，带 mean ± SEM 阴影；legend 是 5 色的，因为跨 panel 目标细胞不同
- **另一个 held-out 细胞系被排除**：两者在 test 染色体上的覆盖几乎不重叠。以 AlphaGenome 选 HCT116 的 top 0.1% 为例，36 条入选 CRE 里只有 2 条有 A549 实测值（top 1% 是 363 比 20），画出来是两三个点的均值和几百条的均值并排，会误导
- 排序宇宙是 test 染色体 ∩ 目标细胞有实测值，与 `analysis/09` 一致，所以深度刻度和 3D/3E 对得上
- 这是论文用来说明「EpiCast 不是简单地挑出普遍活跃/不活跃的元件」的那张图

---

## Fig 4 — VEF 编码了什么

五个 panel 都有活跃脚本了（4B/4D/4E 是 2026-08-25 补的）：

- 4A / 4C：4×4 的 VEF 两两相关热图（4C 是 residual 版），Sei 和 AlphaGenome 各一张下三角，格子里写 `mean ± SEM`（跨 5 个细胞系）→ `fig4ac_vef_correlation_heatmap.py`
- 4B / 4D：8 个 feature panel（4 assay × 2 个 VEF 源），每个 panel 两个 x 位置 Marginal 和 3 VEFs conditioned，5 个细胞系散点 + 常规箱线图 → `fig4bde_vef_partial_correlation.py`
- 4E：2 行（activity / residual）× 2 列（Sei / AlphaGenome），x 是 4 个 assay，y 是标准化回归系数 β → 同一个脚本

### `fig4ac_vef_correlation_heatmap.py` → 4A、4C
- **读**：`results/vef_pairwise_correlation/vef_pairwise_correlation.csv`（先跑 `analysis/11_vef_pairwise_correlation.py`）
- **写**：`figures/fig4a_vef_correlation.pdf`、`figures/fig4c_vef_residual_correlation.pdf`、`fig4ac_colorbar.pdf`
- 一个 PDF 里横向两张热图（`figsize=(12, 6)`，每张 6×6），**左 Sei、右 AlphaGenome**，panel title 就是 VEF 源名。手稿只说是 pairwise correlation，没写哪套矩阵对应哪个 VEF 源，所以这个左右顺序是我们自己定的，caption 要写一句
- assay 顺序 = `config.assays`（DNase / H3K4me3 / H3K27ac / CTCF）。**样式与 fig1df 完全对齐**：只画下三角、保留对角线、黑字、手画格子边框 + 阶梯黑外框、colorbar 单独出文件且只标 0 / 0.5 / 1。两个脚本各写一份这套代码，故意不抽公共函数
- 对角线是 VEF 跟自己相关，写成 `1.00` 不带 SEM；其余格子 `mean` 上、`± SEM` 下，跨 5 个细胞系汇总
- 四张热图共用 0–1 色标（所以 4A 和 4C 也共用一份 colorbar 文件）：绝对值 0.72–0.93，residual 0.32–0.82，共用色标才能直接看出去掉共享成分后共线性掉了多少。单个细胞系在 residual 下最低到 −0.26，但画出来的均值全是正的
- 之前试过「一个 4×4 方块、上下三角各放一套」，两套数值挤在一起读不出来，已废弃

### `fig4bde_vef_partial_correlation.py` → 4B、4D、4E
每个 VEF 在控制其余三个之后还剩多少与活性的关系。

- **读**：`results/vef_partial_correlation/vef_partial_correlation.csv`（先跑 `analysis/11_vef_partial_correlation.py`）
- **写**：`figures/fig4b_vef_partial_correlation.pdf`、`fig4d_vef_residual_partial_correlation.pdf`、`fig4e_vef_regression_beta.pdf`、`fig4bde_legend.pdf`
- 4B（absolute）和 4D（residual）是**一行 8 个 panel**，版式照 fig1e 那套嵌套 gridspec：外层 1×2 是两个 VEF 源（`wspace=0.25`，间距明显大于 assay 之间），内层 1×4 是四个 assay，八个 panel 共用 y 轴。`figsize=(32,6)`，单个 panel 实际约 2.05×4.5 英寸
- 每个 panel 两个 x 位置：`Marginal`（VEF 与活性的 PCC）和 `3 VEFs conditioned`（控制其余三个 VEF 后的偏相关）。x 标签竖排，标签比 panel 还宽所以必须竖
- 4E 是 2×2（`figsize=(12,12)`）：列是 Sei / AlphaGenome，行是 absolute / residual，x 是四个 assay，y 是标准化 β。ylabel 用真的 `β` 字符（Arial 有这个字形，不走 mathtext）
- 4E 的 y 轴**按行共用，不是四格共用**（`sharey="row"`）：absolute 的 β 落在 −0.59 到 0.83，residual 只有 −0.27 到 0.37，四格共用会把下面一排压成一条线。代价是上下两排不能直接比数值高低，但两排本来问的就不是同一个问题
- 三张图的 title 都是**两级**，都由脚本画：4B/4D 每个 panel 只写 assay，`Sei` / `AlphaGenome` 是罩住四个 panel 的居中组头（外层格上一个 `set_axis_off()` 的空 Axes，`set_title(pad=40)`）；4E 只有两列，罩不住谁，所以退回重复写 title——第一行是这一排回归的是什么（`Activity-VEF` / `Activity residual-VEF residual`），第二行是 VEF 源，同排重复，拼版时删掉多余的那个
- 三张图里**颜色都表示细胞系**（`cell_colors`），摘要是常规箱线图，和 fig1c / fig5 一致
- 前身是 `_fig4bde_vef_partial_correlation.py`（只画 CTCF，读老的 `ctcf_ablation.csv`），已归档

### `_fig4bde_vef_partial_correlation.py`（旧版，只有 CTCF）
- **读**：`results/ctcf_ablation/ctcf_ablation.csv` + `config.mpra_path` + `config.vef_paths`
- **写**：`figures/_fig4a_*`（K562 的 3 张散点）、`figures/_fig4b_*`（符号翻转 boxplot）
- ⚠️ 脚本内部的输出路径还是不带 `_` 的旧名，重跑会占用最终命名空间，跑完记得给产物补前缀。它读的那张 csv 也已经没有生成脚本了

---

## Fig 5 — Castillo 外部数据集零样本验证

### `fig5_castillo_metrics.py` → 5A–5E
指标表和 panel 版式源自师兄 C.Z. 的方案。

- **读**：`results/castillo/castillo_{regression,classification}_metrics.csv`（先跑 `analysis/12_eval_castillo.py`）
- **写**：`figures/fig5{a,b,c,d,e}_*.pdf` + `fig5_legend.pdf`，一个 panel 一个文件
- 5A / 5B / 5C：1 行 × 2 列（PCC、SCC，各 `(8,6)`），分别是全集活性、CTS 集活性、CTS 集 residual。**不画 MAE/RMSE**：Castillo 活性从没做过 z-score，误差类指标量的主要是两个尺度之间的偏移。这六个子图**共用一个 y 范围**（PCC 和 SCC 也共用），全集→CTS→residual 的下降能直接读出来
- 5D / 5E：2 行 × 2 列（AUROC / Normalized AUPRC 上排，2% EF / 5% EF 下排，`(8,12)`），分别是 CTS-high 和 CTS-low，同一个 `plot_classification_panel` 跑两遍。上排的竖排模型名要占满整个行间距（`hspace=0.45`），不然会被下排 axes 的白底盖掉
- **四个 panel 每个都在 x 轴写全模型名**（竖排）。5A–5C 叠起来会得到三份重复的标签，删掉比补上容易；反过来单独摆一个 panel 时它必须能自己说清哪个箱是哪个模型
- Normalized AUPRC = `(AUPRC − prevalence) / (1 − prevalence)`，在 `analysis/12_eval_castillo.py` 里算，所以随机期望画在 0
- 一个模型一列（7 个细胞系散点 + 一个常规箱线图，跟 fig1c / fig4bde 同一套），不按模型上色，散点也不描白边——整张图里颜色只表示细胞系这一件事。模型顺序和名字取 `config.castillo_model_names` / `castillo_model_styles`（DHS64 / DNase-AG / AG-VEF-only / EpiCast-AG），后者的配色只有归档的 `_fig5` 在画
- 细胞系配色 `config.castillo_cell_colors` 就是 `cell_colors` 那组蓝→黄→绿→红→紫再往后接棕和青。前三个是 Gosai 也有的细胞系，颜色跟 fig1/fig3 一致；红和紫跟 Gosai 的 HCT116 / A549 撞色，但没有任何一张图会同时画两个数据集的细胞系
- **5E 只有 3 个细胞系**（2026-08-25 恢复，等 C.Z. 定去留）：CTS-low 阳性数是 HepG2 0、HeLa-S3 1、SK-N-SH 6、WERI-Rb-1 11、K562 84、GM12878 169、MCF-7 281，过阈值后只剩 K562 / GM12878 / MCF-7，所以这张图上一列只有三个点，箱体基本退化成两点连线。手稿里 Fig 5E 的结论（EpiCast 最好、AG-VEF-only 次之、两个 DNase 模型垫底）在这三个细胞系上仍然成立：AUROC 中位 0.655 / 0.546 / 0.569 / 0.495，2% EF 中位 1.78 / 1.18 / 0.59 / 0.53。每个细胞系的阳性数在 `results/castillo/castillo_cts_counts.csv`，图注必须写出来
- 阈值 `config.castillo_min_positives = 20`（2026-08-25 从 6 提上来），即**按 (细胞系, task) 计，阳性数 ≥ 20 才进图或进均值**。CTS-high 最少的 MCF-7 有 95 个，一个都不排除。图注可写：*Cell type–task combinations with fewer than 20 positive CTS CREs were excluded (CTS-low: HepG2 n = 0, HeLa-S3 n = 1, SK-N-SH n = 6, WERI-Rb-1 n = 11).*
- ⚠️ **不含 EpiCast (Sei)**，而且不该补上：它的 Castillo 预测是拿 AlphaGenome VEF 的 dataset config 喂 Sei checkpoint 跑出来的（四个 `*castillo_dataset*` 配置全指向 AG VEF，没有一个用 `data/castillo_mpra/sei_vef.tsv`），VEF 口径与训练时不一致。导出那行已在 `analysis/14` 注释掉，旧产物改名为 `results/predictions/_castillo_epicast_sei_vef_wrong_vef.tsv`。要进图得先做一份 Sei VEF 的推理配置重跑

---

## `figures/` 里的历史文件

带 `_` 前缀的都是旧产物：改名前那一版全套图，加上 `_fig4e_5_seq_61_cell_type_bar.pdf`、`_fig4e_DNase.pdf`、`_fig4e_EpiCast_VEF_only_prediction_heatmap{,_cluster}.pdf`（61 细胞系热图，对应根 README 提到的 web server 场景，已无脚本生成）。

## `deprecated/plot/`

`_fig1c_assay_coverage_heatmap.py`、`_fig2d_virtual_ccre.py`、`_fig3c_cts_threshold.py`、`_fig3d_epicast_residual_scatter.py`、`_fig3e_topk_retrival.py`、`_fig4d_known_promoters.py` —— 只有这些还在引用 `results/train3test2_*` 那几个已无 writer 的目录。

`_fig4c_castillo_mpra.py`、`_fig4d_castillo_cts_classification.py` —— 我原先的 Castillo 雷达图（分位数 CTS 口径），已被 `fig5_castillo_metrics.py` 取代。
