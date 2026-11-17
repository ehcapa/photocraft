# 2026-11-18: ML Select Subject（tract + U²-Netᵖ）

选择主体从"显著性猜测"升级为语义检测。分支 `feat/ml-subject`；决策条目见
[`decisions.md`](decisions.md) 同日两条，本文是完整档案。

## 功能与改动

**新 crate `photocraft-ml`（L4 预留槽位，零 layers 改动）**

- U²-Netᵖ 显著主体检测（`data/u2netp.onnx`，4.6MB，Apache-2.0，md5
  `8e83ca70e441ab06c318d82300c84806`，rembg release）经 tract 0.23 纯 Rust 推理。
  许可文件在旁 + `ATTRIBUTION.md` 行（`crates/algo/data/libfacedetection.bin` 先例）。
- API：`subject_mask(px: &[[f32;3]], w, h) -> Option<SubjectMask>`（0..=1 行主序，
  调用方尺寸）；`AVAILABLE: bool`（wasm 恒 false）；`MODEL_SIDE/PX = 320`。
- 预处理逐行对齐 rembg（`sessions/base.py`）：拉伸到 320²（分数箱面积重采样，
  无 letterbox）→ `÷ max(px)`（非 ÷255）→ ImageNet mean/std → NCHW f32 →
  d1 头输出 min-max 归一 → 面积重采样回原尺寸。常数集中在 `model.rs` 一处，
  换方案三行改完。
- 模型构建 ~783ms（`OnceLock` 进程缓存，失败缓存 None 永久回退）、推理 ~676ms/次
  （release，i7 桌面）。plan `Send+Sync` 编译期断言。
- tract 依赖与权重均 `cfg(not(wasm32))`：web 端零体积影响（24MiB 门不受扰），
  恒走经典回退。

**algo 双路径（`crates/algo/src/segment/subject.rs`）**

- `select_subject_seeded_with(sampler, canvas, seed: Option<&SubjectSeed>, ctl)`：
  ML 掩码阈值 0.65/0.35 造 trimap（**无边框背景先验**），空掩码/畸形 seed 回退经典。
- ML 路径取舍：跳过显著性+Otsu+边框先验+backdrop_pass；`drop_specks`（≥最大组件
  0.5% 全保留）替代 `keep_main_and_nearby`（20% 大小/15% 边距规则会丢第二主体）；
  GrabCut 与全分辨率带状重切保留精边。
- 分层：algo（L2）不依赖 ml（L4）；engine 取掩码后作为数据参数下传。

**engine（4 个调用点统一 `subject_seed` helper）**

- `select.subject`（#2865 缓存原样）、`layer.removeBackground`（stage/取消原样）、
  `layer.maskAllObjects`、Select and Mask 的 subject 编辑。
- `PHOTOCRAFT_SUBJECT_ML=0` 可关；`subject_ml()` 默认 `!cfg!(test)`——单测自动
  经典化（确定性、速度），集成测试与 panic_hunt 走真模型。

**Select Subject +/- 行为对齐 Photoshop（upstreamable）**

- 选项栏按钮经 `subject_mode()` 复用所属工具的模式规则（快速选择：总加、Alt 减；
  对象选择/魔棒：栏状态 + Shift/Shift+Alt/Alt 覆盖）。
- 菜单派发点（`menus/invoke.rs`）对 `select.subject`/`select.sky` 注入按住修饰键的
  mode（Shift=加/Alt=减/Shift+Alt=交），显式 mode 参数优先；Properties 快捷动作保持
  替换（同 PS）。

## 顺带修复的两个算法 bug

1. **maxflow 负容量记账**：`neg_log` 下限 −50 产生负终端容量，违反流量域
   （`flow += min(s,t)` 会减流量）。修法：每节点两容量同减负偏移——能量平移
   标签无关常数，min-cut 最优解不变（经典 Photoshop 参考钉子测试全部原样通过）。
2. **粗种子 sliver 污染级联**：种子按自身分辨率最近邻采样，主体边缘 1-2px 落入
   硬背景 → bg GMM 学会主体色 → 数据代价翻转 → cut 丢弃整个主体（第二主体尤甚）。
   修法：PR_FG 外扩 1px 膨胀带（自由带由 cut 裁决）+ `grabcut_with_certain_bg`
   变体（bg 模型只用确定背景训练，符合 Rother 论文原始设计）。变体收窄在 seeded
   路径，经典参考行为零扰动。

## 模型升级阶梯

| 模型 | 体积 | 许可 | 结论 |
|---|---|---|---|
| U²-Netᵖ（当前） | 4.6MB | Apache-2.0 | 基线，rembg 家族最弱 |
| silueta | 43MB | Apache-2.0 | 小幅提升，同速度档，可提交候选（50MB 红线内） |
| MODNet | 25MB | Apache-2.0 | 人像专用，契合像素蛋糕方向，第二模型候选 |
| BEN2 | ~380MB | Apache-2.0（需复核条款） | 开源边缘质量第一梯队，仅运行时加载候选 |
| isnet-general-use | 170MB | **✗ 权重不开放**（DIS issue #150） | 排除 |
| BiRefNet | 200M+ 参数 | MIT | 太大，排除 |

计划：`PHOTOCRAFT_SUBJECT_MODEL=<path.onnx>` 运行时覆盖 + 从模型输入 fact 动态读输入
尺寸（rembg 家族预处理相同，BEN2 为 1024²+ImageNet 归一化）；大模型本地试跑不进仓库。
换模型用 `U2NETP_REFERENCE=<dir>` 一致性测试仲裁（对 rembg 输出二值化 IoU>0.9）。

## 验证

- `cargo test -p photocraft-ml`：7✓（含真模型 disc 冒烟）
- `cargo test -p photocraft-algo --lib`：984✓（新增 seeded 双主体/满幅/回退/取消场景）
- `cargo test -p photocraft-engine --lib`：4057✓；`--test it`：575✓（含真 ML 端到端 2 项）
- panic_hunt `--ignored`：✓（真 ML 路径被 fuzz）
- `cargo xtask layers` ✓ / `cargo xtask wasm` ✓ / clippy `-D warnings` ✓ / fmt ✓
- ui-egui：menus 208✓、retouch_ui 113/115（2 个失败为本机缺 craft-fonts 的环境性
  失败，改动前即存在，已用还原法验证）
- 未做：真照片人工对比（`cargo run --release -p photocraft -- <照片>`）、rembg 一致性
  测试需 `U2NETP_REFERENCE` 真照片、推理延迟优化（>680ms 再考虑量化/换 rten）

## 改动清单

新：`crates/ml/`（Cargo.toml、lib.rs、model.rs、resample.rs、tests.rs、
data/u2netp.onnx、data/LICENSE-u2netp.txt）、`crates/engine/tests/it/ml_subject.rs`、
`docs/fork/`。
改：根 Cargo.toml（+photocraft-ml、+tract-onnx）、`crates/algo/src/segment.rs`
（grid_cut 容量平移）、`segment/grabcut.rs`（certain_bg 变体）、`segment/subject.rs`
（双路径）、`segment/tests.rs`、engine 4 调用点 + Cargo.toml、
`select_mask_edits.rs`、`tests/it/main.rs`、ui-egui 三处按钮 + `menus/invoke.rs` +
两组测试、`docs/architecture.md` §10、`ATTRIBUTION.md`。
