# Fork 决策日志

fork 级产品与架构决策，`## 日期: 标题`，背景 / 决策 / 后果；细节指向
[`YYYY-MM-DD-<slug>.md`](.) 主题档案。回馈上游的决策按上游格式在
[`../decisions.md`](../decisions.md) 补条目，对应条目只留链接。

---

## 2026-11-16: 分支模型与上游同步策略

**背景。** 上游 storytold/photocraft 合并速度极快（日均数十个 PR，1,000+ 已合并）。
fork 要做的是长周期能力建设（人像精修、批量工作流），不可能一次做完；偏离上游越久，
合并成本越高。同类先例：LayerShop（以生成式 AI 为主题的 PhotoCraft fork）的维护方式是
"每实现一批功能、发版前 merge 一次 upstream"。

**决策。**

- `main` 只做 `upstream/main` 的镜像：只允许 `git merge --ff-only upstream/main`，
  永不直接提交。它是比对、rebase、放弃重来的干净基准。
- 所有开发在 `feat/portrait-retouch`（产品分支）上进行；上游同步进 main 后，
  定期（间隔不超过两周）`git merge main` 进产品分支，冲突当次解决。
- 纯经典算法、无 AI 决策依赖的部分（磨皮 op、追色、批量队列等）切成 PR 级小分支
  （`feat/<slug>`，命名随上游 `fix/`、`refactor/`、`cutover/` 惯例）回馈上游，
  合一个少一个，缩小长期分叉面。
- 生成式 AI 相关实现**对齐上游 [#41](https://github.com/storytold/photocraft/issues/41)
  已定稿的六条架构要求**（provider trait 独立 crate、默认无网络、命令化、密钥进钥匙串、
  隐私文案、无 provider 不显示菜单），保证将来合流阻力最小。

**后果。** 任何时候可以放弃或重置产品分支而不失基准；上游冲突被切小、摊平到每次同步；
fork 的战略部分（人像精修）即使永不回馈也不污染与上游共享的代码路径。

## 2026-11-16: AI 修图能力的三层架构

（完整条目在 `feat/portrait-retouch` 分支的旧 `docs/fork-decisions.md` 上；两分支合并时
移入本文件。）

---

## 2026-11-18: ML Select Subject——tract + U²-Netᵖ 提交权重，经典管线保留回退

**背景。** 现有 Select Subject 是纯经典启发式（SLIC 显著性 → Otsu → GrabCut），
不准的根因是无语义（显著性 ≠ 主体，#2788）、Otsu 双峰假设 + 边框硬编码背景、
组件过滤丢第二主体。上游 roadmap M10 把 `ml` crate 规划为 ort（C++ 绑定），
#41 owner decision 冻结中。

**决策。** 运行时用 tract（纯 Rust、MIT/Apache 双许可、直接加载 ONNX）不用 ort；
u2netp.onnx 4.6MB 直接提交仓库（libfacedetection.bin 先例，>50MB 的模型走运行时
加载不提交）；ML 粗掩码只替换种子阶段，GrabCut + 全分辨率带状重切保留精边，
缺失/失败/空掩码一律静默回退经典管线；algo（L2）不依赖 ml（L4），engine 取掩码后
作为数据参数下传；wasm 全门控（依赖与权重均 cfg 掉）；单测默认经典化（`!cfg!(test)`），
集成测试与 panic_hunt 走真模型；`PHOTOCRAFT_SUBJECT_ML=0` 可关。

**后果。** 双主体、贴边主体、满幅主体这些经典系统性失败模式全部修复；代价是推理
~680ms/次与仓库 +4.6MB。选型细节、模型升级阶梯（silueta/MODNet/BEN2；isnet 权重
不开放、BiRefNet 太大均排除）、调试途中修复的两个算法 bug（maxflow 负容量、
种子 sliver 污染）见
[主题档案](2026-11-18-ml-subject.md)。

## 2026-11-18: Select Subject 的 +/- 行为对齐 Photoshop

**背景。** 选项栏 Subject 按钮写死 `replace` 不跟随模式按钮；菜单项不支持修饰键——
与 Photoshop 不一致（PS 按钮跟随 +/- 状态，菜单 Shift=加/Alt=减，Sky 同款）。

**决策。** 按钮经 `subject_mode()` 复用所属工具的模式解析规则；菜单派发点对
`select.subject`/`select.sky` 注入按住修饰键的 mode，显式 mode 参数优先；Properties
快捷动作保持替换。细节见[主题档案](2026-11-18-ml-subject.md)。

**后果。** upstreamable 的纯 UI 行为修复（无 AI 依赖）：回馈上游后在
[`../decisions.md`](../decisions.md) 补条目，此处改留链接。

## 2026-11-18: 智能人像管线——DAG 编排 + 标准图层，不加 Adjustment variant

**背景。** 像素蛋糕式人像精修要做成"一个智能人像调整层"加进调整层组，约束是长期
同步 upstream（日均数十 PR）与 PSD 往返兼容。初始方向为八段串行链
（FeyNoBg 抠人 → Sapiens2 分区 → FaceMesh/Pose → 皮肤 → GAN 面部增强 → 换发色 →
SD/Flux inpaint → 追色）。

**决策。** 管线按 DAG 建模：分析层（抠像/分区/关键点，缓存元数据）→ 执行层并行
分支（面部增强在皮肤之前；换发色；inpaint）→ 追色殿后。阶段分层：磨皮/换发色/追色
L1 经典（可回馈上游）、抠像/关键点/face-parsing/面部增强 L2 tract（Sapiens2 越界
降级）、SD/Flux L3 provider（对齐 upstream #41）。**不给 `doc::Adjustment`（PSD
序列化格式）加 fork variant**：fork-only 编排 crate 创建标准图层（调整层 + 生成
蒙版、智能滤镜、缓存栅格）+ sidecar 分析缓存，产出可手调图层组。

**后果。** 共享代码触点收到菜单注册一处，doc 格式零分叉；生成类阶段以缓存栅格
落地、按阶段失效重跑。完整落位表、风险与 spike 待办见
[主题档案](2026-11-18-portrait-pipeline.md)。

## 2026-11-19: Smart Portrait 对话框不再记忆上次配方

**背景。** 真图验收首晚出现"处理后完全不可用"的导出：全幅青灰罩层、绿色腮红块、
紫色眼带。逐层转储（portrait_diag）证明管线无罪——默认配方下六个层蒙版全部解剖学
正确、输出与原图差 ~1/255；元凶是 schema 对话框的"记住上次值"机制：前一晚实验留下
的 `hue=164/saturation=-61/lightness=+43`（人物整体层）+ `blushHue=91`（绿腮红）+
`shadowHue=286`（紫眼影）被原样载入执行，用这组残值回放与坏图逐通道一致（残差
1.4 = JPEG 噪声）。

**决策。** `portrait.create` 的全部对话框行 `not_remembered()`（filter_dialog 的
`params()`），每次打开取 spec 默认值；"重复上次配方"的需求由既有的
`portrait.saveRecipe` → Actions/Batch 承担——那才是带名字、可审计的配方载体。
`crates/engine/examples/portrait_diag.rs`（逐层 solo 合成 + 蒙版 PGM 转储）作为
fork 诊断工具保留，与 class_probe 同列。

**后果。** 用户配置里已落盘的 `[sections.remembered."photocraft.portrait.create"]`
残值自然失效（重开时按行过滤，不回填）；新增测试
`smart_portrait_never_remembers_its_recipe_rows`；排障口诀进
[new-machine.md](new-machine.md) §8：输出离谱先查参数来源，再怀疑管线。
