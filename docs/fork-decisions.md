# Fork 决策记录

本文件记录 **fork 级**的产品与架构决策（ehcapa/photocraft，方向：像素蛋糕式 AI 修图）。
上游的 [`docs/decisions.md`](decisions.md) 只记随代码合入的实现决策（带 PR 号、事后补记），
fork 级内容写在那里会制造合并冲突，因此单独成文。

条目格式仿上游：`## 日期: 标题`，正文为背景 / 决策 / 后果。一旦某条决策落了代码并合入，
按上游格式在其 decisions.md 补正式条目，此处只留链接。

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

**背景。** 上游没有任何 ML 运行时（gaps.md G15，官方评估 AI 完成度 10%），
且 workspace 全局 `unsafe_code = "forbid"`、wasm 构建有 24 MiB 体积门。
上游把生成式 AI 定为 owner decision（#41，冻结中），但社区生态里生成式接入已经拥挤
（#730 PR 排队、LayerShop 成品 fork），而"像素蛋糕式"人像精修 / 影楼工作流是空白。
已有积木：YuNet 人脸检测（`crates/algo/src/faces.rs`，纯 Rust 推理、权重编译进二进制、
无运行时依赖——这是上游自己开创的"小模型移植"先例）、surface blur、inpaint、
GrabCut/matting、denoise、relight。

**决策。** 按依赖性质把能力分三层，各自走不同的实现路线：

- **L1 纯经典算法（自研，优先做）**：磨皮（皮肤蒙版 + 频率分离）、瑕疵消除工作流、
  经典追色（Reinhard / 直方图匹配）、天空替换 / 证件照流水线、批量队列、Raw 一键转档
  预设包。零模型依赖，复现像素蛋糕约五六成体验，直接落在 filter op / command 体系里。
- **L2 小模型（纯 Rust 移植，`faces.rs` 模式）**：人像 / 皮肤 / 天空 / 牙齿分割、
  稠密人脸关键点、超分。手写推理或纯 Rust 推理框架（tract/candle），权重编译进二进制；
  不引入 onnxruntime / CoreML 等 C-FFI 运行时（违反 unsafe 禁令，浏览器版不可用，
  上游不会接受）。每个模型都要手工移植，投入按模型逐个评估——L2 是磨皮质量
  （像素蛋糕"不假脸"）的真正护城河，也是投入决策重点。
- **L3 生成式（一律外部 provider，不自建推理）**：创成式填充、换天气、表情 / 妆容、
  上色。走 #41 的 provider interface；首选 ComfyUI 本地适配器（零成本、无账号、
  用户复用已有模型；此优先级为工作假设，待首个 provider 落地时复核），云端 API 为
  后续适配器。fork 自身永不内嵌生成式模型权重。

**后果。** L1/L2 完全离线、浏览器版可承载（受 24 MiB 门约束需逐模型评估体积）；
L3 依赖外部服务但架构与上游合流方向一致；三层互不阻塞，L1 不等模型即可持续出活。
明确放弃的方向：联机拍摄（相机 SDK 必然引入 unsafe C-FFI）、排版 / 选片生态
（超出图像编辑器范畴）。

## 待补条目（代码合入时）

- `filter.retouch.skinSmooth` v0（CbCr 肤色蒙版 + 频率分离，经典算法）——合入后按
  上游 decisions.md 格式补记，并在此处留链接。
