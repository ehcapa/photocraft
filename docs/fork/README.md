# Fork 文档

ehcapa/photocraft（方向：像素蛋糕式 AI 修图）的 **fork 级**文档：产品与架构决策、
大功能的选型档案。上游的 [`../decisions.md`](../decisions.md) 只记随代码合入的实现决策
（带 PR 号、事后补记），fork 级内容写在那里会制造合并冲突，因此独立成目录。

## 组织

| 文件 | 内容 |
|---|---|
| [`decisions.md`](decisions.md) | 决策日志：`## 日期: 标题`，背景 / 决策 / 后果。条目保持精简，细节指向主题档案 |
| `YYYY-MM-DD-<slug>.md` | 主题档案：一个大功能的完整记录（选型、架构、数据、验证、后续） |
| [`macos-dev.md`](macos-dev.md) | 长期有效的环境注意事项：macOS 开发/测试与 upstream CI（无 macOS）的差异、坑与验证配方 |

## 约定

- 决策落了代码并回馈上游后，按上游格式在 `../decisions.md` 补正式条目，
  [`decisions.md`](decisions.md) 对应条目只留链接。
- 与上游共享的文档（`gaps.md`、`target-app-parity.md`、`roadmap.md` 等）**不在 fork
  侧手改**——fork 与上游的事实分歧记录在本目录，避免同步冲突。
- `feat/portrait-retouch` 分支上旧的 `docs/fork-decisions.md`（2026-11-16 两条）在两分支
  合并时并入本目录的 [`decisions.md`](decisions.md)。

## 索引

- [2026-11-18: ML Select Subject（tract + U²-Netᵖ）](2026-11-18-ml-subject.md) —— 选择主体
  换语义模型，经典管线保留回退；含模型升级阶梯与两个算法 bug 的根因
- [2026-11-18: Select Subject 的 +/- 行为对齐 Photoshop](decisions.md) —— upstreamable UI 修复
- [2026-11-18: 智能人像管线（设计选型，未实施）](2026-11-18-portrait-pipeline.md) ——
  DAG 编排 + 标准图层落位（不加 Adjustment variant）；L1/L2/L3 分层表、tract spike 待办
- [macOS 开发与测试注意事项](macos-dev.md) —— 单包测试的 egui/serde feature 统一坑、
  external_svg.rs 的 mkfifoat apple 门控坑、workspace 级验证配方
- [运行时模型清单](models.md) —— PHOTOCRAFT_MODELS_DIR 放置约定、blake3 校验、
  >50MB 模型的来源/许可/指纹表
- [新机器就绪清单](new-machine.md) —— 换机测试:代码/模型下载/环境变量/构建/
  三态测试/真图验收/应用冒烟,一张单子走完
- [2026-11-19: Smart Portrait 对话框不再记忆配方](decisions.md) —— 滑杆残值伪装成
  管线损坏的真图事件;not_remembered + portrait_diag 诊断工具
- [2026-11-19: 选区工具的模型区域化](decisions.md) —— analyze_person 共享(选区与人像
  管线一份分析、一份缓存)、roi/SCRFD 自动放框、FeyNoBg 发丝档软选区、八区域
  `select.region` + 族共用段 UI;FASHN 因许可不引入
- [2026-11-19: faceparser 导出的类语义不可信](decisions.md) —— class_probe 判决
  自相矛盾(左右眼环各被不同类占据),区域集封顶八类,脖/耳/眉/鼻不可给
- [2026-11-20: 服装细类解封(SCHP ATR-18)+ Pose 基础层](decisions.md) —— 优先级链
  替代 union(0.908 围巾错误随 selfie 不跑而消失)、arm/leg 只烧裸皮的判决、
  `Analysis.pose` 无消费者落地
- [2026-11-20: `filter.bodySculpt` —— 关键点驱动身体塑形](decisions.md) —— 解析写场
  四原语(不走 dab 链/不用 freeze,行级证据)、锚点嵌入 params 保回放、折叠守护
  首跑抓 6.4px 叠加、footprint 验收;五批补齐至对标 19/19(AI塑形/天鹅颈组合滑杆、
  直角肩/双下巴/胸位提升、身型流畅、L/R、对称液化、实拍校准)
- [2026-11-20: `filter.faceSculpt` —— 面板化面部重塑](decisions.md) —— 第二张部件表
  只加行不加机器(四原语共享)、47+48 滑杆五组、7 脸型预设、五官对称、3D塑颜
  (mesh z 旋转投影;`sin_cos` 顺序坑)、AI 面部流畅、面部 gain 校准 + 家族化折叠预算
- [2026-11-20: 塑形面板的定制 UI](decisions.md) —— 分区面板窗口 + 预设一键行 +
  L/R 链锁控件 + 应用不关面板;词条 86×2 入 craft-i18n;实时预览的修复路径记录待做
- [2026-11-23: 并入 upstream engine 拆分（step 3-4）](decisions.md) —— 命令文件搬
  engine-core、ort 转发多一跳（engine-core 侧入口变 `-p photocraft-engine-core
  --features ort`）、engine-core 测试的 `crate::session()` 约定（过滤跑假绿陷阱）
