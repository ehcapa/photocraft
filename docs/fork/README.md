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
- [macOS 开发与测试注意事项](macos-dev.md) —— 单包测试的 egui/serde feature 统一坑、
  external_svg.rs 的 mkfifoat apple 门控坑、workspace 级验证配方
