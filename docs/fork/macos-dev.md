# macOS 开发与测试注意事项

upstream CI 只跑 Linux、FreeBSD 与 Windows(GPU 相关),**不覆盖 macOS**。以下是
2026-11-18 同步 upstream `ed14e1c41`(craft UI 迁移)rebase `feat/ml-subject` 时踩到、
且会长期复发的两个坑,以及对应的验证配方。

## 1. 单包测试必失败:egui/serde 的 workspace feature 统一

**现象。** `cargo test -p photocraft-ui-egui`(或 `cargo check -p … --all-targets`)
报约 36 个 `egui::Id: Serialize / DeserializeOwned` 错误,位置在 `dialogs/*_prompt.rs`、
`panels/actions/*`、`menus/invoke.rs` 等 **lib 文件**——看起来像代码坏了,其实不是。

**根因。** `egui::Id` 的 serde impl 在 egui 的 `serde` feature 后面
(`#[cfg_attr(feature = "serde", derive(serde::Serialize, …)])`),而没有任何
ui-egui 自己的依赖打开它。该 feature 靠 **全 workspace 的 feature 统一**带进来:
`apps/photocraft` → eframe/egui-winit 链 → `egui feature "persistence"` → `serde`。
`-p` 单包构建的依赖图里没有 apps,统一随之消失,lib 里的 `json!({"dialog": id})`
(`DialogId = egui::Id`)就编不过。

**判别。** `cargo tree --workspace -i egui -e features` 能看到
`egui feature "serde"`,`cargo tree -p photocraft-ui-egui -i egui -e features` 看不到。

**正确姿势。** 测试一律走 workspace 入口:

```sh
cargo test --workspace --lib -- <过滤器>
```

## 2. external_svg.rs 的 mkfifoat 在 apple 平台不存在

**现象。** `cargo test --workspace --bins|--tests|--all-targets`(包括
`cargo xtask test`)编译 `photocraft (bin "photocraft" test)` 失败:
`cannot find function 'mkfifoat' in module 'rustix::fs'`
(`apps/photocraft/src/external_svg.rs` 的
`unix_saved_file_rejects_symlinks_and_fifos_without_blocking`)。

**根因。** 该测试标 `#[cfg(unix)]`,但 rustix(`=1.1.5`)的 `mkfifoat` 自身 cfg 为
`not(any(apple, …))`——macOS 上根本不存在这个函数。upstream 的 unix 测试只在
Linux 跑,所以上游绿着。与坑 1 叠加成死结:全 workspace(保 feature 统一)必撞
mkfifoat,排除 apps(躲 mkfifoat)必丢 egui/serde。

**临时姿势。** 验证时把该测试的 cfg 临时改成
`#[cfg(all(unix, not(target_os = "macos")))]`,跑完恢复,**不进提交**。

**修法(待回馈 upstream)。** apple 走 `mkfifo`(无 `*at` 变体,用 `libc`/`nix`),
或把该测试 cfg 到 Linux;顺手可在 ui-egui 的 dev-dependencies 显式声明
`egui = { features = ["serde"] }`,让单包测试也能编(坑 1 的防御)。

## 验证配方(2026-11-18 实测)

| 目的 | 命令 |
|---|---|
| 编译(非测试) | `cargo check -p <crate>`(单包无碍,serde 错误只出现在 lib **测试**构建) |
| 单元测试 | `cargo test --workspace --lib -- <过滤器>` |
| 含集成测试 | 临时门控坑 2 的测试后 `cargo test --workspace --lib --tests` |

- `cargo test -p photocraft-ui-egui` 与 `--exclude` 变体都不可用(分别踩坑 1 的两面)。
- 回馈 upstream 后更新本文件与 [`README.md`](README.md) 索引。
