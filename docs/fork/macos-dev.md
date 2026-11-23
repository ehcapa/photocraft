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

## 3. automation crate 的 trusted-root 路径校验在 macOS 误判(2026-11-18 发现)

**现象。** `cargo test --workspace --lib` 里 `crates/automation` 的
`trusted_root_names_are_not_validated_as_relative_request_names` 失败:
`automation path rejected: absolute paths must be inside the automation root:
/var/folders/…/T/photocraft-workspace-…/inside/aux/new-0.bin`——路径明明是测试在
自己的临时根下构造的。

**根因。** macOS 的 `TMPDIR`(`/var/folders/…`)与 canonicalize 后的
`/private/var/folders/…` 符号链接不一致,"在根内"的前缀比较一边 canonical 一边不
canonical,误判为越界。upstream CI 无 macOS,盲区。

**现状。** 与本地改动无关(干净树同样失败),属 upstream bug,待回馈;本地验证时
知晓即可,不影响 portrait/人像链路的任何测试。

**修法(待回馈 upstream)。** apple 走 `mkfifo`(无 `*at` 变体,用 `libc`/`nix`),
或把该测试 cfg 到 Linux;顺手可在 ui-egui 的 dev-dependencies 显式声明
`egui = { features = ["serde"] }`,让单包测试也能编(坑 1 的防御)。

## 验证配方(2026-11-18 实测,同日二次修正)

| 目的 | 命令 |
|---|---|
| 编译 | `cargo check --workspace`;或 `cargo check -p <crate>`(**仅当**该 crate 不依赖 ui-egui) |
| 单元测试 | `cargo test --workspace --lib -- <过滤器>` |
| 含集成测试 | 临时门控坑 2 的测试后 `cargo test --workspace --lib --tests` |

- **`-p photocraft-ui-egui` 的任何构建(lib check 也在内)都失败**——dialogs 的
  `json!({"dialog": id})` 是 lib 代码,`DialogId = egui::Id` 的 serde 就在 egui feature
  后面。多包选择只要含 `photocraft`(app)即可借它的 eframe 链统一 feature
  (如 `cargo check -p photocraft-ui-egui -p photocraft`),但那不如直接 `--workspace`。
- `cargo test -p photocraft-ui-egui` 与 `--exclude` 变体都不可用(分别踩坑 1 的两面)。
- 回馈 upstream 后更新本文件与 [`README.md`](README.md) 索引。

## 4. engine 的 allocation.rs 非 UTF-8 路径测试失败(2026-11-19 发现,2026-11-23 复验)

**现象。** `cargo test -p photocraft-engine-core --lib` 全量跑时
`allocation::tests::read_file_within_reads_a_path_that_is_not_utf8` 失败:
`Os { code: 92, kind: Uncategorized, message: "Illegal byte sequence" }`。

**根因。** macOS(APFS)对非 UTF-8 文件名的系统调用语义与 Linux 不同(EILSEQ 一族,
ext4 接受任意字节,APFS 拒绝),干净树同样失败(stash 验证过),与本地改动无关。
2026-11-23 合并 upstream engine 拆分后该文件随目录改名到
`crates/engine-core/src/allocation.rs`,与 main 逐字节一致,失败照旧。upstream CI
无 macOS,盲区。

**现状。** 环境既有问题,本地验证时忽略;待回馈 upstream。

## 5. 大合并后 --workspace 报 E0425 而 -p 通过:陈旧 rmeta(2026-11-19 发现)

**现象。** 合并 upstream 后 `cargo check --workspace --tests` 报 8 个
`E0425: cannot find … in crate photocraft_ml`(源码里明明有无条件导出),而
`cargo check -p photocraft-engine --lib`(带不带 `--features ort`)全部通过——
现象自相矛盾。

**根因。** 同一 crate 在 workspace 里存在多个 feature 组合的编译单元
(`target/debug/build/photocraft-ml/<hash>/`),engine 链到了**合并前**的陈旧
rmeta。`--workspace` 与 `-p` 选中的单元不同,不是代码问题。

**修法。** `cargo clean -p photocraft-ml` 一次通过。判别:
`cargo check --workspace -v 2>&1 | grep 'extern photocraft_ml='` 看 engine 实际
消费的 rmeta 路径,再对 `ls -la target/debug/build/photocraft-ml/` 的日期。
方向性口诀:**与坑 1(feature 统一)相反**——坑 1 是 `-p` 失败 `--workspace`
通过,本坑是 `--workspace` 失败 `-p` 通过,先分清再动手。(2026-11-23 engine 拆分
后单元测试搬进 engine-core,对照检查用 `-p photocraft-engine-core`。)

## 6. ui-egui 的 kittest 无头测试在 macOS 找不到菜单节点(2026-11-20 发现)

`input::touch_twins::tests::a_tap_opens_a_menu_title_every_time` 在无头
kittest harness 里以 `No nodes found matching the query: By { label: "File" }`
失败——纯 main 上同样失败(git stash 对照验证过),与本机改动无关。跑 ui-egui
单测见到它失败时按预存基线忽略;需要菜单交互测试时在真机窗口环境跑。
