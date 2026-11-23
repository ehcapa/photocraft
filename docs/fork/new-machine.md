# 新机器就绪清单

换机器继续开发/测试人像管线时照此把环境立起来(2026-11-18 首次整理,当天的分支是
`feat/ml-subject`)。

## 1. 代码

```sh
git clone https://github.com/ehcapa/photocraft.git && cd photocraft
git switch feat/ml-subject        # 人像管线全在这里
```

**前提:本机(旧机器)已把未推送的提交推上 origin**——不推,新机器上什么都没有。

## 1.5 git 凭据(新机器要推送时)

HTTPS 推送需要凭据:GitHub → Settings → Developer settings → Personal access tokens
生成新 token(旧 token 已在对话/工具中暴露过,建议 revoke 重发),或配 SSH key。
本仓库 remote:`origin = github.com/ehcapa/photocraft`(fork),`upstream =
github.com/storytold/photocraft`(只读镜像,永不推送)。

## 2. 模型(运行时目录,不进仓库)

```sh
mkdir -p ~/Project/Rust/models && cd ~/Project/Rust/models
# 注意:huggingface.co 直连通常不通,用 hf-mirror.com
curl -LO https://hf-mirror.com/bluefoxcreation/Face_parsing_onnx/resolve/main/faceparser_sim.onnx
mv faceparser_sim.onnx faceparser.onnx                                  # 规范名,52.6MB
curl -L -o facedetector.onnx https://hf-mirror.com/immich-app/buffalo_l/resolve/main/detection/model.onnx   # SCRFD 人脸检测,16MB,自动放框
curl -LO https://hf-mirror.com/neurobytemind/GFPGANv1.4.onnx/resolve/main/GFPGANv1.4.onnx
mv GFPGANv1.4.onnx gfpgan.onnx                                          # 340MB
curl -L -o schp-atr.onnx https://hf-mirror.com/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/model.onnx   # ATR-18 服装分割,110MB
curl -LO https://hf-mirror.com/PinkPixel/fey-nobg-onnx/resolve/main/model_fp16.onnx
mv model_fp16.onnx feynobg.onnx                                         # 563MB,仅 ort 档需要
```

清单/指纹/许可:`models.md`。**不下载任何模型也能跑**——管线自动降级(仅 Person
切片 + 唯一/提亮等 landmark 功能,mesh 内嵌在仓库)。

## 3. 环境变量

```sh
# ~/.zshrc(或运行配置):
export PHOTOCRAFT_MODELS_DIR=~/Project/Rust/models
export CRAFT_I18N_DIR=/Users/<你>/Project/Rust/craft-i18n   # 中文 UI,路径按新机器改
export ORT_DYLIB_PATH=~/Project/Rust/onnxruntime/lib/libonnxruntime.so   # Linux;macOS 同路径是 libonnxruntime.dylib
# ort 档(仅 feynobg 需要)= 编译 feature(§4)+ 上面的 ORT_DYLIB_PATH
# 注意:app 进程必须带着这两个变量启动(GUI 图标启动的进程没有 shell 的 export)。
# 2026-11-20 实况:缺 PHOTOCRAFT_MODELS_DIR 时 hair 档静默回落标准档(两者结果
# 一样,像没生效)、服装子区域报 "no … region — no person found" —— 都先查启动环境
```

## 4. 构建

```sh
cargo build -p photocraft                                   # 纯 Rust 默认档
cargo build -p photocraft --features ort                    # + ort 档(feynobg)
#   app 的 ort 是转发 feature(engine → engine-core → photocraft-ml/ort,2026-11-23
#   engine 拆分后多一跳);`--features photocraft-ml/ort` 只在直接依赖 ml 的包上合法
#   (如 -p photocraft-engine-core),从 app 传会报 does not contain this feature
```

**Linux 坑(Ubuntu 22.04,glibc 2.35)**:pyke 预编译的 onnxruntime 静态库要 glibc ≥ 2.38 +
libstdc++ ≥ 13(`__isoc23_strtol`/`_M_replace_cold` 未定义符号),链接必败。工作区 ort 已改
`load-dynamic`(运行时 dlopen),构建期零原生链接。动态库三个条件:**必须 1.23.x**(ort
2.0-rc.15 的 `GraphOptimizationLevel::Level3` 映射 `ORT_ENABLE_LAYOUT=3`,1.22 会报
`graph_optimization_level is not valid` 且被 ml 的 `.ok()?` 吞成 None)、**必须导出 C API 入口
`OrtGetApiBase`**、GLIBC ≤ 2.27 / GLIBCXX ≤ 3.4.22。GitHub release tarball 与 NuGet 的
`Microsoft.ML.OnnxRuntime` 包皆满足。**陷阱:PyPI wheel 里那把 `libonnxruntime.so` 是
Python 裁剪 ABI,不导出 C API**。GitHub 直连不稳时走 NuGet(可断点续传):

```sh
mkdir -p ~/Project/Rust/onnxruntime/lib && cd /tmp
curl -L -C - "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.23.2" -o ort.nupkg   # ~123MB,断了重跑即续传
unzip -q ort.nupkg -d ort-extract
cp ort-extract/runtimes/linux-x64/native/libonnxruntime.so ~/Project/Rust/onnxruntime/lib/libonnxruntime.so.1.23.2
cd ~/Project/Rust/onnxruntime/lib && ln -sf libonnxruntime.so.1.23.2 libonnxruntime.so
nm -D libonnxruntime.so | grep OrtGetApiBase    # 验证入口存在
# 运行(IDEA 运行配置/终端)都要带:ORT_DYLIB_PATH=~/Project/Rust/onnxruntime/lib/libonnxruntime.so
```

## 5. 测试(macOS 专属注意,见 macos-dev.md)

```sh
# 单元测试必须 workspace 级入口(单包 -p 会踩 egui/serde 统一坑):
cargo test --workspace --lib -- portrait
# 带模型两态(2026-11-23 engine 拆分后单元测试都在 engine-core):
PHOTOCRAFT_MODELS_DIR=~/Project/Rust/models cargo test -p photocraft-engine-core --lib portrait
# 带 ort(三态,全套 ~2 分钟,每个 create 真跑 10s matte):
PHOTOCRAFT_MODELS_DIR=~/Project/Rust/models \
  cargo test -p photocraft-engine-core --lib --features ort portrait
# 全量回归里 automation 的 trusted-root 失败是 macOS 已知预存问题,忽略
```

## 6. 真图验收(换机测试的核心)

```sh
PHOTOCRAFT_PORTRAIT_PHOTO=/任意/人像.jpg \
PHOTOCRAFT_MODELS_DIR=~/Project/Rust/models \
PHOTOCRAFT_PORTRAIT_OUT=/tmp/out.png \
  cargo test -p photocraft-engine-core --lib -- real_photo
# → 全开管线(修复+磨皮+发色+唇色+提亮+追色)→ 压平导出 PNG,计时打在 stderr
# 类语义诊断(如需重验映射):
PHOTOCRAFT_PORTRAIT_PHOTO=/任意/脸部裁剪.jpg PHOTOCRAFT_MODELS_DIR=... \
  cargo run -p photocraft-engine --example class_probe
# 逐层排障(哪个层/哪个蒙版出问题):每层 solo 合成 + 蒙版 PGM 转储到 /tmp/diag/,
# 第二参数可选 params JSON(缺省 = 对话框默认配方;要 ort 档加 --features ort):
PHOTOCRAFT_MODELS_DIR=~/Project/Rust/models \
  cargo run -p photocraft-engine --example portrait_diag -- /任意/人像.jpg
```

## 7. 应用内手动冒烟

`cargo run`(要 ort 则加 feature)→ 打开一张人像 →
Layer › New Adjustment Layer › Smart Portrait… → 对话框调参 → OK →
进度条(Esc 可取消)→ 图层面板出现 "Portrait" 组,逐层可调。

塑形/面部面板(2026-11-20):同一张人像 → Filter ▸ Body Sculpt…(或 Face
Sculpt…)→ 左侧分区切换,右侧滑杆,L/R 的 🔗 可解锁两侧,面部顶部 7 预设一键 →
[应用] 后面板保持打开,每次应用一步 Undo;首次分析数秒(缓存),此后秒回。

## 8. 已知等待项(测试时留意)

- 2026-11-19 数值验收:默认配方下六个层蒙版解剖学正确(人物/皮肤/眼/唇/腮红/眼睑),
  输出与原图差 ~1/255;GFPGAN/磨皮/发色的**真人观感**目检仍待做
- 19 类里 14(neck)等未映射;眼/唇区域来自 landmark 环(此导出不分眼/唇)
- matte(ort 档)~10s/张、峰值内存 ~5GB;批量全开 ~15–20s/张
- 2026-11-20 塑形面板:各 gain 已按 footprint 实拍校准(轮廓 ±100 ≈ 6% 脸宽),
  但**真人观感目检未做**(预设七件的数值同样待实拍微调);实时预览未实现
  (分析每 tick 重跑 + 代理缩放锚点错位,修复路径在 decisions.md)
- **对话框会记住每个命令上次的值**(`~/.config/photocraft/config.toml` 的
  `[sections.remembered]`)。2026-11-19 的"处理后全图青灰不可用"导出即前一晚实验
  滑杆残值(hue 164/sat -61/light +43 + 绿腮红 91),管线本身无罪;
  `portrait.create` 已改为不记忆(decisions.md 2026-11-19),其余 filter.* 对话框
  仍记忆。排障口诀:输出离谱先查参数来源(config.toml / 对话框滑杆位置),
  再怀疑管线(portrait_diag 逐层转储一分钟能定性)

## 9. 中文词条(craft-i18n fork)

UI 新增英文字符串后,中文显示需要 `~/Project/Rust/craft-i18n`(fork 自
storytold/craft-i18n,双远程同 photocraft)。流程:改
`locales/zh-hans/photocraft.tsv` / `zh-hant`(TAB 三列:语境/英文/译文,按英文
排序插入)→ `cargo test --workspace` 过目录检查器 → 按 AGENTS.md 在
ATTRIBUTION.md 登记出处(AI 辅助需注明)→ 提交。**检查器规则提醒**:与
common.tsv 同译文的行必须删(应用目录不允许重复覆盖共享行);顺序错了会挂。
`tl!` 缺词条时回退英文,不阻断。

## 10. target/ 磁盘管理(2026-11-19:164GB 案例)

workspace 的 feature 组合 churn 会把 `target/debug/build/<pkg>/<hash>` 单元目录
堆到天文数字(实测 photocraft-engine 一个包 838 个单元、41GB;cargo 稳定版无 GC)。
每周或大实验后清一次:

```sh
cd target/debug && rm -rf incremental   # 增量缓存,零风险
find build -maxdepth 2 -mindepth 2 -type d -mtime +0 -print0 | xargs -0 rm -rf   # 24h 前的陈旧单元
```

判别与预防:全量 `du -sh target/*`;日常测试可 `CARGO_INCREMENTAL=0` 抑制
最大的增量缓存再增长(实测 60GB)。陈旧单元与"合并后 E0425 而 -p 通过"是同一
现象两面(见 macos-dev.md §5)。
