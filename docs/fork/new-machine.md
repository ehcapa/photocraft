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
curl -LO https://hf-mirror.com/neurobytemind/GFPGANv1.4.onnx/resolve/main/GFPGANv1.4.onnx
mv GFPGANv1.4.onnx gfpgan.onnx                                          # 340MB
curl -LO https://hf-mirror.com/PinkPixel/fey-nobg-onnx/resolve/main/model_fp16.onnx
mv model_fp16.onnx feynobg.onnx                                         # 563MB,仅 ort 档需要
```

清单/指纹/许可:`models.md`。**不下载任何模型也能跑**——管线自动降级(仅 Person
切片 + 唯一/提亮等 landmark 功能,mesh 内嵌在仓库)。

## 3. 环境变量

```sh
# ~/.zshrc(或运行配置)三行:
export PHOTOCRAFT_MODELS_DIR=~/Project/Rust/models
export CRAFT_I18N_DIR=/Users/<你>/Project/Rust/craft-i18n   # 中文 UI,路径按新机器改
# ort 档(仅 feynobg 需要)不是变量,是编译 feature,见下
```

## 4. 构建

```sh
cargo build -p photocraft                                   # 纯 Rust 默认档
cargo build -p photocraft --features photocraft-ml/ort      # + ort 档(feynobg)
#   首次构建 ort 会从 pyke CDN 下载 onnxruntime 二进制(需网络)
```

## 5. 测试(macOS 专属注意,见 macos-dev.md)

```sh
# 单元测试必须 workspace 级入口(单包 -p 会踩 egui/serde 统一坑):
cargo test --workspace --lib -- portrait
# 带模型两态:
PHOTOCRAFT_MODELS_DIR=~/Project/Rust/models cargo test -p photocraft-engine --lib portrait
# 带 ort(三态,全套 ~2 分钟,每个 create 真跑 10s matte):
PHOTOCRAFT_MODELS_DIR=~/Project/Rust/models \
  cargo test -p photocraft-engine --lib --features photocraft-ml/ort portrait
# 全量回归里 automation 的 trusted-root 失败是 macOS 已知预存问题,忽略
```

## 6. 真图验收(换机测试的核心)

```sh
PHOTOCRAFT_PORTRAIT_PHOTO=/任意/人像.jpg \
PHOTOCRAFT_MODELS_DIR=~/Project/Rust/models \
PHOTOCRAFT_PORTRAIT_OUT=/tmp/out.png \
  cargo test -p photocraft-engine --lib -- real_photo
# → 全开管线(修复+磨皮+发色+唇色+提亮+追色)→ 压平导出 PNG,计时打在 stderr
# 类语义诊断(如需重验映射):
PHOTOCRAFT_PORTRAIT_PHOTO=/任意/脸部裁剪.jpg PHOTOCRAFT_MODELS_DIR=... \
  cargo run -p photocraft-engine --example class_probe
```

## 7. 应用内手动冒烟

`cargo run`(要 ort 则加 feature)→ 打开一张人像 →
Layer › New Adjustment Layer › Smart Portrait… → 对话框调参 → OK →
进度条(Esc 可取消)→ 图层面板出现 "Portrait" 组,逐层可调。

## 8. 已知等待项(测试时留意)

- GFPGAN/磨皮/发色的**观感**未做过真人目检——今晚测试的主要目的
- 19 类里 14(neck)等未映射;眼/唇区域来自 landmark 环(此导出不分眼/唇)
- matte(ort 档)~10s/张、峰值内存 ~5GB;批量全开 ~15–20s/张
