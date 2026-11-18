# 运行时模型清单(models manifest)

超过 50MB 提交线的模型权重**不进仓库**,由用户放置、代码发现与校验
(`crates/ml/src/models.rs`)。**无网络代码**——下载是用户自己的动作(默认无网络,
对齐 upstream #41 的精神;显式"Download…"按钮留作二期)。

## 放置方式

```sh
export PHOTOCRAFT_MODELS_DIR=/path/to/models        # 唯一的变量,无默认值
# 文件按下方清单的"规范文件名"命名放入。
# 试验其他导出:符号链接即可,ln -s /任意/新导出.onnx $PHOTOCRAFT_MODELS_DIR/faceparser.onnx
```

加载时按清单记录的 blake3 校验:**不匹配警告但不阻断**(坏文件会在加载器里自然
失败;用户故意指向自己的导出也不拦)。无目录/无文件 → 对应阶段静默降级。

## 清单

| 规范文件名 | 模型 | 体积 | blake3 | 来源(官方 / hf-mirror) | 许可 | 消费阶段 |
|---|---|---|---|---|---|---|
| `faceparser.onnx` | BiSeNet face-parsing(zllrunning 权重,bluefoxcreation 导出;原名 `faceparser_sim.onnx`) | 52.6MB | `c1f14e23…d0a8b8e` | [bluefoxcreation/Face_parsing_onnx](https://huggingface.co/bluefoxcreation/Face_parsing_onnx) / `hf-mirror.com` 同路径 | 代码 MIT;权重训自 CelebAMask-HQ(**研究用途,商用待审**) | Parsing |
| `feynobg.onnx` | FeyNoBg/BiRefNet 抠图(**ort 档专用**,需 `--features photocraft-ml/ort` 编译;fp16 导出) | 563MB | `c9634e83…7885fcef` | [PinkPixel/fey-nobg-onnx](https://huggingface.co/PinkPixel/fey-nobg-onnx) / 同 mirror 路径 | Apache-2.0(Fey Inc 官方,BiRefNet 底座) | Person matte(tract 上 91s 不可用 → ort ~10s;**fp32 已排除:>15min**) |
| `gfpgan.onnx`(版本无关槽位) | GFPGAN **v1.4**(neurobytemind 导出;当前占用槽位,hash 即此版) | 340MB | `42a4f8e1…37a25d08` | [neurobytemind/GFPGANv1.4.onnx](https://huggingface.co/neurobytemind/GFPGANv1.4.onnx) / 同 mirror 路径 | Apache-2.0(训练数据 CelebA-HQ 照旧待审) | 面部增强(待集成) |

- **换版测试**:`gfpgan.onnx` 是无版本槽位,v1.x 系列接口通用(512²/0..1/RGB)——
  下载 v1.3 后改名占用槽位即可运行,加载时打一行指纹警告(不阻断);定型后把新
  hash 登记进 `ModelFile` 常量与本表。跨家族(CodeFormer/RestoreFormer)必须先过
  spike 验输入契约。
- 仓库内嵌(<50MB,提交):`crates/ml/data/u2netp.onnx`(4.6MB)、
  `crates/ml/data/facemesh.onnx`(5.1MB,MIT)——不走本机制。
- 完整 blake3:`faceparser.onnx` =
  `c1f14e23404535fe2b685dfab09abdb9067b823da32110de4cdb12344d0a8b8e`;
  `gfpgan.onnx`(v1.4)=
  `42a4f8e1a701cd6d56b2087ab00de1f76f4f7c1fe015556fa496d19b37a25d08`。
- 输入契约差异(prep 各记各的,详见
  [2026-11-18-portrait-pipeline.md](2026-11-18-portrait-pipeline.md)):parsing 要
  **raw 0–255**,GFPGAN 要 **0..1**。
- 新模型入清单:下载 → `blake3` 指纹 → 加 `ModelFile` 常量 + 本表一行。
