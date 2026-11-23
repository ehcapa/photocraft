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
| `facedetector.onnx` | SCRFD det_10g 人脸检测(insightface buffalo_l 权重,immich-app 托管;仓库内路径 `detection/model.onnx`,含 5 点关键点,letterbox 640²) | 16MB | `fa3e5d8c…356fba56` | [immich-app/buffalo_l](https://huggingface.co/immich-app/buffalo_l) → `resolve/main/detection/model.onnx` | 代码 MIT;insightface 权重**非商用(研究用途)** | Detect(自动放框;无 roi 时的最大脸) |
| `feynobg.onnx` | FeyNoBg/BiRefNet 抠图(**ort 档专用**,需 ort feature 编译:app `--features ort`,engine-core 侧 `-p photocraft-engine-core --features ort`(2026-11-23 拆分后;facade 不再直接依赖 ml);fp16 导出) | 563MB | `c9634e83…7885fcef` | [PinkPixel/fey-nobg-onnx](https://huggingface.co/PinkPixel/fey-nobg-onnx) / 同 mirror 路径 | Apache-2.0(Fey Inc 官方,BiRefNet 底座) | Person matte(tract 上 91s 不可用 → ort ~10s;**fp32 已排除:>15min**) |
| `gfpgan.onnx`(版本无关槽位) | GFPGAN **v1.4**(neurobytemind 导出;当前占用槽位,hash 即此版) | 340MB | `42a4f8e1…37a25d08` | [neurobytemind/GFPGANv1.4.onnx](https://huggingface.co/neurobytemind/GFPGANv1.4.onnx) / 同 mirror 路径 | Apache-2.0(训练数据 CelebA-HQ 照旧待审) | 面部增强(待集成) |
| `selfiemulticlass.onnx` | MediaPipe selfie multiclass 全身六类分割(bg/hair/body-skin/face-skin/clothes/others;Pedrojok01 的 tflite→ONNX 转换,256² NHWC 0..1,**输出 raw logits,softmax 由我们跨末轴施加**——2026-11-20 探针实测 −1.7…3.8,spike 的“softmax 已烘焙”是只瞟了高置信像素) | 16MB | `847b94b7…03c3e31` | [Pedrojok01/linux-broadcast](https://github.com/Pedrojok01/linux-broadcast) → `models/selfie_multiclass.onnx`(GitHub 直连不通时走 contents API raw 媒体类型) | **Apache-2.0**(MediaPipe 官方;转换为格式改写) | Body 区域(**已接入** 2026-11-19:`photocraft-ml::body`,按 person bbox 裁剪采样(2026-11-20,不再全文档),body/face-skin→Skin 经 0.5→0.75 置信地板与小组件抑制后、hair→Hair、clothes→Clothes 按 `Analysis::union` 逐像素取大并入,person 门控;tract ~50ms。验收:astronaut Skin 3.8%→5.3%(脖子并入)。**2026-11-20 起降级为 fashion 档的 fallback**——schp-atr 在场时六类档不再跑(union 会保留其锈色颈部 0.908 误判,见 decisions.md 2026-11-20)) |
| `schp-atr.onnx` | ATR-18 服装分割(SegFormer-B2 训于 ATR;HF `mattmdjaga/segformer_b2_clothes` 的官方 ONNX;512² **NCHW 0..1 ImageNet 归一化**(mean .485/.456/.406,std .229/.224/.225),输出 [1,18,128,128] **raw logits,softmax 归我们**,需钉输入 fact(动态导出);**arm/leg 几何类只烧裸皮**(2026-11-20 探针验证),hat/sunglasses/belt 无证据不映射) | 110MB | `2e34f72e…a1662b2` | [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) → `resolve/main/onnx/model.onnx` / `hf-mirror.com` 同路径 | 代码 Apache-2.0;权重训自 ATR(**研究用途,商用待审**,faceparser 同档) | Fashion 区域(**已接入** 2026-11-20:`photocraft-ml::fashion`,person bbox ×1.25 裁剪采样,tract **~0.5s/run**;upper/skirt/pants/dress→UpperClothes/LowerClothes/Dress + Clothes、L/R shoe→Shoes、bag→Bag(手持物语义)、scarf→Scarf、face/arm/leg→Skin、hair→Hair;`select.region` 新增六个区域名,常驻探针 `cargo run -p photocraft-engine --example fashion_probe`) |

- **换版测试**:`gfpgan.onnx` 是无版本槽位,v1.x 系列接口通用(512²/0..1/RGB)——
  下载 v1.3 后改名占用槽位即可运行,加载时打一行指纹警告(不阻断);定型后把新
  hash 登记进 `ModelFile` 常量与本表。跨家族(CodeFormer/RestoreFormer)必须先过
  spike 验输入契约。
- 仓库内嵌(<50MB,提交):`crates/ml/data/u2netp.onnx`(4.6MB)、
  `crates/ml/data/facemesh.onnx`(5.1MB,MIT)、
  `crates/ml/data/poselandmark.onnx`(12.8MB,Apache-2.0,MediaPipe Pose Landmarks
  Detector / BlazePose full 的 33 关节导出,`photocraft-ml::pose`;NHWC 256² 0..1、
  person bbox 外扩正方形裁剪、vis 为 raw logit 由我们过 sigmoid,~35ms)——不走本机制。
- 完整 blake3:`faceparser.onnx` =
  `c1f14e23404535fe2b685dfab09abdb9067b823da32110de4cdb12344d0a8b8e`;
  `facedetector.onnx` =
  `fa3e5d8c62722a7b7122dc185245236589dcf0b3b5c9c5561cc742ce356fba56`;
  `selfiemulticlass.onnx` =
  `847b94b7812e47cd0daa22986bd1a7fe9fdf846026c38a1922746b78003c3e31`;
  `gfpgan.onnx`(v1.4)=
  `42a4f8e1a701cd6d56b2087ab00de1f76f4f7c1fe015556fa496d19b37a25d08`;
  `schp-atr.onnx` =
  `2e34f72ec449124cbf9b4f01d44fb51792a84b2da8a3e630445bca792a1662b2`。
- 输入契约差异(prep 各记各的,详见
  [2026-11-18-portrait-pipeline.md](2026-11-18-portrait-pipeline.md)):parsing 要
  **raw 0–255**,GFPGAN/selfie/pose 要 **0..1**,schp 要 **0..1 + ImageNet 归一化**。
- 新模型入清单:下载 → `cargo run -p photocraft-ml --example fingerprint -- <file>` →
  加 `ModelFile` 常量 + 本表一行。
