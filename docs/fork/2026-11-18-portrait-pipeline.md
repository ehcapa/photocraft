# 2026-11-18: 智能人像管线(设计选型,未实施)

像素蛋糕式人像精修的完整管线选型。**本文是设计记录,尚未落码**;实施在
`feat/portrait-retouch`(分支模型见 [decisions.md](decisions.md) 2026-11-16 条)。
初始技术方向(用户提出):FeyNoBg 抠人 → Sapiens2 语义分区 → MediaPipe
FaceMesh/Pose → 皮肤重绘 → GFPGAN/CodeFormer/GPEN 面部增强 → HSV/局部重绘换发色 →
SD/Flux Inpainting 背景与衣物 → 追色。评估结论:方向成立,两处结构性修正
(DAG 化、图层落位),若干选型降级。

## 管线结构:DAG,不是链

```
分析层(缓存元数据,不产出可见结果):
  抠人(FeyNoBg) ─┬→ 语义分区(人像 parsing)
                  └→ FaceMesh/Pose(关键点)
执行层(并行分支,互不依赖):
  ├─ 面部增强(GAN)→ 皮肤处理(顺序固定,见下)
  ├─ 换发色(HSV on hair mask)
  └─ 背景/衣物修复(inpaint)
调色层(最后,对合成结果全局):
  追色
```

- 每阶段声明消费的区域(skin/hair/face/clothes/background/whole),改参数只重跑
  该阶段——缓存失效的粒度。
- **面部增强必须在皮肤处理之前**:GAN 重新生成皮肤纹理,先磨皮后增强等于白做。
- **追色殿后**:对合成后的最终像做,否则生成阶段一重跑就破坏调色。

## 阶段落位(L1 经典 / L2 纯 Rust 小模型 / L3 provider)

| 阶段 | 层 | 体积 | 评价 |
|---|---|---|---|
| FeyNoBg 抠人 | L2 | ~几十 MB? | 接 subject 阶梯同一条线(modnet/BEN2 候选在位)。⚠️ 名目、ONNX 可得性、许可待核实 |
| Sapiens2 语义分区 | **L2 越界→降级** | 0.3B+/GB 级 | tract 跑 attention 重模型数倍慢于 ORT。人像精修核心是**脸部分区**(skin/hair/eye/lip/clothes),face-parsing BiSeNet 级(<20MB)覆盖 90%;全身衣物分区才需要 Sapiens 级,推迟到有真实需求再评估 |
| MediaPipe FaceMesh/Pose | L2 | 3–5MB | 合理。注意 YuNet 五点检测已在仓库(`crates/algo/src/faces.rs`,engine 有 face_cmds/Face-Aware Liquify),FaceMesh 是加密细化不是替代 |
| 皮肤处理 | **L1** | 零 | **不走扩散**(身份漂移)。频分/导向滤波磨皮(经典,可回馈上游)+ 现有 healing 去病灶 |
| GFPGAN/CodeFormer/GPEN | L2 边界 | 350MB+ | ONNX 均有;tract 跑 GAN 速度必须实测。CodeFormer 质量最稳(w 旋钮保身份,默认高保真),GFPGAN 最快。⚠️ 三者皆研究许可系,商用逐个查 |
| HSV 换发色 | **L1** | 零 | hair mask + HueSat,纯经典,直接 upstreamable |
| SD/Flux Inpaint | **L3** | GB 级 | tract 不现实。走 fork 已定的 L3 provider 架构(对齐 upstream #41 六条:provider trait 独立 crate、默认无网络、命令化、密钥进钥匙串、隐私文案、无 provider 不显示菜单);本地 provider 用 SD1.5/SDXL-inpaint 侧车进程,Flux 仅云端 |
| 追色 | **L1** | 零 | Reinhard/直方图匹配 + 分区 mask,可回馈上游 |

## 图层与代码落位:标准图层 + 编排器(不加 Adjustment variant)

**否决方案**:给 `crates/doc/src/adjust.rs:215` 的 `Adjustment` enum 加
`PortraitRetouch{…}` variant。该 enum 是 **PSD 往返序列化格式**的一部分:fork variant
= 文档格式永久分叉 + 每次 upstream 同步的固定冲突点 + PSD 兼容性债,与"尽量独立"
的目标相反。

**采纳方案**:fork-only 编排 crate(暂名 `crates/portrait`)创建**标准图层**:

| 管线阶段 | 落成的图层 |
|---|---|
| 换发色 | `HueSaturation` 调整层 + parsing 生成的头发蒙版 |
| 追色 | Curves/色彩平衡 + 分区计算参数 |
| 磨皮 | 现有 surface blur 智能滤镜(filter op 链路自带 undo/选区/智能滤镜) |
| 面部增强 / inpaint | 栅格层(或智能滤镜缓存栅格),进同一图层组 |
| 抠像/分区/关键点 | **不进文档**:缓存在编排器 sidecar 数据,重跑按阶段失效 |

- 共享代码触点收到一处:菜单注册(Layer › New Adjustment Layer 组里加"智能人像",
  但产出是标准图层组,不是新图层类型)。engine 侧走已打通的 filter op 链路
  (algo `FilterParams` → engine `filters.rs` → ui `menu_catalog`/`filter_dialog`)。
- 产出形态与像素蛋糕一致:用户拿到**一组可继续手调的图层**,不是黑盒。PSD 兼容、
  upstream 同步、可编辑性三个方向同向。

## 风险与待验证

1. **tract 逐模型可行性 spike**(最高优先):每阶段候选模型取 2–3 个,2–3 张参考
   人像实测 load/推理耗时/内存。subject 当时的阶梯+回退纪律对每阶段适用。
   重点:GFPGAN/CodeFormer 的 GAN 算子与速度;face-parsing 候选的精度。
2. **许可表**:商用口径逐模型过一遍(GFPGAN/CodeFormer/GPEN 研究许可系;FeyNoBg
   未知;face-parsing 实现多为 MIT/Apache 但预训练权重单独看)。
3. **模型管理器**:>50MB 模型一律运行时下载 + hash 校验(u2netp 4.6MB 直接提交是
   上限先例,BiRefNet 200M+ 参数被排除同理);350MB 级面部增强模型必然走这条路。
4. 身份漂移:面部增强默认保守(CodeFormer w 高保真端),暴露保真度旋钮。
5. 性能预期:多阶段各秒级,必须后台线程 + 分阶段进度 + 可取消(subject 的
   取消机制先例)。

## 下一步

1. ~~`crates/portrait` 骨架~~ ✅ 2026-11-18 完成(垂直切片可用:`portrait.create` →
   Person 蒙版 HueSat 层 → "Portrait" 组,单 undo 步,6+10 测试)。
2. ~~ Parsing 阶段 tract spike ~~ ✅ 见下节;**同日接入** ✅:`photocraft-ml` 新增
   `parsing_probs`(env `PHOTOCRAFT_PORTRAIT_PARSING=<onnx路径>` 运行时加载、失败缓存
   None、raw 0–255 输入契约、逐像素 softmax 后 19 概率面),engine 的
   `parsing_regions` 把类面按 `photocraft_portrait::parsing::CLASS_REGIONS`
   (1=Skin、15=Clothes 待核、16=Hair 已证)组成区域、与人像区域取交集(域外预测被
   主体门压制)后上采样进 Analysis;`portrait.create` 带 `hairHue` 等参数即产第二个
   "Hair Color" 层。测试:ml 8✓(含模型冒烟)、portrait crate 13✓、engine 7✓×
   (有/无模型两态)。
3. L1 三件套(磨皮/换发色/追色)——换发色已随 Parsing 落地(Hair 蒙版 + HueSat);
   磨皮(Skin 蒙版 + 频分/导向滤波)与追色待做,零模型依赖,可切小分支回馈上游。
4. 面部增强 spike 定阶梯;L3 provider(SD inpaint)按 #41 对齐另立档案;模型管理器
   (下载 + hash)把 env 路径升级为正式入口。

## Parsing 阶段 spike 结果(2026-11-18,i7 桌面,tract-tflite/onnx 0.23,release;同日质量复验后定稿)

| 候选 | 体积 | 许可 | 结论 |
|---|---|---|---|
| **BiSeNet face-parsing**(zllrunning 权重;两个独立导出:litert-community tflite、bluefoxcreation onnx) | 52.6MB | 代码 MIT;权重训自 CelebAMask-HQ(**数据集研究用途,商用许可表待过**;bluefox 导出标 license:other) | ✅ **首选梯级**,速度与质量双过(下) |
| MediaPipe selfie_multiclass_256x256(float32) | 16.4MB | Apache-2.0 | ❌ tract-tflite 0.23 无法翻译 `TRANSPOSE_CONV`。类映射本最理想——等 tract 支持后重试 |
| SegFormer(jonathandinu / Xenova face-parsing onnx) | 89–340MB | 各异 | ❌ 体积超预算,未测 |

**速度**:tflite 60ms / onnx 74ms @ 512² NCHW(u2netp 的 1/9–1/11 耗时、更大输入);
加载+优化 31–79ms(OnceLock 进程缓存摊销);峰值 RSS ~344MB。完全交互级。

**质量**(真实人像学位照验证):tflite 与 onnx 两个独立导出**逐类吻合**
(皮肤 41.8%/40.8%、头发 22.2%/22.3%、背景 20.7%/21.8%,空间位置全部解剖合理:
皮肤居中、头发环上方、背景偏侧)——tract 翻译忠实,非转换伪影。

**输入契约(踩坑记录)**:此导出族把归一化**烘焙进图里**,要喂 **raw 0–255 RGB、
NCHW 512²**;README 宣称的 ImageNet 归一化(/255 或 mean/std)反而输出人形大团块
(仅人物/背景二分、细节类全灭)。接线时以 raw 为准,并留一致性测试钉住。

**遗留**:① 19 通道的确切类语义(标准 CelebAMask-HQ 是 18 类)未定——小类
(眼/眉/唇,各 <2%)无法用质心判定,接入时需带标注样张对照或按 zllrunning 推理脚本
定序;② 商用许可审查;③ 模型 52.6MB 走运行时下载 + hash,不提交。

- bench 为丢弃型工程(`/tmp/portrait-spike`,不在仓库):双格式加载、布局自动识别、
  类直方图 + 质心 + 逐通道统计、彩色类图、归一化模式参数。
- 网络注意:huggingface.co 本机不可达,经 hf-mirror.com 镜像拉取;
  storage.googleapis.com 直连可用。
