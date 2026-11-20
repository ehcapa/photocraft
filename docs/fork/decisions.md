# Fork 决策日志

fork 级产品与架构决策，`## 日期: 标题`，背景 / 决策 / 后果；细节指向
[`YYYY-MM-DD-<slug>.md`](.) 主题档案。回馈上游的决策按上游格式在
[`../decisions.md`](../decisions.md) 补条目，对应条目只留链接。

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

（完整条目在 `feat/portrait-retouch` 分支的旧 `docs/fork-decisions.md` 上；两分支合并时
移入本文件。）

---

## 2026-11-18: ML Select Subject——tract + U²-Netᵖ 提交权重，经典管线保留回退

**背景。** 现有 Select Subject 是纯经典启发式（SLIC 显著性 → Otsu → GrabCut），
不准的根因是无语义（显著性 ≠ 主体，#2788）、Otsu 双峰假设 + 边框硬编码背景、
组件过滤丢第二主体。上游 roadmap M10 把 `ml` crate 规划为 ort（C++ 绑定），
#41 owner decision 冻结中。

**决策。** 运行时用 tract（纯 Rust、MIT/Apache 双许可、直接加载 ONNX）不用 ort；
u2netp.onnx 4.6MB 直接提交仓库（libfacedetection.bin 先例，>50MB 的模型走运行时
加载不提交）；ML 粗掩码只替换种子阶段，GrabCut + 全分辨率带状重切保留精边，
缺失/失败/空掩码一律静默回退经典管线；algo（L2）不依赖 ml（L4），engine 取掩码后
作为数据参数下传；wasm 全门控（依赖与权重均 cfg 掉）；单测默认经典化（`!cfg!(test)`），
集成测试与 panic_hunt 走真模型；`PHOTOCRAFT_SUBJECT_ML=0` 可关。

**后果。** 双主体、贴边主体、满幅主体这些经典系统性失败模式全部修复；代价是推理
~680ms/次与仓库 +4.6MB。选型细节、模型升级阶梯（silueta/MODNet/BEN2；isnet 权重
不开放、BiRefNet 太大均排除）、调试途中修复的两个算法 bug（maxflow 负容量、
种子 sliver 污染）见
[主题档案](2026-11-18-ml-subject.md)。

## 2026-11-18: Select Subject 的 +/- 行为对齐 Photoshop

**背景。** 选项栏 Subject 按钮写死 `replace` 不跟随模式按钮；菜单项不支持修饰键——
与 Photoshop 不一致（PS 按钮跟随 +/- 状态，菜单 Shift=加/Alt=减，Sky 同款）。

**决策。** 按钮经 `subject_mode()` 复用所属工具的模式解析规则；菜单派发点对
`select.subject`/`select.sky` 注入按住修饰键的 mode，显式 mode 参数优先；Properties
快捷动作保持替换。细节见[主题档案](2026-11-18-ml-subject.md)。

**后果。** upstreamable 的纯 UI 行为修复（无 AI 依赖）：回馈上游后在
[`../decisions.md`](../decisions.md) 补条目，此处改留链接。

## 2026-11-18: 智能人像管线——DAG 编排 + 标准图层，不加 Adjustment variant

**背景。** 像素蛋糕式人像精修要做成"一个智能人像调整层"加进调整层组，约束是长期
同步 upstream（日均数十 PR）与 PSD 往返兼容。初始方向为八段串行链
（FeyNoBg 抠人 → Sapiens2 分区 → FaceMesh/Pose → 皮肤 → GAN 面部增强 → 换发色 →
SD/Flux inpaint → 追色）。

**决策。** 管线按 DAG 建模：分析层（抠像/分区/关键点，缓存元数据）→ 执行层并行
分支（面部增强在皮肤之前；换发色；inpaint）→ 追色殿后。阶段分层：磨皮/换发色/追色
L1 经典（可回馈上游）、抠像/关键点/face-parsing/面部增强 L2 tract（Sapiens2 越界
降级）、SD/Flux L3 provider（对齐 upstream #41）。**不给 `doc::Adjustment`（PSD
序列化格式）加 fork variant**：fork-only 编排 crate 创建标准图层（调整层 + 生成
蒙版、智能滤镜、缓存栅格）+ sidecar 分析缓存，产出可手调图层组。

**后果。** 共享代码触点收到菜单注册一处，doc 格式零分叉；生成类阶段以缓存栅格
落地、按阶段失效重跑。完整落位表、风险与 spike 待办见
[主题档案](2026-11-18-portrait-pipeline.md)。

## 2026-11-19: Smart Portrait 对话框不再记忆上次配方

**背景。** 真图验收首晚出现"处理后完全不可用"的导出：全幅青灰罩层、绿色腮红块、
紫色眼带。逐层转储（portrait_diag）证明管线无罪——默认配方下六个层蒙版全部解剖学
正确、输出与原图差 ~1/255；元凶是 schema 对话框的"记住上次值"机制：前一晚实验留下
的 `hue=164/saturation=-61/lightness=+43`（人物整体层）+ `blushHue=91`（绿腮红）+
`shadowHue=286`（紫眼影）被原样载入执行，用这组残值回放与坏图逐通道一致（残差
1.4 = JPEG 噪声）。

**决策。** `portrait.create` 的全部对话框行 `not_remembered()`（filter_dialog 的
`params()`），每次打开取 spec 默认值；"重复上次配方"的需求由既有的
`portrait.saveRecipe` → Actions/Batch 承担——那才是带名字、可审计的配方载体。
`crates/engine/examples/portrait_diag.rs`（逐层 solo 合成 + 蒙版 PGM 转储）作为
fork 诊断工具保留，与 class_probe 同列。

**后果。** 用户配置里已落盘的 `[sections.remembered."photocraft.portrait.create"]`
残值自然失效（重开时按行过滤，不回填）；新增测试
`smart_portrait_never_remembers_its_recipe_rows`；排障口诀进
[new-machine.md](new-machine.md) §8：输出离谱先查参数来源，再怀疑管线。

## 2026-11-19: 选区工具的模型区域化——共享分析、八区域选区、发丝档

**背景。** 人像管线的分析(主体 → SCRFD 检测 → FaceMesh → 脸部 parsing → 全身
selfie_multiclass)只服务 Smart Portrait 一家,而选区工具侧只有内置 U²-Netᵖ 的
Select Subject。全身皮肤/衣服/五官等区域没有选区入口;FaceMesh/BiSeNet 直接怼全图
时输出崩塌(妆面落在下巴以下)也缺少 crop 级。

**决策。** 三件事:(1) `analyze_person` 从 portrait.create 的 job 闭包提取为共享
函数,`select.region`(皮肤/面部/头发/眼睛/眼睑/嘴唇/面颊/衣服八区域)与 Smart
Portrait 共用同一份像素缓存分析——第一次付几秒,之后跨命令瞬时;(2) `roi`
参数(用户框)与 SCRFD 自动放框并进 mesh/parsing 的 crop 级,two-pass 收紧;
`select.subject` 的 `matting` 参数让 FeyNoBg alpha 直接成为软选区(发丝档);
(3) UI 上这些动作是**选区族共用段**(十个选区工具的选项栏 + Select 菜单
`Person ▸`),Subject 一个按钮 + 标准/发丝档位段控件(PS 的 Cloud/Device 先例),
区域是 SplitButton。全身六类按 `Analysis::union` 逐像素取大并入既有区域
(身体皮肤并入 Skin → 磨皮自动覆盖全身),而非新开区域。

**后果。** `selfiemulticlass.onnx`(Apache-2.0)成为身体区域的既定来源;
FASHN human parser 明确不引入(NVIDIA 非商用 + 专有训练数据,见
[models.md](models.md) 许可列)。选区语义保持:软 alpha、±/交集、模式修饰键照常。

## 2026-11-19: faceparser 导出的类语义不可信——区域集封顶为八类

**背景。** BiSeNet faceparser.onnx(CelebAMask-HQ 19 类)的类别映射自引入起标注
"顺序未验证"(face.rs 曾以 landmark 环绕过眼/唇)。脖/耳/眉/鼻四类是否可解锁,
决定区域选区的上限。

**决策。** 用 `class_probe`(landmark 锚定的环/带覆盖法)在真图上重验:该导出
19 类里只有 6 类会响应,且**自相矛盾**——左眼环 78% 被 class 17 占据、右眼环
100% 被 class 1 占据。结论:细类语义不可恢复,CLASS_REGIONS 的保守映射
(1→Skin、16→Clothes、17/18→Hair)就是可辩护上限;脖/耳/眉/鼻正式判为不可给,
眼/唇继续用 FaceMesh 环。

**后果。** 区域选区全集封顶为八类(见上条);要解锁四细类只能换可信导出或自训,
记录在案避免重复验证。

## 2026-11-20: select.region 质量重建——softmax 归位、body 采样收缩、Face/Cheeks 改宗 parsing

**背景。** 两张真实人像验收 select.region:body 类区域(skin/hair/clothes)全线
崩坏——无结构色块、椒盐、方形接缝、锈色围巾高置信误判;Face 是 mesh 几何椭圆,
Cheeks 是颊点直立椭圆,精修不可用。逐阶段 mask 落盘(region_dump_diagnostic)
+ 窗口置信度探针(body_confidence_probe)定位。

**决策。**

- **softmax 归我们**:selfiemulticlass 导出输出 raw logits(探针实测 −1.7…3.8),
  spike 的"已烘焙"结论作废;`body_probs` 跨类 softmax + 契约测试(0..=1、六类和 1)。
  实测修复后围巾 mean 0.304 vs 脸 0.900——地板从此有意义。
- **body 阶段收缩**:按 person matte bbox(×1.25)裁剪采样,不再全文档(治全景
  小人物已知边界);0.5→0.75 线性置信地板;连通域小组件抑制(<2% 最大组件且
  <16px);Skin/Hair 在 mesh ring 内让位 parsing(杀双源拼接直边)。
- **Face 改宗 parsing**:CelebAMask skin 平面即发际线级脸皮肤,hull 仅在 parsing
  无 skin 时兜底;**Cheeks 同宗**——parsing skin × 颊带(眼线下 18% 起、嘴线上
  8% 止)− 眼环/唇环 − 鼻列(内 70% 平零、外 30% smoothstep),椭圆退兜底。
  依据:2026-11-19 已证该导出细类不可信,但 1(skin)可信且边界质量最高。
- **验收方法学**:env 门控诊断测试(逐区域 PNG+JSON、窗口探针)+ PIL 叠加图
  人审,替代"瞟几个 confident 像素"。

**后果。** 两张验收图:skin 0.219→0.091(围巾大 blob 消失、头发/大衣/背景干净),
face 变像素级发际线追踪,cheeks 双颊独立 patch 鼻/眼/唇全排除;残留为模型级
(围巾 0.908 高置信像素、身体皮肤在小人脸图上仍低于地板)——下一档是 SCHP
(ATR-18/LIP-20,512²,scarf/upper-clothes 独立类;其 arm/leg 为几何部位类的
映射坑 spike 时验),Sapiens 0.3B(ort 档)列第二梯队。

## 2026-11-20: 服装细类解封(SCHP ATR-18 落地)+ Pose 基础层

**服装 tier。** 上条留的下一档落地:选 HF `mattmdjaga/segformer_b2_clothes` 的
ONNX 导出(SegFormer-B2,ATR-18 训练,110MB,runtime slot `schp-atr.onnx`,
blake3 `2e34f72e…a1662b2`)。`fashion_probe`(engine example,常驻)验证:
tract 钉输入后可载,**~0.5s/run**(注意力架构远快于预期,3s 门内);契约 =
导出自身 preprocessor(ImageNet 归一化 0..1 NCHW 512²,raw 值 −13…9,softmax
归我们);类语义经窗口探针 + 视觉模型 ground truth 双向核实(坐姿照 dress/裸腿
left-leg/长发/抬手全对,宇航服照袖子无 arm blob、颈部黑色高领读 scarf)。

**arm/leg 映射坑:通过,变体 1。** 几何部位类只烧裸皮(裸腿→leg、裸前臂→arm,
着装手臂不触发),入 `Skin`;永不成为独立区域。

**优先级链,不是 union。** `Analysis::union` 是逐像素 max,SCHP 与
selfiemulticlass 并跑会保留六类档的"锈色颈部 0.908 读成皮肤"(union 救不了
错的一方)。故 **fashion 有答案 → 它接管 Clothes/Skin/Hair + 服装子区域,
selfie 降为缺席 fallback**。验收(test-portrait.png):scarf 1.01% 落颈部,
skin bounds 止于 y=208(=脸下缘,0.908 错误随 selfie 不跑而消失),
upperclothes 22.07% 紧贴其下;test-user.jpg(全身):dress 5.36% 与探针 blob
同位、skin 延到画面底(裸腿 union)、裙装时 upper/lower 正确为空。

**区域词汇表:对 2026-11-19 八类封顶的定向解封。** 封顶是 faceparser 导出
类语义不可信的裁决,不是产品上限;服装类经逐类探针挣得区域——`UpperClothes`
(ATR upper-clothes)、`LowerClothes`(skirt∪pants)、`Dress`、`Shoes`
(L∪R shoe,位置邻接验证、未见实景鞋照,复验待补)、`Bag`(ATR 手持物语义,
手里的头盔也读 bag,已验)、`Scarf`。hat/sunglasses/belt 无探针证据,不映射。
每服装类同时喂 `Clothes`。select.region 的 match/spec、PERSON_REGIONS、
menu_catalog、选项栏 SplitButton 四点同步,command_ids.txt 不动(无状态
菜单 id 非 engine 命令)。

**Pose 基础层。** MediaPipe Pose Landmarks Detector(BlazePose full,33 关节)
的社区 ONNX 导出(Apache-2.0,12.8MB,**内嵌** `crates/ml/data/poselandmark.onnx`,
facemesh 先例)。契约(README + pose_probe 验证):NHWC 256² 0..1、输出 5 张量、
screen landmarks [1,195]=39×(x,y,z,vis,presence)、**vis 是 raw logit,sigmoid
归我们**(探针读 8.5/−0.8)。裁剪:person matte bbox 的**轴对齐外扩正方形**
(`to_square`),v1 不做旋转对齐(端正照合理,倾斜退化已知并记录);~35ms。
`Analysis.pose: Vec<[f32;4]>` 存储,**无消费者**——身体液化/瘦身锚点
(PuppetPin.src / LiquifyField 均已就绪)是下一迭代;本期只到探针验收
(pose_landmarks_dump_probe:33 点、有限、vis 值域、肩<髋)。

## 2026-11-20: `filter.bodySculpt` —— 关键点驱动身体塑形(v1,对标像素蛋糕全身美型)

**对标定位。** 像素蛋糕全身美型(wiki 9412)约 19 滑杆、面部重塑(wiki 9405)
约 60 滑杆,全部是"关键点→形变"同一台机器(身体 BlazePose 33 点、面部 FaceMesh
478 点,两个模型都已内嵌)。v1 落身体面板 **16 滑杆**(瘦身/瘦腰/瘦腹/手臂/
大腿/小腿/肩宽/美臀/丰胸/小头/瘦脖子 + 增高/长腿/腰长/上身长/脖长),独立命令
`Filter ▸ Body Sculpt…`(像素蛋糕自己也是独立面板);架构按全部滑杆设计——
algo 的部件表是数据,面部面板三期只加表项。

**解析写场,不走 dab 链,不用 freeze。** 三条都有 liquify.rs 行级证据:
①身体尺度笔刷(200-400px)的 Pucker dab 链要数千 dab,智能滤镜每次重渲染重付,
且压力×间距的合成让滑杆→位移成为涌现函数,没法校准——直接 `LiquifyField::new`
+ 写 `d`,滑杆→边缘位移一阶线性;②`apply_liquify` 在 freeze>0 处把 Catmull-Rom
混向双线性(:879-886),冻结带边缘成可见折痕,各部件横向剖面自身 C1 到零
(复用 pub `falloff`)即够局部;③逆向映射本无洞,折叠才是风险,测试断言
`相邻节点|Δd| < cell`。**该断言首跑即抓住真问题**:四个长度 ramp 的横向衰减
叠在同一剪影边缘(实测 6.4px/4px)——修复 = ramp 横向改高原+羽毛(剪影内
刚体骑移零梯度,剪切移进背景)+ 五长度滑杆**共享折叠预算**(幅度和封顶
0.20×宽度,堆满等比缩而非散架)。

**锚点嵌入 params,重渲染纯数据。** SmartFilter 在合成时重渲染,没有 Session/
portrait_cache 可问——命令应用时把 pose [x,y,vis]×33 + 头部中心/半径
(FACE_OVAL 质心×1.15 含发)+ chin(FaceMesh 152)+ doc 尺寸塞进 params 随滤镜
存储;`apply_to_surface`(distort_cmds 分发臂)从 params 确定性重建场。
分层规则不破:algo 只吃纯几何,ml 依赖止于 engine。蒙版跟场走(#3083,
puppet 先例):链接蒙版同一 `apply_liquify`,瘦身不留胖蒙版。

**UI 三点。** Filter 菜单行;通用 schema 对话框(16 滑杆行,not_remembered
——portrait.create 2026-11-19 同款理由:跨会话残留的极端滑杆呈现为"管线坏了");
**预览显式退出**——验证了 preview 每 tick 建新 Session(缓存必 miss→整套分析)
且代理 k 缩放下 doc 像素锚点几何错位,修复路径(对话框开时捕获锚点 +
scale_preview_params 钩子,blur_pins 先例)记录待做。

**验收。** algo 8 测(合成站姿,CI 安全):恒等/确定性/单调/方向/ramp 连续/
vis 门控/±100 叠加平滑。engine 真图门控测试(PHOTOCRAFT_SCULPT_PHOTO,
region_dump 先例——分析链是 crate 内部,example 够不着):test-user.jpg
16 滑杆表 + 门控正确(半身照,踝/膝不可见 → calves/legs/height 正确跳过)、
**footprint 量具**(直接量图层像素,不依赖蒙版):waist-100 → 腰带 10.4%
像素可见移动,head/legs 带 0.000;蒙版 rim 量具在此照降级如实报告
(u2netp 蒙版整幅覆盖——已知主体档短板,ort 档存在的原因);智能对象存
anchors + 真路径 `apply_filter_to_surface` 两次重渲染逐位相等。校准:各部件
gain 是表常量(gain×局部宽/骨长),±100 约 10% 量级,探针表打印可调。

**延后表。** AI塑形/身型流畅(需 AI 平滑机制,另一原语)、直角肩/天鹅颈
(滑杆组合件)、胸位提升、左右分开小锁、双下巴(像素蛋糕归"祛除瑕疵"修复类,
可先近似下颌收紧)、面部面板(~60 件,FaceMesh,三期)。zh 词条走外部
craft-i18n,`tl!` 回落英文。

## 2026-11-20: body 面板补齐至对标(第二批 5 滑杆)

上条延后表里"很近"的四件 + 双下巴液化近似落地,身体面板 21 滑杆,对标像素蛋糕
全身美型 19 件全覆盖(身型流畅除外,它是平滑原语,三期):

- **aiSculpt(AI塑形)/swanNeck(天鹅颈)= 组合滑杆**,不是表行:`SculptSliders::composed()`
  在 `sculpt_field` 入口把比例混入目标件(AI塑形 → 瘦身系 7 件,髋/胸/头不参与——
  滑杆语义是剪影不是丰腴;天鹅颈 → 瘦脖/脖长/直角肩三件),叠加手动值后 clamp。
  组合放 algo 侧 = 单一确定性入口,存储的 raw 滑杆重放逐位一致,engine 零改动。
- **straightShoulders(直角肩)= Contract 换水平轴**(肩线 11→12 本身当轴),斜方肌
  坡向线收缩即"压平";**doubleChin(双下巴)= chin 下行短轴 Contract**(像素蛋糕归
  祛除瑕疵类,液化近似挂面板);两件都是一行表项。
- **bustLift(胸位提升)= 第三原语 `Shape::Shift`**:带内刚体平移(幅度均匀 + 高原
  横向 + 纵向 smoothstep 窗),逆向映射逆着骑行方向——第一个非缩放类部件。
- 真图表(半身照):aiSculpt 7 行生效且不可见件如实跳过、swanNeck 三件、
  straightShoulders 1.8px、bustLift 5.5px、doubleChin 2.6px;±100 全叠加折叠界
  仍守住。剩余延后:身型流畅(三期)、L/R 小锁(对话框无成对控件,待 UI 决策)、
  面板预设(像素蛋糕 7 脸型预设属面部面板)。

## 2026-11-20: `filter.faceSculpt` —— 面部面板(三连之二,FaceMesh 部件表)

**架构承诺兑现:**第二张部件表(`algo/face_sculpt.rs`)只加表项——四个原语
(Contract/Radial/Ramp/Shift)从 sculpt.rs 以 `pub(crate) Shape + write_shape` 共享,
engine 侧 `face_sculpt_cmds.rs` 是 sculpt_cmds 的镜像(同一缓存分析、锚点(478
mesh 点)嵌入 params、蒙版跟场、预览退出、not_remembered、智能滤镜回放钩子)。

**47 滑杆五组**(对标面部重塑主体):轮廓 16(瘦脸/太阳穴/颧骨/下颌/小脸/V脸/
脸宽/发际线/额头宽高/人中/中庭/下庭/下巴长短/尖下巴/收下巴)、眼 10(整体/眼瞳
(iris 468/473)/眼高/眼宽/眼距/内外眼角/下至/倾斜/位置,左右分行共用滑杆)、
鼻 6、嘴 8(含露龈/M唇向的丰上下唇)、眉 7。宽类滑杆正向=增宽(像素蛋糕自己的
右向),瘦类正向=收。

**两个真 bug 被测试抓住:**①`FaceFrame::mid` 写成半差的负值,嘴中心算到画面
原点,嘴角带成了对角线——折叠守护在 (150,202) 测到 64px/2px 悬崖,逐组二分到
lip_tilt 再定位;②面部特征尺度小(眼 30px vs 腰 300px),47 滑杆全 ±100 时
Shift 家族在短羽化上叠加剪切 9.4px/2px——面部自己的折叠预算(Shift+Ramp 共享
0.10×face_w,家族等比缩)收平,与身体面板长度家族同款裁决。另:奇宽区域
`chunks(3)` 残块越界(read_region 平面缓冲),body 的量具同隐患一并改
`chunks_exact`。

**验收:**合成脸 7 测(恒等/确定性/47 单滑杆/方向/±100 折叠界)+ 真图
(test-user.jpg):47 行表全部有位移(0.4-8px 量级,面级 gain 待实拍校准)、
eyeSize footprint **eye 带 69.2% 移动 chin 带 0.000**、mesh 入 params 回放逐位
相等。**延后:** 7 脸型预设(参数束)、对称液化(镜像)、3D塑颜(伪转头来
的透视重采样)、AI 面部流畅(平滑原语,与身型流畅同期)。

## 2026-11-20: 身型流畅 + AI 面部流畅(三连之三,轮廓松弛)

两面板各加一个 `smooth` 滑杆(0..100 单向),唯一非关键点形变的部件:轮廓自
身的凸起(斜方肌坡、手臂赘肉、下颌角)读出来、低通、把差值盖回场。algo
`contour.rs`:闭合多段线二项式 [1,2,1]/4 低通(次数∝强度)+ 散点插值盖印
(**falloff 加权平均,不是叠加**——叠加把相干修正放大 3-4 倍,1.6px 的质心守恒
漂移被放大成 7.4px 推挤,实测后改)+ 盖印后两遍节点域松弛(液化 Smooth 笔刷
语义,修修正曲线自身的折叠梯度)。

轮廓来源:身体 = Person 蒙版逐行左右缘(整行全宽的坏蒙版行直接不贡献——
u2netp 分不开背景的照片如实报 "no contour to relax" 而不是去平滑画框);
面部 = FACE_OVAL 环(有序闭合)。**轮廓随 params 存储**(同锚点规则),
智能滤镜回放零蒙版零分析;真图验收里 smooth:60 经智能滤镜全链路回放逐位相等。

至此对标全景:身体面板 22 滑杆(19/19 + 双下巴 + 组合件)、面部面板 48
滑杆(47 + smooth)。**仍未做:** 7 脸型预设(参数束)、对称液化(镜像)、
3D塑颜(伪转头透视)、左右分开小锁、L/R 成对锁 UI。

## 2026-11-20: 脸型预设(四期之一)

`filter.faceSculpt` 加 `preset`(none/classic/youthful/heroic/juvenile/round/long/
square)+ `presetStrength`(0..100,对话框默认 60、解析默认 0——存储态永远带显式值)。
七个束按像素蛋糕 wiki 的描述立值(经典=瘦脸25/大眼15;长脸=中庭−30/下庭−25/发际
线下移;方脸=下颌40/下巴+20…),`FacePreset::delta()` 返回滑杆差量、`composed()`
按强度混入手动值后 clamp——AI塑形 同一条确定性规则,回放逐位一致。对话框的
schema 行自动出现(preset 是 choice 参数)。

## 2026-11-20: L/R 分开调(四期之二)

`FaceSides`:21 对两侧特征(太阳穴/颧骨/眼组 10/眉组 7/嘴宽/唇倾斜)各得
`<key>Left/Right` 覆盖参数(0..100…实为 −100..100 同基础滑杆),`None` = 跟基础值
——**穷人版小锁**:成对联动直到一侧被单独赋值,锁控件(定制对话框部件)留待
UI 期。表侧零改几何:42 条左右行本就分行,滑杆 fn 改读本侧覆盖。缺省路径
(不传任何 Left/Right)行为与拆分前逐位一致。
