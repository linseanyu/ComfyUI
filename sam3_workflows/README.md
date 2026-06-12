# SAM3 在 ComfyUI 中的核心功能指南

## 1. 接入情况（无需额外安装）

你的 ComfyUI 版本：**0.24.0** ✅

好消息：你的 ComfyUI 已经 **原生支持 SAM3 / SAM 3.1**（PR #13408 已合并进核心代码）。
不需要装任何 custom_nodes 插件，只缺一个模型文件。

> 4 个原生节点在 `comfy_extras/nodes_sam3.py`：
> - `SAM3_Detect`（图像检测+分割）
> - `SAM3_VideoTrack`（视频对象跟踪）
> - `SAM3_TrackToMask`（跟踪结果 → mask 张量）
> - `SAM3_TrackPreview`（跟踪结果可视化）

## 2. 安装模型（必做）

模型下载（~3-4 GB）：
https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors

放置位置：
```
ComfyUI/
└── models/
    └── checkpoints/
        └── sam3.1_multiplex_fp16.safetensors   ← 放这里
```

下载命令（任选其一）：
```bash
# 方式 A：huggingface-cli
pip install -U huggingface_hub
huggingface-cli download Comfy-Org/sam3.1 sam3.1_multiplex_fp16.safetensors \
  --local-dir ComfyUI/models/checkpoints/

# 方式 B：wget
wget -O ComfyUI/models/checkpoints/sam3.1_multiplex_fp16.safetensors \
  https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors
```

## 3. 8 个核心功能详解（每个都带具体案例 + 工作流）

> 工作流文件都在 `sam3_workflows/` 目录下。

---

### 功能 ① 文本提示图像分割（`01_text_prompt_image.json`）

**作用**：用自然语言描述要分割的物体，SAM3 自动在图像中找到**所有**匹配的对象并输出 mask。

**节点组合**：`LoadImage` → `CheckpointLoaderSimple`（SAM3） → `CLIPTextEncode` → `SAM3_Detect` → `MaskToImage` → `PreviewImage` / `SaveImage`

**具体案例**：一张街景照片（example.png），在 CLIPTextEncode 里写 `car`，SAM3_Detect 会自动检测出图中**所有**汽车的位置和 mask。结果通过 `MaskToImage` 转换成可视化图像，预览 + 保存。
- 文本提示上限 32 tokens
- `individual_masks=False`（默认）：所有汽车 mask 合成一张；设为 `True` 则每辆车一张独立 mask

---

### 功能 ② 边界框提示分割（`02_box_prompt_image.json`）

**作用**：当你想精准切某个已知位置的对象时，给 SAM3 一个边界框（xywh 像素坐标），它只切这个框内的内容。

**节点组合**：`LoadImage` → `CheckpointLoaderSimple`（SAM3） + `PrimitiveBoundingBox` → `SAM3_Detect`

**具体案例**：街景图中你已经知道某辆红色车在 (100, 80) 位置、宽 220、高 280。用 `PrimitiveBoundingBox` 节点（xywh 格式）给出这个框，连到 SAM3_Detect 的 `bboxes` 输入，模型只切这一辆红色车，避免误检其他汽车。
- 注意：必须**不连接** `conditioning`，否则 SAM3 会用文本模式而不是 box 模式

---

### 功能 ③ 点提示分割（`03_point_prompt_image.json`）

**作用**：在物体上点一个或几个点，SAM3 自动识别"被点中的物体"并切出完整 mask。可选负点排除相似对象。

**节点组合**：`LoadImage` → `CheckpointLoaderSimple`（SAM3） + `PrimitiveString`（正点 JSON） + `PrimitiveString`（负点 JSON） → `SAM3_Detect`

**具体案例**：一张人像照，背景也有相似的衣服纹理。用 `PrimitiveString` 节点给 SAM3_Detect 的 `positive_coords` 输入 `[{"x": 256, "y": 256}]`（在人物脸上点一下），给 `negative_coords` 输入 `[{"x": 50, "y": 50}]`（在背景上点一下排除背景），SAM3 会精准切出完整的人像。
- 坐标是**像素值**（不是 normalized）
- JSON 必须是合法格式 `[{"x": int, "y": int}, ...]`

---

### 功能 ④ 多类别同时分割（`04_multi_prompt_text.json`）

**作用**：一条文本提示里写**多个类别**（用逗号分隔 + `:N` 限定每类最大检测数），SAM3 一次输出**所有类别**的 mask。

**节点组合**：`LoadImage` → `CheckpointLoaderSimple`（SAM3） → `CLIPTextEncode`（多类别提示） → `SAM3_Detect`

**具体案例**：同一张街景图，单个 CLIPTextEncode 节点写 `car:3, person:2, traffic light:4`，意思是"最多 3 辆汽车 + 2 个人 + 4 个红绿灯"。SAM3_Detect 一次跑完，输出每张 mask **独立**（设置 `individual_masks=True`）。
- 适合需要**批量**分析一张图的场景
- 配合 `SaveImage` 的 batch 模式可以一次性保存所有 mask

---

### 功能 ⑤ 视频对象跟踪（文本驱动，零标注） — `05_video_text_track.json`

**作用**：加载一段视频，**仅用文本提示**，SAM3 在每一帧自动找到对象并保持 ID 一致（基于 SAM 3.1 的 memory bank + multiplex tracking）。

**节点组合**：`LoadVideo` → `GetVideoComponents`（拆出帧） → `CheckpointLoaderSimple`（SAM3） + `CLIPTextEncode` → `SAM3_VideoTrack` → `SAM3_TrackToMask` / `SAM3_TrackPreview`

**具体案例**：一段 10 秒 24fps 的马路监控视频（example.mp4），用 `person` 提示，SAM3_VideoTrack 自动在每一帧检测所有人，并给每个人分配稳定 ID（即使穿出画面再回来也保持原 ID）。下游两个分支：
- `SAM3_TrackToMask`（empty indices = 全部）→ `MaskToImage` → `PreviewImage`：所有跟踪到的人的 mask 帧序列
- `SAM3_TrackPreview`：把每帧的 mask 渲染成**带对象编号 + 置信度**的彩色 mp4 预览

关键参数：
- `detection_threshold=0.5`：检测置信度阈值
- `max_objects=8`：最多跟踪 8 个对象（0 = 用内部上限 64）
- `detect_interval=1`：每 1 帧跑一次检测（>1 可省算力但会丢失新出现对象）

---

### 功能 ⑥ 视频初始掩码跟踪（`06_video_initial_mask_track.json`）

**作用**：在视频的**第一帧**用文本检测出 mask，然后 SAM3 跨帧传播这个 mask。比纯文本跟踪更稳，适合对象特征明显但跨帧容易丢失的场景。

**节点组合**：`LoadVideo` → `GetVideoComponents` → `ImageFromBatch`（取第 0 帧） → `CLIPTextEncode` → `SAM3_Detect`（首帧检测） → `SAM3_VideoTrack`（用 initial_mask 初始化） → `SAM3_TrackToMask` / `SAM3_TrackPreview`

**具体案例**：一段街景视频，提示 `yellow school bus`（黄色校车）。流程：
1. 用 `ImageFromBatch`（batch_index=0, length=1）从视频里抽出第 0 帧
2. 用 `SAM3_Detect` 在第 0 帧检测"黄色校车" → 得到 initial mask
3. 把 initial mask 和全部视频帧一起传给 `SAM3_VideoTrack`（initial_mask 入口）
4. SAM3 跨帧传播 mask，无需每帧重新检测
5. 输出 mask 序列和可视化 mp4

注意：`SAM3_TrackToMask` 输出的 MASK 需经 `MaskToImage` 转换才能被 `PreviewImage` 显示（MASK → IMAGE）。

---

### 功能 ⑦ 视频选择性对象提取（`07_track_select_objects.json`）

**作用**：跟踪完成后，从结果中**挑出指定 ID 的对象**作为最终 mask 输出，用于后续 inpaint、compositing 等局部处理。

**节点组合**：`LoadVideo` → `GetVideoComponents` → `CLIPTextEncode` → `SAM3_VideoTrack` → `SAM3_TrackToMask`（指定 indices） → `MaskToImage` → `PreviewImage`

**具体案例**：一段包含 6 个人的视频，先用 `person:6` 跟踪所有 6 个人，然后用 `SAM3_TrackToMask` 节点的 `object_indices` 参数填 `"0,2"`（只取 ID 0 和 ID 2 这两个人），输出这两人的 union mask。可以直接接到 inpaint 节点做"把这两个人从背景中抠出来"等后续处理。
- 留空 `object_indices` = 选全部
- 输入格式：逗号分隔的 ID 字符串

---

### 功能 ⑧ 跟踪结果可视化（`08_track_preview.json`）

**作用**：把视频跟踪结果**一键**渲染为带颜色 + 对象编号 + 置信度数字的 mp4 预览。无需自己拼装可视化代码。

**节点组合**：`LoadVideo` → `GetVideoComponents` → `CLIPTextEncode`（多对象） → `SAM3_VideoTrack` → `SAM3_TrackPreview`（输出节点）

**具体案例**：一个混合场景视频，提示 `person:3, dog:2`，SAM3 跟踪最多 3 个 person 和 2 个 dog。`SAM3_TrackPreview` 自动给每个对象分配不同颜色（在 mask 中心绘制编号 0/1/2/3/4，编号下方是 0-100 的置信度百分数），叠在原图上输出 mp4 预览（`opacity=0.5` 可调，`fps=24.0` 可调）。
- 输出节点，写到 ComfyUI 临时目录，可在 UI 中直接预览
- 不需要先经过 TrackToMask

---

## 4. 文件总览

```
sam3_workflows/
├── README.md                        ← 本文档
├── 01_text_prompt_image.json         ← 功能 ①
├── 02_box_prompt_image.json          ← 功能 ②
├── 03_point_prompt_image.json        ← 功能 ③
├── 04_multi_prompt_text.json         ← 功能 ④
├── 05_video_text_track.json          ← 功能 ⑤
├── 06_video_initial_mask_track.json  ← 功能 ⑥
├── 07_track_select_objects.json      ← 功能 ⑦
└── 08_track_preview.json             ← 功能 ⑧
```

## 5. 使用方法

1. 打开 ComfyUI 网页 UI
2. 左上角菜单 `Workflow` → `Open` → 选 `sam3_workflows/0X_xxx.json`
3. 修改 `LoadImage` / `LoadVideo` 节点的输入文件名为你的素材
4. 修改 `CLIPTextEncode` 节点的文本提示为你想检测的物体
5. 点击 `Queue Prompt` 运行

## 6. 常见参数说明

- **threshold**（检测阈值 0-1）：越高越严格
- **refine_iterations**（SAM decoder 精炼 0-5）：越高 mask 边界越精细（速度变慢）
- **individual_masks**（True/False）：是否把每个对象输出独立 mask
- **max_objects**（视频跟踪上限）：超过会被截断
- **detect_interval**（视频检测间隔 N）：每 N 帧跑一次检测

## 7. 重要提示

- **文本提示上限 32 tokens**，保持简短具体
- **多对象提示**：`物体1:2, 物体2:3`（冒号后是单个类别最多检测数）
- **Box 坐标**：左上角 + 宽高（像素）
- **Point 坐标**：JSON 格式 `[{"x": 100, "y": 200}]`（像素）
- **首次运行**会自动下载 SAM3 CLIP tokenizer（约 200 MB）
- **PreviewImage 只接受 IMAGE 类型**，从 SAM3_TrackToMask / SAM3_Detect 直接拿到的 MASK 必须先过 `MaskToImage`
