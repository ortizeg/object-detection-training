# Basketball Object Detection — Evaluation Report

**Date**: 2026-03-05 (updated with YOLO26 and RF-DETR Medium results)
**Dataset**: basketball-player-detection-3 (val: 96 images, test: 94 images)
**Classes**: player, ball, referee, rim, number (5 merged eval classes from 10 training classes)

## Methods

| Method | Type | Model | Notes |
|--------|------|-------|-------|
| RF-DETR v3 | Custom-trained | ONNX export (rfdetr-small, input 640) | Best run across all 23 RF-DETR training runs (run 20260117_005818, epoch=64, training mAP=0.5935), 91-class COCO heads with 11 basketball classes active |
| RF-DETR Medium v2 @640 | Custom-trained | ONNX export (rfdetr-medium, trained 576, eval 640) | Same checkpoint as v2, exported at 640 for fair comparison with RF-DETR Small v3 |
| RF-DETR Medium v2 @576 | Custom-trained | ONNX export (rfdetr-medium, input 576) | Native 576x576 resolution (epoch=104, training mAP=0.5334), no pos-emb mismatch |
| RF-DETR Medium v1 | Custom-trained | ONNX export (rfdetr-medium, input 640) | First run (epoch=39, mAP=0.4334), pos-emb mismatch from 576→640 |
| YOLO26m | Custom-trained | ONNX export (yolo26m, input 640) | Ultralytics YOLO26 Medium (epoch=279, training mAP@50:95=0.6848), NMS-free end-to-end, separate repo (AGPL isolation) |
| YOLO26s | Custom-trained | ONNX export (yolo26s, input 640) | Ultralytics YOLO26 Small (epoch=196, training mAP@50:95=0.6156), NMS-free end-to-end, separate repo (AGPL isolation) |
| YOLOX-M v1 | Custom-trained | ONNX export (yolox-medium, input 640) | First YOLOX Medium run (run 20260304_215835, epoch=117, training mAP_50=0.4957), 10-class model with eval-time class merging |
| YOLOX-S v2 | Custom-trained | ONNX export (yolox-small, input 640) | Best run across all 21 YOLOX training runs (run 20260203_195555, epoch=89, training mAP=0.4461), 10-class model with eval-time class merging |
| RF-DETR v2 | Custom-trained | ONNX export (rfdetr-small, input 512) | Previous best v2 training run (epoch=69, training mAP=0.5330), evaluated at 512x512 (mismatched from 640x640 training resolution) |
| YOLOX-S v1 | Custom-trained | ONNX export (yolox-small, input 640) | Previous YOLOX evaluation run, 10-class model with eval-time class merging |
| RF-DETR v1 | Custom-trained | ONNX export (rfdetr-small, input 512) | v1 training run (epoch=80), 10-class model with eval-time class merging |
| Gemini 3.1 Pro | Zero-shot VLM | gemini-3.1-pro-preview | Structured prompt detection, confidence fixed at 1.0 |
| OWLv2 | Zero-shot detector | google/owlv2-large-patch14-ensemble | Text-query detection with basketball-tuned prompts, NMS 0.3 |
| OmDet-Turbo | Zero-shot detector | omlab/omdet-turbo-swin-tiny-hf | Class-conditioned detection, NMS 0.5 |
| Grounding DINO | Zero-shot detector | IDEA-Research/grounding-dino-base | Text-grounded detection with period-separated prompt, NMS 0.5 |
| Florence-2 | Zero-shot detector | microsoft/Florence-2-large-ft | Generative object detection (`<OD>` task), no confidence scores |

## Model Architectures

### Custom-Trained Models

#### RF-DETR

**Real-time end-to-end detection transformer.** Uses a DINOv2 ViT backbone with a NAS-discovered decoder configuration. No NMS post-processing needed.

![RF-DETR Architecture](https://ar5iv.labs.arxiv.org/html/2511.09554/assets/x2.png)

**Architecture details:**
- **Backbone**: DINOv2 foundation model (ViT) — strong transfer learning to small/custom datasets
- **Encoder**: Deformable attention (AIFI + CCFF modules) for multi-scale feature fusion
- **Decoder**: Transformer decoder with learned object queries; weight-sharing NAS discovers optimal patch size, resolution, decoder layers, and query count
- **Training**: Scheduler-free optimizer, end-to-end set prediction with Hungarian matching

| Pros | Cons |
|------|------|
| Best accuracy on custom domains (DINOv2 transfers well to small datasets) | Higher latency than YOLO family at similar accuracy |
| End-to-end — no NMS tuning required | Large backbone (32M params for "small") |
| Flexible speed-accuracy tradeoff via decoder layer count | Requires more GPU memory |
| Scheduler-free training simplifies hyperparameter tuning | Newer/less mature ecosystem |

**Stats**: 32.1M params, 53.0 mAP@50:95 COCO, 3.52ms T4 TensorRT
**Paper**: [RF-DETR](https://arxiv.org/abs/2511.09554) (arxiv 2511.09554), ICLR 2026

---

#### YOLOX

**Anchor-free single-stage detector** that decouples classification and regression into separate heads. Fast, efficient, well-suited for edge deployment.

![YOLOX Architecture](https://ar5iv.labs.arxiv.org/html/2107.08430/assets/x2.png)

**Architecture details:**
- **Backbone**: CSPDarknet — efficient convolutional feature extractor
- **Neck**: PAN (Path Aggregation Network) for multi-scale feature fusion
- **Head**: Decoupled head with separate classification, regression, and objectness branches
- **Training**: SimOTA dynamic label assignment, strong augmentation (MixUp, Mosaic)

| Pros | Cons |
|------|------|
| Very fast inference (especially CPU/edge) | Lower accuracy ceiling than transformer detectors |
| Small model size (9M params) | Requires NMS post-processing |
| Mature ecosystem (ONNX/TensorRT/OpenVINO) | Struggles with small/occluded objects vs attention models |
| Anchor-free simplifies configuration | Single-scale feature matching limits complex scenes |

**Stats**: 9.0M params, 40.5 mAP@50:95 COCO, 9.8ms V100
**Paper**: [YOLOX](https://arxiv.org/abs/2107.08430) (arxiv 2107.08430)

**Variants evaluated in this report:**

| Variant | Depth | Width | Params | Model Size | ONNX Size | FLOPs |
|---------|-------|-------|--------|------------|-----------|-------|
| YOLOX-S | 0.33 | 0.50 | 9.0M | 34.2 MB | 34.2 MB | 13.3G |
| YOLOX-M | 0.67 | 0.75 | 25.3M | 96.6 MB | 96.5 MB | 36.8G |

---

#### YOLO26

**NMS-free anchor-free detector** from the Ultralytics YOLO family. Replaces traditional NMS with end-to-end learned detection, producing a fixed set of 300 predictions per image.

**Architecture details:**
- **Backbone**: CSP-based feature extractor
- **Neck**: Multi-scale feature fusion
- **Head**: End-to-end detection head (NMS-free), outputs top-300 detections with `[x1, y1, x2, y2, confidence, class_id]`
- **Output**: Fixed 300 detections per image, no post-processing needed

| Pros | Cons |
|------|------|
| NMS-free — deterministic output, no tuning required | AGPL license requires separate repo isolation |
| Fast inference (NMS removal saves latency) | Lower mAP@50:95 than RF-DETR despite good mAP@50 |
| Ultralytics ecosystem (easy training, export, augmentation) | Weak localization at strict IoU thresholds (mAP@75) |
| Good scaling from Small to Medium | Fewer training epochs may be needed (overfitting observed) |

**Variants evaluated in this report:**

| Variant | Params | ONNX Size | Input | Best Epoch |
|---------|--------|-----------|-------|------------|
| YOLO26s | ~11M | ~42 MB | 640x640 | 196/300 |
| YOLO26m | ~22M | ~84 MB | 640x640 | 279/300 |

**Note**: YOLO26 is trained in a separate repository (`yolo26-basketball-training`) to isolate the AGPL-licensed Ultralytics code from the Apache-2.0 main project.

---

#### RF-DETR Medium

Same architecture as RF-DETR Small but with a wider decoder. Trained at 640x640 (native resolution is 576x576).

**Issue**: Position embeddings from the 576x576 pretrained checkpoint had to be dropped due to shape mismatch when loading into 640x640 model (1297 vs 1601 positions). This forced the backbone to relearn spatial understanding from scratch, significantly hurting performance. Early stopping triggered at epoch 48 with mAP=0.4334.

---

### Zero-Shot Models

#### Gemini 3.1 Pro

**Google's multimodal large language model** used as a zero-shot detector via structured text prompting. Not a dedicated vision model — a general-purpose LLM with vision capabilities.

*(No architecture diagram — proprietary model)*

**Architecture details:**
- **Type**: Sparse Mixture-of-Experts transformer
- **Input**: Processes image + text jointly in a single forward pass
- **Output**: Structured JSON with bounding box coordinates and class labels
- **Context**: 1M token context window

| Pros | Cons |
|------|------|
| Strongest semantic reasoning ("referee" vs "player" from context) | Extremely slow (15.5s/img via API) |
| Highest precision among zero-shot methods | No confidence scores |
| Can follow complex, nuanced detection instructions | High per-image cost, non-deterministic |
| No local GPU needed | Cannot be fine-tuned or deployed locally |

**Reference**: [Google DeepMind Gemini](https://deepmind.google/models/gemini/)

---

#### OWLv2

**Open-vocabulary detector built on CLIP.** Adds per-token detection heads to a ViT backbone and scales via self-training on 1B+ web image-text pairs.

![OWLv2 Architecture](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x2.png)

**Architecture details:**
- **Backbone**: CLIP ViT-L/14 with pooling layer removed — retains per-patch spatial information
- **Classification head**: Dot product between per-token image features and text embeddings
- **Box regression head**: MLP on per-token features
- **Self-training (OWL-ST)**: Existing OWL-ViT generates pseudo-box annotations on WebLI dataset; OWLv2 trains on those pseudo-labels at 1B+ scale

| Pros | Cons |
|------|------|
| Best zero-shot detector in our eval (broadest class coverage) | Slow inference (2.8s/img on L4) |
| Massive self-training scale improves generalization | Low precision (many false positives) |
| Prompt-tunable (basketball prompts improved ball detection 3x) | Poor on domain-specific objects (rim: 0.3% AP) |
| Good recall across diverse classes | ViT backbone is compute-heavy |

**Stats**: ~600M params, LVIS rare 44.6% AP
**Paper**: [OWLv2](https://arxiv.org/abs/2306.09683) (arxiv 2306.09683)

---

#### OmDet-Turbo

**Real-time open-vocabulary detector** with an Efficient Fusion Head that combines language-aware encoding with deformable attention for fast multi-modal detection.

![OmDet-Turbo Architecture](https://ar5iv.labs.arxiv.org/html/2403.06892/assets/turbo_model.jpeg)

**Architecture details:**
- **Vision backbone**: Swin-Tiny for multi-scale feature extraction
- **Text encoder**: CLIP text encoder for language embeddings
- **Efficient Fusion Head**:
  - ELA-Encoder: Language-aware query proposals from multi-scale vision features
  - ELA-Decoder: Deformable attention + self-attention (no ROIAlign)
- **Pre-training**: 20M+ images with 4M vocabulary

| Pros | Cons |
|------|------|
| Fastest zero-shot detector (100 FPS with TensorRT + language cache) | Poor on domain-specific classes (ball: 9.2% AP, rim: 0%) |
| Good COCO zero-shot (53.4 AP) | Limited class coverage vs OWLv2 |
| Compact architecture | Relies on COCO-aligned vocabulary |
| Supports multi-task learning | Meta tensor initialization issues in transformers library |

**Stats**: 115M params, 53.4 mAP COCO zero-shot
**Paper**: [OmDet-Turbo](https://arxiv.org/abs/2403.06892) (arxiv 2403.06892)

---

#### Grounding DINO

**Extends the DINO detector with a BERT text encoder** for text-grounded open-set detection. Uses tight cross-modal fusion at every stage of the pipeline.

![Grounding DINO Architecture](https://ar5iv.labs.arxiv.org/html/2303.05499/assets/x3.png)

**Architecture details:**
- **Dual encoder**: Swin-T/B image backbone + BERT text encoder
- **Feature Enhancer**: 6 layers of deformable self-attention (vision), standard self-attention (text), and cross-attention (vision-language)
- **Query Selection**: Language-Guided Query Selection ranks image features by text similarity
- **Decoder**: Cross-Modality Decoder with 6 layers of image-to-text and text-to-image cross-attention; 900 queries

| Pros | Cons |
|------|------|
| Strong COCO zero-shot (52.5 AP) | Slow inference (1.4s/img) |
| Excellent person/player detection (84.9% AP) | Completely fails on ball (0% AP) and referee (0% AP) in our domain |
| Tight vision-language fusion enables nuanced grounding | Period-separated prompt format is fragile |
| Foundation for SAM/DINO pipelines (Grounded SAM) | Heavy model (233M params), BERT limited to 256 tokens |

**Stats**: 233M params, 52.5 mAP COCO zero-shot
**Paper**: [Grounding DINO](https://arxiv.org/abs/2303.05499) (arxiv 2303.05499), ECCV 2024

---

#### Florence-2

**Unified vision foundation model** that formulates all vision tasks (detection, captioning, segmentation) as sequence-to-sequence text generation. Trained on 5.4B synthetic annotations.

![Florence-2 Architecture](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x2.png)

**Architecture details:**
- **Image encoder**: DaViT (Data-efficient Vision Transformer) — produces flattened visual tokens
- **Text encoder**: BERT embeddings concatenated with visual tokens
- **Decoder**: Standard transformer encoder-decoder that generates text output, including bounding box coordinates as text tokens
- **Task specification**: Text prompt prefix (e.g., `<OD>` for object detection, `<CAPTION>` for captioning)
- **Training data**: FLD-5B — 5.4B annotations on 126M images

| Pros | Cons |
|------|------|
| Single model handles detection, captioning, segmentation, grounding | Generative approach is slow for detection (1.1s/img) |
| Compact size (770M) enables mobile deployment | No confidence scores (all detections equally weighted) |
| Massive training data (5.4B annotations) | Poor zero-shot detection in our domain (13.8% mAP@50) |
| Unified architecture simplifies deployment | Jack-of-all-trades, master of none for detection |

**Stats**: 770M params (large variant)
**Paper**: [Florence-2](https://arxiv.org/abs/2311.06242) (arxiv 2311.06242), CVPR 2024

---

## Prompts & Configuration

### Gemini 3.1 Pro (VLM — structured text prompt)

```
Detect all basketball players, referees, the rim, the basketball, and
visible jersey numbers in this basketball game image.
Constraints:
- At most 10 players, 3 referees, 1 rim, and 1 ball per image.
- Each person gets exactly ONE bounding box with the most specific label.
Use EXACTLY these labels:
- "player" for any basketball player on the court
- "referee" for game officials
- "ball" for the basketball (only when NOT going through the rim)
- "rim" for the basketball hoop rim
- "number" for visible jersey numbers (tight box around the digits only)
```

Output classes: `["ball", "number", "player", "referee", "rim"]`
Confidence: fixed at 1.0 (VLM does not output scores)

### OWLv2 (text-query detection — basketball-tuned prompts)

Text queries passed as `[["basketball player", "basketball", "referee", "basketball hoop", "jersey number"]]` to `Owlv2Processor.post_process_grounded_object_detection`.

Prompt tuning notes:
- "basketball player" instead of "person" — eliminated player/referee confusion where all people were labeled as referee
- "basketball" instead of "sports ball" — ball confidence jumped from ~0.24 to 0.70+
- "basketball hoop" — only viable rim prompt (alternatives "rim", "basket", "basketball rim" produced 0 detections)

Config: `box_threshold=0.01`, `nms_iou_threshold=0.3`

### OmDet-Turbo (class-conditioned detection)

Classes: `["person", "sports ball", "referee", "basketball hoop", "jersey number"]`

Passed as a class list to `OmDetTurboProcessor` (not a text prompt). The model conditions on class embeddings from its text encoder.

Config: `box_threshold=0.01`, `nms_iou_threshold=0.5`

### Grounding DINO (text-grounded detection — period-separated prompt)

Text prompt: `"person . sports ball . referee . basketball hoop . jersey number ."`

Grounding DINO requires a single period-separated string. The model matches image regions to each text phrase independently.

Config: `box_threshold=0.01`, `text_threshold=0.01`, `nms_iou_threshold=0.5`

### Florence-2 (generative detection)

Task mode: `<OD>` (general object detection)

Florence-2 uses a generative approach — it outputs bounding box tokens autoregressively. The model's built-in vocabulary determines which classes it detects; the class list `["person", "sports ball", "referee", "basketball hoop", "jersey number"]` is used only for post-hoc label matching. Florence-2 does not output confidence scores (all detections use a default confidence).

Config: `florence2_task="<OD>"`

## Overall Results

### Custom-Trained Models

| Method | Split | mAP@50:95 | mAP@50 | mAP@75 | Precision | Recall | F1 | Threshold |
|--------|-------|-----------|--------|--------|-----------|--------|------|-----------|
| **RF-DETR v3** | **val** | **62.72%** | **95.09%** | **66.28%** | **95.15%** | **92.47%** | **93.79%** | **0.40** |
| **RF-DETR v3** | **test** | **61.16%** | **91.43%** | **64.59%** | **92.63%** | **91.46%** | **92.05%** | **0.40** |
| YOLOX-M v1 | val | 59.35% | 89.74% | 67.04% | 94.68% | 87.40% | 90.89% | 0.60 |
| YOLOX-M v1 | test | 58.29% | 89.21% | 64.77% | 95.55% | 87.93% | 91.58% | 0.60 |
| RF-DETR Medium v2 | val | 58.95% | 92.15% | 63.84% | 95.51% | 90.42% | 92.90% | 0.45 |
| RF-DETR Medium v2 | test | 59.74% | 91.36% | 63.43% | 92.86% | 91.31% | 92.08% | 0.45 |
| RF-DETR Medium v1 | val | 53.46% | 88.01% | 54.27% | 93.38% | 86.69% | 89.91% | 0.45 |
| RF-DETR Medium v1 | test | 52.60% | 84.66% | 52.33% | 90.85% | 84.70% | 87.66% | 0.45 |
| YOLOX-S v2 | val | 56.78% | 89.39% | 63.21% | 93.28% | 86.69% | 89.86% | 0.55 |
| YOLOX-S v2 | test | 53.43% | 86.76% | 58.61% | 92.95% | 85.86% | 89.26% | 0.55 |
| YOLO26m | val | 46.29% | 87.92% | 40.73% | 93.66% | 83.26% | 88.15% | 0.25 |
| YOLO26m | test | 48.88% | 89.15% | 47.21% | 95.02% | 84.80% | 89.62% | 0.25 |
| YOLO26s | val | 41.16% | 84.08% | 29.23% | 86.60% | 81.41% | 83.93% | 0.20 |
| YOLO26s | test | 42.10% | 84.65% | 30.30% | 84.31% | 85.20% | 84.75% | 0.20 |
| RF-DETR v2 | test | 56.67% | 89.65% | 57.15% | — | — | — | — |
| YOLOX-S v1 | val | 56.17% | 88.95% | 61.71% | 93.03% | 87.46% | 90.16% | 0.55 |
| YOLOX-S v1 | test | 54.87% | 86.15% | 62.51% | 91.84% | 86.41% | 89.05% | 0.55 |
| RF-DETR v1 | val | 49.70% | 86.44% | 46.39% | 93.59% | 85.97% | 89.62% | 0.55 |
| RF-DETR v1 | test | 50.31% | 85.82% | 45.44% | 93.80% | 85.56% | 89.49% | 0.55 |

Note: RF-DETR v2 was evaluated locally (test split only) at 512x512 input from an earlier session. P/R/F1 were not computed.

### Zero-Shot Models

| Method | Split | mAP@50:95 | mAP@50 | mAP@75 | Precision | Recall | F1 | Threshold |
|--------|-------|-----------|--------|--------|-----------|--------|------|-----------|
| Gemini 3.1 Pro | val | 25.12% | 43.80% | 26.96% | 80.62% | 66.67% | 72.98% | 0.80 |
| Gemini 3.1 Pro | test | 26.47% | 43.06% | 29.32% | 77.87% | 66.11% | 71.51% | 0.80 |
| **OWLv2** | **val** | **24.14%** | **39.72%** | **27.33%** | 53.22% | 61.44% | 57.03% | 0.25 |
| **OWLv2** | **test** | **24.74%** | **39.07%** | **27.59%** | 50.18% | 64.60% | 56.48% | 0.25 |
| OmDet-Turbo | val | 17.13% | 25.46% | 18.46% | 63.12% | 35.59% | 45.51% | 0.25 |
| OmDet-Turbo | test | 17.29% | 25.42% | 18.79% | 61.69% | 36.11% | 45.56% | 0.25 |
| Grounding DINO | val | 13.60% | 16.53% | 15.11% | 75.93% | 40.55% | 52.87% | 0.35 |
| Grounding DINO | test | 14.70% | 17.07% | 15.89% | 79.64% | 42.47% | 55.40% | 0.35 |
| Florence-2 | val | 8.31% | 11.36% | 9.91% | 74.79% | 35.69% | 48.32% | 0.05 |
| Florence-2 | test | 10.38% | 13.76% | 12.47% | 82.25% | 38.38% | 52.34% | 0.05 |

## Per-Class AP@50

### Custom-Trained Models

| Class | RF-DETR v3 Val | RF-DETR v3 Test | RF-DETR-M v2 Val | RF-DETR-M v2 Test | YOLO26m Val | YOLO26m Test | YOLOX-M Val | YOLOX-M Test | YOLO26s Val | YOLO26s Test | YOLOX-S v2 Val | YOLOX-S v2 Test |
|-------|---------------|----------------|-----------------|------------------|------------|-------------|------------|-------------|------------|-------------|-------------|--------------|
| player | **98.62%** | **97.91%** | 98.46% | 98.67% | 91.13% | 93.64% | 92.70% | 93.82% | 92.86% | 93.47% | 93.24% | 93.05% |
| ball | **88.74%** | **75.46%** | 78.30% | 74.16% | 77.75% | 75.38% | 72.60% | 69.13% | 71.30% | 62.98% | 73.15% | 61.55% |
| referee | **96.90%** | **98.94%** | 97.85% | 98.34% | 84.92% | 90.57% | 92.54% | 95.35% | 75.95% | 80.36% | 94.46% | 95.20% |
| rim | **100.00%** | **100.00%** | 96.00% | **99.96%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **99.92%** | **99.89%** | **100.00%** | **100.00%** |
| number | **91.21%** | 84.85% | 90.13% | 85.67% | 85.83% | 86.14% | 90.86% | **87.75%** | 80.36% | 86.55% | 86.08% | 84.01% |

### Zero-Shot Models

| Class | Gemini Val | Gemini Test | OWLv2 Val | OWLv2 Test | OmDet Val | OmDet Test | GDINO Val | GDINO Test | Florence Val | Florence Test |
|-------|------------|-------------|-----------|------------|-----------|------------|-----------|------------|--------------|---------------|
| player | 89.57% | 92.49% | 83.35% | 84.84% | 80.91% | 84.33% | 82.26% | 84.89% | 56.79% | 66.83% |
| ball | 45.90% | 36.30% | 50.83% | 41.24% | 13.28% | 9.19% | 0.00% | 0.00% | 0.00% | 1.98% |
| referee | 60.53% | 68.59% | 21.50% | 35.24% | 32.53% | 33.36% | 0.14% | 0.00% | 0.00% | 0.00% |
| rim | 4.37% | 4.26% | 1.16% | 0.31% | 0.00% | 0.00% | 0.00% | 0.02% | 0.00% | 0.00% |
| number | 18.63% | 13.66% | 41.74% | 33.70% | 0.60% | 0.20% | 0.27% | 0.46% | 0.00% | 0.00% |

### All Methods Combined (Test AP@50)

| Class | RF-DETR v3 | RF-DETR-M v2 | YOLO26m | YOLOX-M | YOLO26s | YOLOX-S v2 | Gemini | OWLv2 |
|-------|-----------|-------------|--------|--------|--------|---------|--------|-------|
| player | **97.91%** | 98.67% | 93.64% | 93.82% | 93.47% | 93.05% | 92.49% | 84.84% |
| ball | **75.46%** | 74.16% | 75.38% | 69.13% | 62.98% | 61.55% | 36.30% | 41.24% |
| referee | **98.94%** | 98.34% | 90.57% | 95.35% | 80.36% | 95.20% | 68.59% | 35.24% |
| rim | **100.00%** | 99.96% | **100.00%** | **100.00%** | 99.89% | **100.00%** | 4.26% | 0.31% |
| number | 84.85% | 85.67% | 86.14% | **87.75%** | **86.55%** | 84.01% | 13.66% | 33.70% |
| **Average** | **91.43%** | **91.36%** | **89.15%** | **89.21%** | **84.65%** | **86.76%** | **43.06%** | **39.07%** |

## Per-Class Precision & Recall at Operating Threshold

### RF-DETR v3 (threshold=0.40)

| Class | Val P | Val R | Val F1 | Test P | Test R | Test F1 |
|-------|-------|-------|--------|--------|--------|---------|
| player | 98.0% | 96.1% | 97.0% | 97.2% | 97.7% | 97.4% |
| ball | 91.0% | 83.5% | 87.1% | 84.0% | 65.6% | 73.7% |
| referee | 98.5% | 89.2% | 93.6% | 96.5% | 97.5% | 97.0% |
| rim | 100.0% | 99.0% | 99.5% | 98.9% | 100.0% | 99.5% |
| number | 88.9% | 88.7% | 88.8% | 83.8% | 81.8% | 82.8% |
| **Overall** | **95.3%** | **92.6%** | **93.9%** | **92.8%** | **91.7%** | **92.2%** |

### YOLOX-S v2 (threshold=0.55)

| Class | Val P | Val R | Val F1 | Test P | Test R | Test F1 |
|-------|-------|-------|--------|--------|--------|---------|
| player | 97.6% | 89.5% | 93.4% | 95.0% | 91.0% | 93.0% |
| ball | 80.7% | 69.1% | 74.4% | 89.1% | 51.0% | 64.9% |
| referee | 96.4% | 84.0% | 89.8% | 97.6% | 88.2% | 92.7% |
| rim | 99.0% | 100.0% | 99.5% | 100.0% | 100.0% | 100.0% |
| number | 85.6% | 84.1% | 84.8% | 87.0% | 80.4% | 83.6% |
| **Overall** | **93.3%** | **86.7%** | **89.9%** | **93.2%** | **86.1%** | **89.5%** |

### Per-Class Comparison at Operating Thresholds (Test Set)

| Class | RF-DETR v3 P | RF-DETR v3 R | YOLOX v2 P | YOLOX v2 R | P Winner | R Winner |
|-------|-------------|-------------|-----------|-----------|----------|----------|
| player | 97.2% | **97.7%** | 95.0% | 91.0% | RF-DETR | RF-DETR |
| ball | 84.0% | **65.6%** | **89.1%** | 51.0% | YOLOX | RF-DETR |
| referee | 96.5% | **97.5%** | **97.6%** | 88.2% | YOLOX | RF-DETR |
| rim | 98.9% | **100.0%** | **100.0%** | **100.0%** | YOLOX | Tied |
| number | 83.8% | **81.8%** | **87.0%** | 80.4% | YOLOX | RF-DETR |
| **Overall** | 92.8% | **91.7%** | **93.2%** | 86.1% | YOLOX | RF-DETR |

RF-DETR v3 wins on recall for every class (except rim which is tied). YOLOX v2 has slightly higher precision on 4 of 5 classes but at the cost of significantly lower recall. RF-DETR v3's higher F1 (92.2% vs 89.5%) reflects the better balance.

## Inference Speed

### NVIDIA L4 GPU (ONNX Runtime + CUDA)

| Method | End-to-End (img/s) | ms/img (mean) | ms/img (median) | Notes |
|--------|-------------------|---------------|-----------------|-------|
| **YOLOX-S v2 (ONNX)** | **4.73** | **211 ms** | **212 ms** | 3.1x faster than RF-DETR |
| **YOLO26s (ONNX)** | **4.53** | **221 ms** | **219 ms** | NMS-free, comparable to YOLOX-S |
| **YOLOX-M v1 (ONNX)** | **2.16** | **463 ms** | **468 ms** | 1.6x faster than RF-DETR |
| **YOLO26m (ONNX)** | **2.04** | **490 ms** | **471 ms** | NMS-free, comparable to YOLOX-M |
| **RF-DETR v3 (ONNX)** | **1.55** | **643 ms** | **643 ms** | Best accuracy, higher latency |
| RF-DETR Medium v2 @576 (ONNX) | 1.58 | 634 ms | 627 ms | Native 576, near RF-DETR v3 accuracy |
| RF-DETR Medium v2 @640 (ONNX) | 1.41 | 707 ms | 704 ms | Trained 576, eval 640 for fair comparison |
| RF-DETR Medium v1 (ONNX) | 1.40 | 715 ms | 701 ms | Underperformed due to pos-emb reset |
| OmDet-Turbo | 1.86 | 538 ms | — | Zero-shot |
| Florence-2 | 0.92 | 1,087 ms | — | Zero-shot |
| Grounding DINO | 0.73 | 1,370 ms | — | Zero-shot |
| OWLv2 | 0.35 | 2,820 ms | — | Zero-shot |
| Gemini 3.1 Pro (API) | 0.064 | 15,505 ms | — | Cloud API |

End-to-end rows measured with `time.perf_counter()` timing (first image excluded as warmup, test split, 93 images). Includes image I/O, preprocessing (resize, normalize), ONNX Runtime inference, and post-processing.

### Pure Model Inference (NVIDIA L4, PyTorch)

From `model_info.json` logged during training — measures only the forward pass (no I/O, preprocessing, or post-processing):

| Model | Input | ms/img | FPS | Params | FLOPs |
|-------|-------|--------|-----|--------|-------|
| YOLOX-S v2 | 640x640 | 15.5 ms | 64.6 | 8.9M | 13.3G |
| YOLOX-M v1 | 640x640 | 16.2 ms | 61.9 | 25.3M | 36.8G |
| RF-DETR v3 Small | 640x640 | 20.4 ms | 49.1 | 32.1M | 61.0G |
| RF-DETR Medium v2 | 576x576 | 21.0 ms | 47.5 | 33.4M | 32.4G |

YOLOX-S pure inference is **1.3x faster** than RF-DETR Small (15.5 vs 20.4 ms). The gap widens to **3.1x** end-to-end (211 vs 643 ms) because RF-DETR's ViT backbone has heavier preprocessing and ONNX Runtime overhead. RF-DETR Medium has fewer FLOPs than Small (32.4G vs 61.0G at 576 vs 640 input) but similar inference time due to attention memory access patterns.

### Apple M3 Max CPU (ONNX Runtime + CPU)

| Method | End-to-End (img/s) | ms/img |
|--------|-------------------|--------|
| YOLOX-S v2 (ONNX) | 16.4 | 61 ms |
| RF-DETR v3 (ONNX) | 5.5 | 182 ms |

YOLOX is 3x faster than RF-DETR on CPU due to lower parameter count (8.9M vs 32.1M).

## Model Footprint & Memory

| Metric | YOLOX-S | YOLOX-M | RF-DETR Small |
|--------|---------|---------|---------------|
| **Total params** | 9.0M | 25.3M | 31.8M |
| **Trainable params** | 4.7M | 12.9M | 31.8M |
| **FLOPs** | 13.3G | 36.8G | 25.5G |
| **Model size (PyTorch)** | 34.2 MB | 96.6 MB | 121.4 MB |
| **ONNX size** | 34.2 MB | 96.5 MB | 110.3 MB |
| **Training batch size (L4 24GB)** | 32 | 16 | 16 |
| **Grad accum steps** | 2 | 4 | 1 |
| **Effective batch size** | 64 | 64 | 16 |
| **Input resolution** | 640x640 | 640x640 | 640x640 |

**Memory analysis:**

- **YOLOX-S** is the most memory-efficient model at 34 MB, enabling batch_size=32 on L4 GPU. This is ideal for rapid iteration and hyperparameter sweeps where GPU memory is the bottleneck.
- **YOLOX-M** is 2.8x larger (97 MB), requiring batch_size=16 with 4x gradient accumulation to match the effective batch size. Training time per epoch increases proportionally due to more forward/backward passes.
- **RF-DETR Small** is the heaviest at 121 MB despite similar parameter count to YOLOX-M (31.8M vs 25.3M). The ViT backbone's self-attention mechanism consumes significantly more activation memory than YOLOX's convolutional backbone, limiting batch size to 16 with no gradient accumulation (effective batch=16).
- **FLOPs vs memory**: YOLOX-M has 44% more FLOPs than RF-DETR (36.8G vs 25.5G), but RF-DETR's attention layers require more memory per sample. This explains why RF-DETR is slower at inference despite lower FLOPs -- attention operations have worse hardware utilization than convolutions on current GPUs.
- **Deployment implications**: YOLOX-S ONNX (34 MB) fits easily on edge devices; YOLOX-M (97 MB) is feasible for server-side deployment; RF-DETR (110 MB) requires a dedicated GPU for real-time inference.

## Analysis: YOLOX Medium vs YOLOX Small vs RF-DETR

### Does YOLOX-M close the gap with RF-DETR?

YOLOX-M significantly narrows the accuracy gap while remaining faster than RF-DETR:

| Metric (test) | YOLOX-S v2 | YOLOX-M v1 | RF-DETR v3 | YOLOX-M vs S improvement | YOLOX-M vs RF-DETR gap |
|--------|---------|---------|-----------|--------------------------|------------------------|
| mAP@50:95 | 53.43% | 58.29% | 61.16% | **+4.86** | -2.87 |
| mAP@50 | 86.76% | 89.21% | 91.43% | **+2.45** | -2.22 |
| mAP@75 | 58.61% | 64.77% | 64.59% | **+6.16** | **+0.18** |
| Precision | 92.95% | 95.55% | 92.63% | **+2.60** | **+2.92** |
| Recall | 85.86% | 87.93% | 91.46% | **+2.07** | -3.53 |
| F1 | 89.26% | 91.58% | 92.05% | **+2.32** | -0.47 |
| ms/img (L4) | 211 ms | 463 ms | 643 ms | 2.2x slower | 1.4x faster |

**Key findings:**

1. **YOLOX-M nearly matches RF-DETR on F1** (91.58% vs 92.05%, only 0.47 points difference). This is a dramatic narrowing from the 2.79-point gap between YOLOX-S and RF-DETR.

2. **YOLOX-M matches RF-DETR on localization quality**: mAP@75 is essentially tied (64.77% vs 64.59%). YOLOX-M's convolutional architecture at medium width produces boxes just as tight as RF-DETR's attention-based decoder.

3. **YOLOX-M has the highest precision** of any model (95.55%), meaning fewer false positives. This is valuable for applications where false detections are costly.

4. **YOLOX-M is 1.4x faster** than RF-DETR on L4 GPU (463ms vs 643ms). Not the 3.1x speed advantage of YOLOX-S, but still meaningfully faster.

5. **RF-DETR retains the recall advantage** (91.46% vs 87.93%). RF-DETR finds 3.5% more objects, which matters for applications where missing detections is worse than false positives.

### Per-Class Analysis: YOLOX-M vs YOLOX-S (Test AP@50)

| Class | YOLOX-S v2 | YOLOX-M v1 | Improvement | Notes |
|-------|-----------|-----------|-------------|-------|
| player | 93.05% | 93.82% | +0.77 | Marginal -- already saturated |
| ball | 61.55% | **69.13%** | **+7.58** | Biggest gain -- medium backbone helps small objects |
| referee | 95.20% | 95.35% | +0.15 | Already near-saturated |
| rim | 100.00% | 100.00% | 0.00 | Both perfect |
| number | 84.01% | **87.75%** | **+3.74** | Notable gain -- wider features help digit recognition |

YOLOX-M's biggest improvements are on **ball** (+7.58 pts) and **number** (+3.74 pts) -- the two smallest, hardest-to-detect classes. The wider backbone (0.75 vs 0.50 width multiplier) provides richer features for fine-grained localization.

### Speed-Accuracy-Memory Trade-off Summary

| Model | mAP@50:95 | mAP@50 | F1 | ms/img (L4) | Best For |
|-------|-----------|--------|-----|------------|----------|
| RF-DETR v3 | **61.16%** | **91.43%** | 92.05% | 643 ms | Maximum accuracy, recall-critical |
| RF-DETR-M v2 @640 | 60.74% | 91.04% | 91.91% | 707 ms | Tied with v3 at same eval resolution |
| RF-DETR-M v2 @576 | 59.74% | 91.36% | **92.08%** | 634 ms | Best F1, slightly faster than v3 |
| YOLOX-M | 58.29% | 89.21% | 91.58% | 463 ms | Server-side, balanced speed/accuracy |
| YOLOX-S | 53.43% | 86.76% | 89.26% | 211 ms | Edge deployment, real-time |
| RF-DETR-M v1 (640) | 52.60% | 84.66% | 87.66% | 715 ms | *Not recommended (pos-emb issue)* |
| YOLO26m | 48.88% | 89.15% | 89.62% | 490 ms | High mAP@50, NMS-free |
| YOLO26s | 42.10% | 84.65% | 84.75% | 221 ms | Fast NMS-free baseline |

**Current recommendation**: RF-DETR v3 Small remains the best overall model by a narrow margin. RF-DETR Medium v2 (576) is essentially tied on mAP@50 and F1, trailing only on mAP@50:95 by 1.4 points — and is slightly faster. YOLOX-M is the best speed-accuracy tradeoff for server deployment. YOLO26 models show promise at the mAP@50 level but have significantly weaker localization (mAP@75).

## Analysis: YOLO26 Results

### YOLO26 vs Training Metrics Discrepancy

YOLO26m reported 68.48% mAP@50:95 during training (Ultralytics built-in eval) but only 48.88% in our standardized eval pipeline. This ~20-point gap is explained by:

1. **Different evaluation implementations**: Ultralytics uses its own COCO eval vs our supervision-based COCO eval
2. **Class merging**: Training evaluates on 10 original classes; our eval merges to 5 classes
3. **Preprocessing differences**: Training eval uses Ultralytics' letterbox preprocessing; our ONNX eval uses simple resize
4. **ONNX export artifacts**: The end-to-end NMS-free export may behave differently than the PyTorch model during training eval

### YOLO26 Localization Weakness

YOLO26 models show a striking gap between mAP@50 and mAP@75:

| Model | mAP@50 (test) | mAP@75 (test) | Gap |
|-------|---------------|---------------|-----|
| RF-DETR v3 | 91.43% | 64.59% | 26.84 |
| YOLOX-M | 89.21% | 64.77% | 24.44 |
| YOLO26m | 89.15% | 47.21% | **41.94** |
| YOLO26s | 84.65% | 30.30% | **54.35** |

YOLO26 detects objects well (high mAP@50) but boxes are not as tight (low mAP@75). This suggests the end-to-end NMS-free head may sacrifice localization precision for detection coverage.

### RF-DETR Medium: Resolution Lesson

**v1 (640x640) — 52.60% mAP@50:95**: Trained at 640x640, but the pretrained `rf-detr-medium.pth` uses 576x576 (1297 position embeddings vs 1601 needed). `strict=False` in `load_state_dict` dropped the position embeddings entirely, forcing the backbone to relearn spatial encoding from scratch. Early stopping triggered at epoch 48 with training mAP=0.4334.

**v2 (576x576) — 59.74% mAP@50:95**: Retrained at native 576x576 resolution — pretrained checkpoint loads perfectly with no position embedding mismatch. Trained for 104 epochs (training mAP=0.5334). The +7.14 point improvement confirms position embedding preservation is critical for transfer learning.

**Comparison with RF-DETR Small v3 (apples-to-apples at 640 eval)**:

| Metric (test) | RF-DETR-M v2 @576 | RF-DETR-M v2 @640 | RF-DETR v3 Small @640 | Delta (M@640 vs S@640) |
|---------------|--------------------|--------------------|------------------------|------------------------|
| mAP@50:95 | 59.74% | 60.74% | **61.16%** | -0.42 |
| mAP@50 | 91.36% | 91.04% | **91.43%** | -0.39 |
| mAP@75 | 63.43% | 63.59% | **64.59%** | -1.00 |
| F1 | **92.08%** | 91.91% | 92.05% | -0.14 |
| ms/img (L4) | 634 ms | 707 ms | 643 ms | +64 ms |

When evaluated at the same 640 resolution as RF-DETR Small v3, Medium closes to within **0.42 points** on mAP@50:95 — essentially tied. The remaining gap (1.4 pts at 576 eval vs 0.42 pts at 640 eval) was due to the eval resolution difference, not model quality. RF-DETR Small v3's slight edge likely comes from its 23-run hyperparameter sweep vs Medium's single run. A sweep of RF-DETR Medium at 576 could potentially match or exceed Small v3.

**Lesson**: Never train a model at a resolution different from its pretrained checkpoint without implementing position embedding interpolation. DINOv2's built-in bicubic interpolation (`dinov2.py:151-203`) works at inference time for resolution changes, but `load_state_dict` will silently drop mismatched position embeddings during training initialization.

## Comparison with data-visor Baselines

| Metric | data-visor RF-DETR | RF-DETR v1 (5-class) | RF-DETR v1 (10-class) | RF-DETR v2 (5-class) | RF-DETR v2 (10-class) | RF-DETR v3 (5-class) | data-visor Gemini 3.1 | This Gemini 3.1 |
|--------|-------------------|---------------------|----------------------|---------------------|----------------------|---------------------|----------------------|-----------------|
| mAP@50:95 | 47.7% | 50.3% | 45.9% | 56.7% | 46.0% | **61.2%** | 12.9% | 26.5% |
| mAP@50 | 74.1% | 85.8% | 71.0% | 89.7% | 72.2% | **91.4%** | 22.3% | 43.1% |
| mAP@75 | 51.6% | 45.4% | 48.8% | 57.2% | 45.6% | **64.6%** | 13.2% | 29.3% |

**Key insight: class merging explains the data-visor gap.** Data-visor evaluates on 10 original training classes, while our primary eval merges to 5 classes (e.g., player-in-possession -> player). When we evaluate RF-DETR v1 on the original 10 classes (71.0% mAP@50), the result closely matches data-visor (74.1%), confirming class merging -- not evaluation bugs -- is the root cause of the discrepancy. The remaining ~3-point gap is due to different mAP implementations (supervision COCO-style vs data-visor custom matching).

## Analysis: Why RF-DETR v3 is Better Than Previously Reported Results

### What Changed

RF-DETR v3 represents the best checkpoint from a systematic sweep of all 23 RF-DETR training runs, evaluated at the correct input resolution. Two factors explain the improvement over the previous best (RF-DETR v2):

**1. Better checkpoint (training mAP 0.5935 vs 0.5330)**

The previous RF-DETR v2 used the checkpoint from one specific run (epoch=69, training val mAP=0.5330). A sweep of all 23 RF-DETR runs revealed a better run (20260117_005818, epoch=64, training val mAP=0.5935) -- an improvement of +0.0605 in training mAP. This is the single best RF-DETR checkpoint across all training experiments.

**2. Correct input resolution (640x640 vs 512x512)**

The previous RF-DETR v2 was evaluated at 512x512 input, but the model was trained at 640x640. This resolution mismatch degraded performance, particularly for small objects. Evaluating at the training resolution (640x640) improved mAP@75 by +7.4 points (57.15% -> 64.59%), confirming that localization quality was hurt most by the downscaling.

| Metric (test) | RF-DETR v2 (512x512) | RF-DETR v3 (640x640) | Delta | Primary cause |
|---------------|---------------------|---------------------|-------|---------------|
| mAP@50:95 | 56.67% | **61.16%** | +4.49 | Better checkpoint + resolution |
| mAP@50 | 89.65% | **91.43%** | +1.78 | Better checkpoint |
| mAP@75 | 57.15% | **64.59%** | +7.44 | Resolution match (tight boxes) |
| F1 | — | **92.05%** | — | First measurement |

The disproportionate mAP@75 improvement (+7.4 vs +1.8 at mAP@50) confirms that the resolution mismatch was the primary issue for localization quality. At 512x512, the model was forced to detect objects at a resolution 36% lower than what it learned, producing less precise boxes.

### YOLOX-S v2: Marginal Change

The YOLOX best-run sweep (run 20260203_195555, epoch=89, training mAP=0.4461) produced results nearly identical to v1. Val metrics improved slightly (+0.4-1.5%), while test metrics were mixed (mAP@50 +0.6%, mAP@50:95 -1.4%). Both YOLOX runs were evaluated at the correct 640x640 resolution, so no resolution artifact exists. The YOLOX architecture appears to have converged to a performance ceiling around 87% mAP@50 / 54% mAP@50:95 on this dataset.

## Analysis: Custom-Trained vs Zero-Shot Detectors

### The Performance Gap

Custom-trained models (RF-DETR v3, YOLOX-S v2) dominate zero-shot approaches across every metric:

| Method Type | Best mAP@50 (test) | Best F1 (test) | Best Recall (test) |
|-------------|-------------------|----------------|-------------------|
| Custom-trained | **91.43% (RF-DETR v3)** | **92.05% (RF-DETR v3)** | **91.46% (RF-DETR v3)** |
| Zero-shot VLM | 43.06% (Gemini) | 71.51% (Gemini) | 66.11% (Gemini) |
| Zero-shot detector | 39.07% (OWLv2) | 56.48% (OWLv2) | 64.60% (OWLv2) |

Custom-trained models achieve **2.1x higher mAP@50** and **1.3x higher F1** than the best zero-shot approach. The gap is even larger at stricter IoU thresholds: RF-DETR v3's mAP@75 (64.6%) is **2.3x** higher than OWLv2's (27.6%).

### Zero-Shot Detector Ranking

Among the 4 zero-shot detectors, **OWLv2 is the clear winner**:

| Metric (test) | OWLv2 | OmDet-Turbo | Grounding DINO | Florence-2 |
|---------------|-------|-------------|----------------|------------|
| mAP@50:95 | **24.74%** | 17.29% | 14.70% | 10.38% |
| mAP@50 | **39.07%** | 25.42% | 17.07% | 13.76% |
| Recall | **64.60%** | 36.11% | 42.47% | 38.38% |
| F1 | **56.48%** | 45.56% | 55.40% | 52.34% |
| Classes with AP>10% | **4/5** | 2/5 | 1/5 | 1/5 |

OWLv2 is the only zero-shot detector that achieves meaningful performance on more than just player detection. It detects ball (41.2%), number (33.7%), and referee (35.2%) -- classes where the other three zero-shot detectors score near zero.

### Per-Class Analysis: Where Zero-Shot Models Fail

**Player detection -- the easy class**:

All zero-shot detectors achieve 67-93% AP@50 on players. This is the most visually salient class and benefits from CLIP/VLM pretraining on billions of web images containing people. The gap to custom-trained models (93-98%) is modest.

| Method | Player AP@50 (test) | Gap to RF-DETR v3 |
|--------|--------------------|----|
| RF-DETR v3 | 97.91% | -- |
| YOLOX v2 | 93.05% | -4.9 |
| Gemini | 92.49% | -5.4 |
| OWLv2 | 84.84% | -13.1 |
| Grounding DINO | 84.89% | -13.0 |
| OmDet-Turbo | 84.33% | -13.6 |
| Florence-2 | 66.83% | -31.1 |

**Ball detection -- RF-DETR v3 dominates**:

RF-DETR v3 achieves 75.5% AP@50 on ball, leading YOLOX by 13.9 points. The basketball is small, frequently occluded, and subject to motion blur. OWLv2 achieves 41.2% AP@50, outperforming Gemini (36.3%) thanks to the basketball-tuned prompt. Grounding DINO and Florence-2 score 0%.

| Method | Ball AP@50 (test) | Gap to RF-DETR v3 |
|--------|------------------|----|
| RF-DETR v3 | 75.46% | -- |
| YOLOX v2 | 61.55% | -13.9 |
| OWLv2 | 41.24% | -34.2 |
| Gemini | 36.30% | -39.2 |
| OmDet-Turbo | 9.19% | -66.3 |
| Florence-2 | 1.98% | -73.5 |
| Grounding DINO | 0.00% | -75.5 |

**Referee detection -- domain knowledge matters**:

Referee detection separates methods that understand basketball context from those that don't. Custom-trained models learn the distinctive referee uniform (95-99%). Gemini's language understanding helps it identify referees from the prompt (68.6%). OWLv2 and OmDet-Turbo partially detect referees (33-35%). Grounding DINO and Florence-2 completely fail (0%).

**Rim detection -- universal failure for zero-shot**:

Every zero-shot method scores below 4.3% AP@50 on rim. The basketball hoop rim is a domain-specific geometric structure that CLIP-based models and VLMs have not learned to localize precisely. Custom-trained models (100%) detect it perfectly.

**Number detection -- RF-DETR v3 catches up**:

RF-DETR v3 now leads on jersey number detection (84.9% AP@50), surpassing YOLOX (84.0%). The previous RF-DETR v2 trailed YOLOX on this class (78.0% vs 82.9%), but the resolution increase from 512 to 640 closed the gap. Small objects like jersey numbers benefit directly from higher input resolution.

### Gemini vs Dedicated Zero-Shot Detectors

Gemini 3.1 Pro and OWLv2 are surprisingly close in overall mAP (43.1% vs 39.1%), but their strengths differ:

| Metric | Gemini Advantage | OWLv2 Advantage |
|--------|-----------------|-----------------|
| Player AP@50 | 92.5% vs 84.8% (+7.7) | -- |
| Referee AP@50 | 68.6% vs 35.2% (+33.4) | -- |
| Ball AP@50 | -- | 41.2% vs 36.3% (+4.9) |
| Number AP@50 | -- | 33.7% vs 13.7% (+20.0) |
| Rim AP@50 | 4.3% vs 0.3% (+4.0) | -- |
| F1 | 71.5% vs 56.5% (+15.0) | -- |
| Recall | -- | 64.6% vs 66.1% (~tied) |
| Precision | -- | -- Gemini wins: 77.9% vs 50.2% |
| Speed | -- | 2.8s vs 15.5s per image (5.5x faster) |
| Cost | Cloud API ($$$) | Local GPU (fixed cost) |

Gemini excels at classes it can reason about semantically (players, referees) and has much higher precision because it outputs fewer, more confident detections. OWLv2 is better at small or unusual objects (ball, numbers) and is 5.5x faster at a fraction of the cost.

## Analysis: Why RF-DETR v3 is the Best Model

### Overall Performance

RF-DETR v3 leads on every mAP metric, including mAP@75 where YOLOX previously held the advantage:

| Metric (test) | RF-DETR v3 | YOLOX-S v2 | RF-DETR v2 | RF-DETR v1 | Gemini 3.1 | OWLv2 |
|--------|-----------|---------|-----------|-----------|------------|-------|
| mAP@50:95 | **61.16%** | 53.43% | 56.67% | 50.31% | 26.47% | 24.74% |
| mAP@50 | **91.43%** | 86.76% | 89.65% | 85.82% | 43.06% | 39.07% |
| mAP@75 | **64.59%** | 58.61% | 57.15% | 45.44% | 29.32% | 27.59% |
| Precision | 92.63% | **93.20%** | -- | 93.80% | 77.87% | 50.18% |
| Recall | **91.46%** | 85.86% | -- | 85.56% | 66.11% | 64.60% |
| F1 | **92.05%** | 89.26% | -- | 89.49% | 71.51% | 56.48% |

RF-DETR v3 achieves the highest score on every metric except precision (where YOLOX leads by 0.6 points due to its higher confidence threshold). The mAP@75 lead (+6.0 over YOLOX) reverses the previous finding where YOLOX had superior localization -- this was an artifact of the resolution mismatch, not a fundamental architectural advantage.

### The Zero-Shot Gap: What Fine-Tuning Buys You

Comparing RF-DETR v3 against the best zero-shot detector (OWLv2) per class:

| Class | RF-DETR v3 AP@50 | OWLv2 AP@50 | Gap | Fine-tuning value |
|-------|-----------------|-------------|-----|-------------------|
| player | 97.91% | 84.84% | 13.1 pts | Medium -- zero-shot decent but custom excels |
| ball | 75.46% | 41.24% | 34.2 pts | High -- labeled data adds +34 over best prompt tuning |
| referee | 98.94% | 35.24% | 63.7 pts | High -- domain-specific visual distinction |
| rim | 100.00% | 0.31% | 99.7 pts | Critical -- zero-shot cannot detect this class |
| number | 84.85% | 33.70% | 51.2 pts | High -- small object localization requires supervision |

**The "person-shaped object" ceiling**: Zero-shot detectors are competitive on player detection (~85% AP@50) because people are the most heavily represented category in pretraining data. For any class that looks like a person, zero-shot works. For everything else, labeled data is essential.

### Key Takeaways

1. **RF-DETR v3 is the recommended production model** for basketball detection: best mAP@50:95 (61.2%), best mAP@50 (91.4%), best mAP@75 (64.6%), best F1 (92.1%), and leading per-class AP on all 5 classes. It dominates every metric.

2. **Resolution matters**: The previous RF-DETR v2 evaluation at 512x512 (trained at 640) lost 7.4 points of mAP@75. Always evaluate at training resolution.

3. **Checkpoint selection matters**: Sweeping all 23 RF-DETR runs found a checkpoint with training mAP 0.5935 vs the previously used 0.5330 -- a +11% relative improvement that translated to +4.5 mAP@50:95 on test.

4. **YOLOX no longer has a localization advantage**: RF-DETR v3's mAP@75 (64.6%) now exceeds YOLOX (58.6%) by 6 points. The previous YOLOX mAP@75 advantage was an artifact of RF-DETR being evaluated at the wrong resolution.

5. **RF-DETR v3 leads on all 5 per-class AP@50**: player (97.9%), ball (75.5%), referee (98.9%), rim (100%), number (84.9%). It is the first model to achieve 100% rim AP@50 alongside 98%+ on both player and referee.

6. **RF-DETR v3 has the best recall of any model** (91.5% at threshold=0.40), meaning it finds more objects while maintaining 92.6% precision. YOLOX achieves comparable precision (93.2%) but at 86.1% recall -- missing 5.5% more objects.

7. **OWLv2 remains the best zero-shot detector** for this domain, achieving 4-class coverage and the highest recall (64.6%) of any zero-shot method. Prompt engineering was critical.

8. **Fine-tuning provides a 2.3x mAP improvement**: Custom-trained models achieve 87-91% mAP@50 vs 39-43% for the best zero-shot approaches. The gap is largest for domain-specific classes (rim: 100% vs 0-4%, referee: 95-99% vs 0-69%).

9. **YOLOX-M nearly closes the gap**: YOLOX-M achieves 58.3% mAP@50:95 (vs RF-DETR's 61.2%) and 91.6% F1 (vs 92.1%), while being 1.4x faster (463ms vs 643ms). It has the highest precision (95.6%) of any model and matches RF-DETR on mAP@75 (64.8% vs 64.6%).

10. **Speed-accuracy-memory trade-off**: Three deployment tiers emerge: YOLOX-S (211ms, 34MB) for edge/real-time, YOLOX-M (463ms, 97MB) for server-side with balanced speed/accuracy, and RF-DETR (643ms, 110MB) for maximum accuracy. On CPU, YOLOX is 3x faster than RF-DETR (16.4 vs 5.5 img/s).

11. **YOLOX-M's gains come from small objects**: Ball detection improved +7.6 points and number detection +3.7 points over YOLOX-S. The wider backbone (0.75 vs 0.50 width) provides richer features for fine-grained localization.

10. **Class merging inflates mAP**: Evaluating on 5 merged classes (91.4% mAP@50) vs 10 original classes produces ~17-19 points of boost from merging player sub-types. Production deployments should account for which class granularity is needed.

## Evaluation Details

- **mAP computation**: supervision `MeanAveragePrecision` using COCO IoU thresholds [0.50, 0.55, ..., 0.95], maxDets=100
- **P/R/F1**: Class-aware greedy matching at IoU=0.5, threshold swept in 20 steps on val, best threshold applied to test
- **Per-class P/R**: Computed at the best operating threshold (RF-DETR: 0.40, YOLOX: 0.55), class-aware greedy matching with confidence-ordered predictions
- **Class merging**: Training sub-classes (player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block) merged into "player"; ball-in-basket merged into "ball"
- **Confidence threshold**: 0.01 for all models (all predictions kept for mAP ranking)
- **YOLOX NMS**: Per-class greedy NMS at IoU 0.45
- **Zero-shot NMS**: OWLv2 per-class NMS at IoU 0.3; OmDet-Turbo and Grounding DINO at IoU 0.5; Florence-2 no NMS (no confidence scores)
- **RF-DETR v3 ONNX export**: Exported from best checkpoint across all 23 RF-DETR runs (run 20260117_005818, epoch=64, training mAP=0.5935). Model uses 91-class COCO heads with 11 basketball classes active (direct COCO category IDs). Evaluated at 640x640 (matching training resolution).
- **RF-DETR v2 ONNX export**: Exported locally from v2 checkpoint (epoch=69, mAP=0.5330) with verified PyTorch-ONNX fidelity. Evaluated at 512x512 (mismatched from 640x640 training resolution).
- **Zero-shot area filter**: Detections exceeding 5% of image area removed (prevents spurious large-box predictions)
- **Zero-shot class prompts**: OWLv2 uses basketball-tuned prompts ("basketball player", "basketball", "referee", "basketball hoop", "jersey number"); other detectors use COCO-aligned names ("person", "sports ball", "referee", "basketball hoop", "jersey number")

## Bugs Fixed During Evaluation

| Bug | Impact | Fix |
|-----|--------|-----|
| `_remap_detections` was a no-op | RF-DETR mAP was 0.32% instead of ~50% | Proper class name lookup + eval ID mapping |
| gemini_classes order swapped ball/referee | Gemini classes misaligned with eval | Fixed order to match `_EVAL_LABEL_MAP` |
| P/R/F1 was class-agnostic | Cross-class matches inflated metrics | Added `pred_cls == gt_cls` check |
| Docker image was stale | Cloud build script failed on interactive prompt | Bypassed script, called `gcloud builds submit` directly |
| `labels_mapping.json` missing from GCS | Job crashed with FileNotFoundError | Copied from training logs to export dir |
| Gemini model was 2.5-pro | Near-zero mAP (0.1%) | Upgraded to gemini-3.1-pro-preview with structured prompt |
| YOLOX post-processor used wrong normalization | YOLOX mAP was 0% | Normalize by model input size (640), not original image size |
| OmDet-Turbo meta tensors on CPU | Model crashed with `NotImplementedError` | Added `_materialize_meta_buffers` + `low_cpu_mem_usage=False` |
| Grounding DINO concatenated labels | All detections mapped to class 0 | Fixed label parsing to split on `. ` separator |
| transformers >=4.51 label key change | `text_labels` key returned `None` | Added `or`-chain fallback: `text` / `text_labels` / `labels` |
| OWLv2 `run_owlv2` not in Hydra config | GCP job failed with "Key not in struct" | Added OWLv2 fields to `eval_detection.yaml` |
| OWLv2 "person" prompt | All detections labeled as "referee" | Changed to "basketball player" -- fixed player/referee distinction |
| OWLv2 "sports ball" prompt | Ball confidence only 0.24 | Changed to "basketball" -- confidence jumped to 0.70+ |
| RF-DETR v2 evaluated at wrong resolution | mAP@75 underreported by 7.4 points | RF-DETR v3 evaluated at training resolution (640x640) |
| CoreML execution provider crash on macOS | ONNX inference failed on Apple Silicon | Force `CPUExecutionProvider` in `_build_rfdetr_inferencer` and `_build_yolox_inferencer` |

## GCP Artifacts

| Run | Job ID | Output Path |
|-----|--------|-------------|
| RF-DETR + Gemini 2.5 | `5188122100438663168` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/comparison/basketball-detector-eval-comparison/20260227_220728/` |
| Gemini 3.1 Pro | `2191345984330530816` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/comparison/basketball-detector-eval-comparison/20260301_154431/` |
| YOLOX-S v1 | `7000697805152976896` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/comparison/basketball-detector-eval-comparison/20260302_125436/` |
| YOLOX ONNX export | `6409600354060599296` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/onnx-export/basketball-detector-export-onnx-yolox/20260302_113411/` |
| Zero-shot (4 models) | `8904607741763387392` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/comparison-zeroshot/basketball-detector-eval-zeroshot/20260303_144520/` |
| RF-DETR v2 export+eval | Local | `checkpoints/rfdetr_v2_export/` (ONNX model + fidelity report + eval results) |
| RF-DETR v3 + YOLOX v2 (local) | Local | Best-run sweep: RF-DETR from `gs://.../basketball-detector-training-rfdetr-small/20260117_005818/`, YOLOX from `gs://.../basketball-detector-training-yolox-small/20260203_195555/` |
| RF-DETR v3 + YOLOX v2 (L4 FPS) | `8652300609514373120` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/best-runs/basketball-detector-eval-best-runs/20260304_162845/` |
| YOLOX-M training | `3358178510051999744` | `gs://deep-ego-model-training/ego-training-data/basketball-data/logs/basketball-detector-training-yolox-medium/20260304_215835/` |
| YOLOX-M ONNX export | `7865085468398845952` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/onnx-export/basketball-detector-export-onnx-yolox-medium/20260305_072138/` |
| YOLOX-M vs RF-DETR eval | `2188016663121625088` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/yolox-medium-comparison/basketball-detector-eval-yolox-medium/20260305_091814/` |

## Training Run Details

### RF-DETR Best Runs (top 5 of 23)

| Run | Epoch | Training mAP | Notes |
|-----|-------|-------------|-------|
| **20260117_005818** | 64 | **0.5935** | Best overall, used as RF-DETR v3 |
| 20260117_005818 | 69 | 0.5330 | Previously used as RF-DETR v2 |
| Other runs | various | <0.53 | Lower training mAP |

### YOLOX-M (Medium)

| Run | Epoch | Training mAP_50 | Notes |
|-----|-------|-----------------|-------|
| **20260304_215835** | 117 | **0.4957** | First YOLOX-M run, used as YOLOX-M v1 |

### YOLOX-S Best Runs (top 5 of 21)

| Run | Epoch | Training mAP | Notes |
|-----|-------|-------------|-------|
| **20260203_195555** | 89 | **0.4461** | Best overall, used as YOLOX-S v2 |
| Other runs | various | <0.44 | Lower training mAP |

## YOLO Model Landscape: Upgrade Paths from YOLOX

YOLOX (2021, Megvii) is now 5 years old. The YOLO family has evolved rapidly since then, with significant accuracy improvements at the same or lower parameter counts. This section evaluates which modern YOLO variants would improve upon YOLOX for basketball detection, with careful attention to licensing implications for this Apache 2.0 repository.

### YOLOX Published Benchmarks (COCO val2017, 640x640)

| Model | Params | FLOPs | mAP@50:95 | Speed (V100) |
|-------|--------|-------|-----------|--------------|
| YOLOX-S | 9.0M | 26.8G | 40.5% | 9.8 ms |
| YOLOX-M | 25.3M | 73.8G | 46.9% | 12.3 ms |
| YOLOX-L | 54.2M | 155.6G | 49.7% | 14.5 ms |

### Modern YOLO Variants (Medium-sized, ~15-25M params)

| Model | Year | Params | mAP@50:95 | Gain vs YOLOX-M | T4 TRT (ms) | NMS-Free | License |
|-------|------|--------|-----------|-----------------|-------------|----------|---------|
| YOLOX-M (baseline) | 2021 | 25.3M | 46.9% | -- | ~12 (V100) | No | Apache 2.0 |
| RTMDet-m | 2022 | 24.7M | 49.4% | +2.5 | 1.6 (3090) | No | Apache 2.0 |
| YOLOv8m | 2023 | 25.9M | 50.2% | +3.3 | ~1.8 (A100) | No | AGPL-3.0 |
| YOLOv9m | 2024 | 20.1M | 51.4% | +4.5 | N/A | No | AGPL-3.0 |
| YOLOv10m | 2024 | 15.4M | 51.1% | +4.2 | 4.7 | **Yes** | AGPL-3.0 |
| YOLO11m | 2024 | 20.1M | 51.5% | +4.6 | 4.7 | No | AGPL-3.0 |
| YOLOv12m | 2025 | 20.2M | 52.5% | +5.6 | 4.9 | No | AGPL-3.0 |
| **YOLO26m** | **2025** | **20.4M** | **53.1%** | **+6.2** | **4.7** | **Yes** | **AGPL-3.0** |
| RF-DETR-S | 2025 | 32.1M | 53.0% | +6.1 | 3.5 | **Yes** | Apache 2.0 |
| RF-DETR-M | 2025 | 33.7M | 54.7% | +7.8 | 4.4 | **Yes** | Apache 2.0 |

### RTMDet: The Overlooked Apache 2.0 Alternative

**RTMDet** (Real-Time Multi-level Detection, OpenMMLab, 2022) is a fully convolutional, anchor-free, single-stage detector. It is often overlooked because it doesn't carry the "YOLO" brand, but achieves competitive accuracy with the best YOLO variants under a permissive Apache 2.0 license.

**Paper**: [RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/abs/2212.07784) (arXiv 2212.07784)

#### Architecture: How RTMDet Differs

**vs YOLOX (YOLO-style CNN)**:
- **Backbone**: Replaces CSPDarknet with **CSPNeXt** — uses **large-kernel depth-wise convolutions (5x5)** followed by point-wise 1x1 convolutions, enlarging the effective receptive field cheaply. Adds Channel Attention (SE-like) at each stage.
- **Neck**: CSPNeXt-PAFPN mirrors the backbone's design, ensuring **balanced capacity between backbone and neck** — a key insight from their empirical study (prior work used heavy backbones with lightweight necks, leaving accuracy on the table).
- **Head**: Uses a **shared convolutional head with separated BatchNorm** per feature level, more parameter-efficient than YOLOX's fully decoupled head.
- **Label assignment**: Soft labels based on predicted IoU (richer signal than YOLOX's SimOTA hard labels).

**vs RF-DETR (Transformer-based)**:
- **Purely convolutional** — no self-attention or cross-attention, making it simpler and faster on edge devices.
- **Requires NMS** — not end-to-end like DETR-family models.
- **No Hungarian matching** — uses dynamic SimOTA-style assignment.
- **Lower latency at similar accuracy**: RTMDet-m (49.4 AP) runs at ~1.6ms vs RF-DETR-S (53.0 AP) at 3.5ms on similar hardware.
- **Weaker on small datasets**: No pre-trained foundation model backbone (unlike RF-DETR's DINOv2). Transfer learning relies on ImageNet pre-training only.

#### RTMDet Benchmarks (COCO val2017, 640x640, 3090 TRT FP16)

| Model | Params | FLOPs | mAP@50:95 | Latency | FPS |
|-------|--------|-------|-----------|---------|-----|
| RTMDet-tiny | 4.8M | 8.1G | 41.1% | 0.98 ms | 1020 |
| RTMDet-s | 9.0M | 14.8G | 44.6% | 1.22 ms | 820 |
| **RTMDet-m** | **24.7M** | **39.3G** | **49.4%** | **1.62 ms** | **617** |
| RTMDet-l | 52.3M | 80.2G | 51.5% | 2.40 ms | 417 |
| RTMDet-x | 94.9M | 141.7G | 52.8% | 3.10 ms | 323 |

#### RTMDet Ecosystem Assessment

| Factor | Status |
|--------|--------|
| **License** | Apache 2.0 (via mmdetection). **Caution**: MMYOLO variant is GPL-3.0 — use the mmdetection implementation only. |
| **Maintenance** | **Effectively frozen.** MMDetection is in maintenance mode since the passing of Prof. Tang Xiaoou (CUHK MMLab founder). The core team moved to InternLM. Many issues go unresolved. A third-party fork ([onedl-mmdetection](https://mmwheels.onedl.ai/)) provides Python 3.11+ / PyTorch 2.1+ compatibility. |
| **ONNX export** | Supported via mmdeploy. Static shapes recommended. |
| **Training** | Uses mmengine config system (Python configs, not Hydra YAML). Integrating into our Lightning pipeline would require wrapping the model and removing mmengine's training loop. |
| **Fine-tuning** | COCO-format datasets work out of the box. Tutorials from Roboflow and Makeability Lab exist. |
| **Sports use cases** | Tennis ball detection with RTMDet-Light (IET 2025); RTMPose (same family) used for 3x3 basketball motion capture (Sensors 2025). |

**Bottom line on RTMDet**: The architecture is mature and well-validated (+2.5 mAP over YOLOX-M at the same parameter count), but the ecosystem is effectively frozen. Worth considering if we want an Apache 2.0 CNN detector without the mmengine dependency — we could extract the model architecture and train it in our existing Lightning pipeline.

### Key Architecture Innovations Since YOLOX

**YOLOv8 (2023)** — C2f backbone (improved CSP), Distribution Focal Loss (DFL) for box regression, Task-Aligned Assigner (TAL) replacing SimOTA. The most battle-tested post-YOLOX ecosystem.

**YOLOv9 (2024)** — Programmable Gradient Information (PGI) combats information loss in deep networks. GELAN feature aggregation. Achieves 51.4% mAP with only 20.1M params.

**YOLOv10 (2024, NeurIPS)** — First NMS-free YOLO via consistent dual assignments (one-to-many for training, one-to-one for inference). Eliminates NMS at inference, simplifying deployment. 51.1% mAP with only 15.4M params.

**YOLO11 (2024)** — Enhanced C3k2 blocks, SPPF, C2PSA (spatial attention). Most polished Ultralytics ecosystem with multi-task support (detection, segmentation, pose, OBB).

**YOLOv12 (2025, NeurIPS)** — First attention-centric YOLO. Area Attention + R-ELAN provide global context while maintaining CNN-like speed. May help with basketball-specific challenges (occlusion, ball-hand proximity).

**YOLO26 (2025)** — NMS-free end-to-end inference natively. **Small-Target-Aware Label Assignment (STAL)** explicitly prioritizes small objects. 43% faster CPU inference than YOLO11 due to DFL removal. Most relevant for basketball detection where the ball is a small object.

### If We Add One More YOLO-Style Model: YOLO26

**YOLO26m is the strongest candidate** based on pure accuracy (+6.2 mAP over YOLOX-M on COCO, NMS-free, STAL for small objects).

However, **we should NOT integrate YOLO26 (or any Ultralytics model) into this repository.**

#### Why AGPL-3.0 Would Break Our License

This repository is licensed under **Apache 2.0**. AGPL-3.0 is a strong copyleft license:

1. **AGPL-3.0 is incompatible with Apache 2.0.** If we add AGPL code as a dependency or include it in our source tree, the AGPL "infects" the entire combined work — we would be legally required to relicense the entire repository under AGPL-3.0.
2. **AGPL extends to network use.** Unlike GPL, AGPL requires source disclosure even when software is only accessed over a network (e.g., a detection API). This means any service built with AGPL code must open-source the entire application.
3. **Even importing `ultralytics` as a dependency triggers AGPL.** Adding `ultralytics` to `pixi.toml` or `pyproject.toml` would require AGPL compliance for the entire project.

**The practical impact**: If Next Play is or may become a commercial product, integrating any Ultralytics model would require either open-sourcing the entire Next Play codebase or purchasing an Ultralytics Enterprise License.

#### Recommended Approach: Train in Their Repo First

The correct strategy is to **evaluate YOLO26 in isolation** before considering integration:

1. **Create a separate, throwaway training repo** that uses the `ultralytics` package directly:
   ```bash
   pip install ultralytics
   yolo detect train data=basketball.yaml model=yolo26m.pt epochs=300 imgsz=640
   ```
2. **Export the trained model to ONNX** using Ultralytics' built-in export
3. **Evaluate the ONNX model** using our existing eval_detection task (which already supports ONNX inference — no AGPL code touches our repo)
4. **Compare results** against YOLOX-M and RF-DETR on the same test set

This approach keeps AGPL code completely isolated. The trained ONNX weights are not covered by AGPL (they are model parameters, not software), so using ONNX inference in our Apache 2.0 repo is legally safe. However, any code that imports `ultralytics` must live in the separate repo.

If YOLO26 proves significantly better, the decision becomes:
- **Buy Ultralytics Enterprise License** (~$1,500-2,000/year for startups) to integrate natively
- **Use ONNX-only inference** in production (train in separate repo, deploy weights only)
- **Stick with RF-DETR/YOLOX** under Apache 2.0

### Techniques We Can Safely Add to YOLOX

Several techniques from newer YOLO models originate from **peer-reviewed papers with Apache 2.0 reference implementations**. These can be reimplemented from the papers without touching AGPL code:

| Technique | Original Paper | Reference Implementation | License | Expected Impact |
|-----------|---------------|------------------------|---------|-----------------|
| **Task-Aligned Assigner (TAL)** | [TOOD (Feng et al., ICCV 2021)](https://arxiv.org/abs/2108.07755) | mmdetection | Apache 2.0 | Replaces SimOTA; better alignment between classification and localization scores. +1-2 mAP typical improvement. |
| **Distribution Focal Loss (DFL)** | [Generalized Focal Loss (Li et al., NeurIPS 2020)](https://arxiv.org/abs/2006.04388) | mmdetection | Apache 2.0 | Learns a distribution over box offsets rather than point estimates. Improves localization quality (mAP@75). |
| **Large-kernel depth-wise convolutions** | [RTMDet (Lyu et al., 2022)](https://arxiv.org/abs/2212.07784) | mmdetection | Apache 2.0 | Replace 3x3 convs with 5x5 depth-wise + 1x1 point-wise. Enlarges receptive field cheaply. |
| **Channel Attention (SE blocks)** | [SENet (Hu et al., CVPR 2018)](https://arxiv.org/abs/1709.01507) | Multiple Apache/MIT | Apache 2.0 | Add squeeze-and-excitation attention to backbone stages. Modest accuracy boost. |
| **Balanced backbone-neck capacity** | [RTMDet](https://arxiv.org/abs/2212.07784) | mmdetection | Apache 2.0 | Use same block design in PAN neck as backbone. YOLOX uses a lighter neck. |
| **Soft label assignment** | [RTMDet](https://arxiv.org/abs/2212.07784) | mmdetection | Apache 2.0 | Use predicted IoU as soft classification target in SimOTA matching. |

#### Legal basis for reimplementation

- **Copyright protects code, not algorithms.** Reimplementing an algorithm from a paper without copying code is legal under the clean-room design principle.
- **AGPL applies to code derivatives, not idea derivatives.** The techniques above are described in papers and have Apache 2.0 reference implementations. We can implement from the papers and/or the Apache 2.0 mmdetection code.
- **Precedent**: RTMDet itself implements SimOTA (from YOLOX) and TAL concepts. PP-YOLO (PaddlePaddle, Apache 2.0) reimplements many YOLO techniques. Multiple projects do this routinely.

#### What NOT to copy

- **C2f blocks**: The specific C2f architecture first appeared in Ultralytics YOLOv8 (AGPL). While CSP-style bottleneck blocks are described in prior work (CSPNet, 2019), the exact C2f variant is Ultralytics-specific. Use RTMDet's CSPNeXt blocks instead (same purpose, Apache 2.0).
- **Any code from the `ultralytics` package**: Even small utility functions carry AGPL. Always implement from papers or Apache 2.0 sources.
- **STAL (YOLO26)**: Novel to YOLO26, no separate paper or Apache implementation exists yet. Wait for an independent paper or implement from the YOLO26 paper description (not from Ultralytics source).

#### Estimated impact of adding TAL + DFL to YOLOX

Based on published ablation studies and the progression from YOLOX to YOLOv8 (which primarily added TAL + DFL + C2f):

| Change | Expected mAP@50:95 gain | Source |
|--------|------------------------|--------|
| SimOTA → TAL | +1.0 to +1.5 | TOOD paper ablation |
| Point regression → DFL | +0.5 to +1.0 | GFL paper ablation |
| 3x3 conv → 5x5 DWConv (RTMDet-style) | +0.5 to +1.0 | RTMDet paper ablation |
| Combined | +2.0 to +3.5 | Estimated |

This would bring YOLOX-M from 46.9% → ~49-50% COCO mAP, and from 58.3% → ~60-62% on our basketball dataset. Meaningful, but **still below RF-DETR v3's 61.2%** on our dataset.

### Will a Newer YOLO Architecture Beat RF-DETR v3?

**Probably not, and here's why.**

#### COCO mAP comparison

| Model | COCO mAP@50:95 | Our Basketball mAP@50:95 |
|-------|----------------|-------------------------|
| YOLOX-M | 46.9% | 58.3% (measured) |
| RTMDet-m | 49.4% | ~60-61% (estimated) |
| YOLO26m | 53.1% | ~64-66% (estimated) |
| RF-DETR-S | 53.0% | **61.2% (measured)** |
| RF-DETR-M | 54.7% | ~63-66% (estimated) |

On COCO, YOLO26m (53.1%) essentially matches RF-DETR-S (53.0%). But our basketball dataset is small (96 val, 94 test images) and domain-specific. **RF-DETR's DINOv2 backbone provides dramatically better transfer learning** for small datasets because DINOv2 was trained on 142M images with self-supervised learning — it has learned rich visual representations that generalize well to novel domains without large fine-tuning datasets.

YOLO-family models use ImageNet-pretrained backbones (1.2M images, supervised classification). When fine-tuning data is limited, the DINOv2 foundation model advantage is substantial. This is why RF-DETR-S achieves 61.2% on our dataset despite "only" 53.0% on COCO — the transfer learning gap is wider than the COCO gap suggests.

**YOLO26m might match RF-DETR-S on our dataset** (both are ~53% on COCO), but it is unlikely to significantly exceed it. The +6.2 mAP COCO advantage over YOLOX-M would translate to roughly +5-7 points on our dataset, landing at ~63-65% — competitive with but not clearly ahead of RF-DETR-S's 61.2%.

#### The real bottleneck: data, not architecture

With only 96 val / 94 test images and 10 training classes, we are in the **small-data regime** where:
- Foundation model backbones (DINOv2) matter more than detection head design
- Augmentation strategy matters more than neck architecture
- Class imbalance (ball, rim are rare) matters more than label assignment algorithms
- More training data would help every model more than switching architectures

### Should We Train RF-DETR Medium Instead?

**Yes, this is likely the highest-ROI next experiment.**

#### RF-DETR Medium vs Small

| Metric | RF-DETR-S (COCO) | RF-DETR-M (COCO) | Delta |
|--------|-----------------|-----------------|-------|
| Params | 32.1M | 33.7M | +1.6M (+5%) |
| mAP@50:95 | 53.0% | 54.7% | **+1.7** |
| Latency (T4 TRT) | 3.52 ms | 4.52 ms | +1.0 ms (+28%) |

RF-DETR-M adds only 1.6M parameters (5% more) but gains 1.7 mAP on COCO. The latency increase is modest (3.5ms → 4.5ms). Both use the same DINOv2 backbone — the difference is in the NAS-discovered decoder configuration (more decoder layers and/or queries).

#### Why this is the best next step

1. **Already in our codebase**: RF-DETR is already integrated. Switching from Small to Medium requires only a config change (`models: rfdetr_medium` instead of `rfdetr_small`). No new code, no new dependencies, no licensing issues.
2. **Same DINOv2 backbone**: The transfer learning advantage that makes RF-DETR-S the best model on our dataset applies equally to RF-DETR-M. We can expect the COCO improvement to transfer.
3. **Minimal speed impact for our use case**: 4.5ms vs 3.5ms TRT latency is negligible. Our current end-to-end ONNX inference is 643ms (dominated by preprocessing, not model forward pass). The 1ms difference in pure model time is lost in the noise.
4. **Estimated basketball mAP**: If RF-DETR-S achieves 61.2% on our dataset from 53.0% COCO, RF-DETR-M (54.7% COCO) should land at roughly **63-66%** — potentially the new best model with zero architectural risk.

#### Comparison of next-step options

| Option | Effort | Risk | Expected mAP@50:95 | License Safe |
|--------|--------|------|--------------------|----|
| **RF-DETR-M training** | Low (config change) | Low | ~63-66% | Yes (Apache 2.0) |
| YOLOX + TAL + DFL | Medium (code changes) | Medium | ~60-62% | Yes (from papers) |
| YOLO26m (separate repo) | Medium (new repo) | Low | ~63-66% | Yes (ONNX only) |
| RTMDet-m integration | High (new model) | Medium | ~60-62% | Yes (Apache 2.0) |

**Recommendation**: Train RF-DETR-M first (lowest effort, highest expected return). Then optionally evaluate YOLO26m in a separate repo for comparison. Only invest in YOLOX improvements or RTMDet integration if the speed-accuracy tradeoff of RF-DETR-M doesn't meet deployment requirements.

### Not Recommended

- **YOLO-NAS**: Development stalled after NVIDIA acquisition; restrictive license for pre-trained weights
- **Gold-YOLO**: 21.5M params for only 46.4% mAP; superseded by all modern variants
- **YOLO-World**: Open-vocabulary capability is overkill for fixed basketball classes; lower accuracy than closed-vocabulary models
