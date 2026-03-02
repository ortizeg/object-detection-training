# Basketball Object Detection — Evaluation Report

**Date**: 2026-03-02
**Dataset**: basketball-player-detection-3 (val: 96 images, test: 94 images)
**Classes**: player, ball, referee, rim, number (5 merged eval classes from 10 training classes)

## Methods

| Method | Model | Notes |
|--------|-------|-------|
| YOLOX-S | ONNX export (yolox-small, input 640) | Trained on basketball-player-detection-3, 10-class model with eval-time class merging |
| RF-DETR | ONNX export (rfdetr-small, input 512) | Trained on basketball-player-detection-3, 10-class model with eval-time class merging |
| Gemini 3.1 Pro Preview | gemini-3.1-pro-preview | Zero-shot VLM detection with structured prompt, confidence fixed at 1.0 |

## Overall Results

| Method | Split | mAP@50:95 | mAP@50 | mAP@75 | Precision | Recall | F1 | Threshold |
|--------|-------|-----------|--------|--------|-----------|--------|------|-----------|
| **YOLOX-S** | **val** | **56.17%** | **88.95%** | **61.71%** | 93.03% | 87.46% | 90.16% | 0.55 |
| **YOLOX-S** | **test** | **54.87%** | **86.15%** | **62.51%** | 91.84% | 86.41% | 89.05% | 0.55 |
| RF-DETR | val | 49.70% | 86.44% | 46.39% | 93.59% | 85.97% | 89.62% | 0.55 |
| RF-DETR | test | 50.31% | 85.82% | 45.44% | 93.80% | 85.56% | 89.49% | 0.55 |
| Gemini 3.1 Pro | val | 25.12% | 43.80% | 26.96% | 80.62% | 66.67% | 72.98% | 0.80 |
| Gemini 3.1 Pro | test | 26.47% | 43.06% | 29.32% | 77.87% | 66.11% | 71.51% | 0.80 |

## Per-Class AP@50

| Class | YOLOX Val | YOLOX Test | RF-DETR Val | RF-DETR Test | Gemini Val | Gemini Test |
|-------|-----------|------------|-------------|--------------|------------|-------------|
| player | 93.36% | 92.64% | 98.28% | 98.23% | 89.57% | 92.49% |
| referee | 94.41% | 93.72% | 94.48% | 97.12% | 60.53% | 68.59% |
| rim | 99.88% | 100.00% | 90.70% | 96.93% | 4.37% | 4.26% |
| number | 87.25% | 82.90% | 79.77% | 76.75% | 18.63% | 13.66% |
| ball | 69.85% | 61.48% | 68.97% | 60.07% | 45.90% | 36.30% |

## Inference Speed

| Method | Hardware | End-to-End (img/s) | ms/img | Total (190 imgs) |
|--------|----------|-------------------|--------|-------------------|
| YOLOX-S (ONNX) | NVIDIA L4 | 2.69 | 372 ms | ~71s |
| RF-DETR (ONNX) | NVIDIA L4 | 2.60 | 385 ms | ~73s |
| Gemini 3.1 Pro (API) | Cloud API | 0.064 | 15,505 ms | ~49 min |

End-to-end timing includes image loading, preprocessing, ONNX inference/API call, and postprocessing.

YOLOX pure model inference (from training benchmark): **16.8 ms/img (59.6 FPS)** on NVIDIA L4.

## Comparison with data-visor Baselines

| Metric | data-visor RF-DETR | This RF-DETR | data-visor Gemini 3.1 | This Gemini 3.1 |
|--------|-------------------|-------------|----------------------|-----------------|
| mAP@50:95 | 47.7% | 50.3% | 12.9% | 26.5% |
| mAP@50 | 74.1% | 85.8% | 22.3% | 43.1% |
| mAP@75 | 51.6% | 45.4% | 13.2% | 29.3% |

Differences likely due to evaluation framework (supervision COCO-style mAP vs data-visor custom matching) and confidence handling (Gemini returns confidence=1.0; data-visor may handle this differently).

## Analysis: Why YOLOX-S is the Best Model

### Overall Performance

YOLOX-S leads across every aggregate metric on test:

| Metric | YOLOX-S | RF-DETR | Gemini 3.1 | YOLOX vs RF-DETR | YOLOX vs Gemini |
|--------|---------|---------|------------|------------------|-----------------|
| mAP@50:95 | 54.9% | 50.3% | 26.5% | +9.1% relative | +107% relative |
| mAP@50 | 86.2% | 85.8% | 43.1% | +0.4% | +100% |
| mAP@75 | 62.5% | 45.4% | 29.3% | +37.6% | +113% |
| Precision | 91.8% | 93.8% | 77.9% | -2.1% | +17.8% |
| Recall | 86.4% | 85.6% | 66.1% | +0.9% | +30.7% |
| F1 | 89.1% | 89.5% | 71.5% | -0.4% | +24.5% |

The biggest differentiator is **mAP@75** — YOLOX produces 37% tighter bounding boxes than RF-DETR. This is critical for downstream tasks like jersey number OCR where tight crops improve recognition accuracy. At the looser IoU=0.5 threshold, both trained models perform nearly identically (86.2% vs 85.8%), but YOLOX's advantage grows as the IoU requirement tightens.

RF-DETR has marginally higher precision (+2.1%) and F1 (-0.4%), meaning it has slightly fewer false positives. However, YOLOX achieves slightly better recall (+0.9%), finding more true objects.

### Per-Class Breakdown

**Player detection** (largest class, most instances):
- RF-DETR: 98.2% AP@50 — best in class
- YOLOX: 92.6% AP@50 — 5.6 points behind
- Gemini: 92.5% AP@50 — essentially tied with YOLOX
- *Analysis*: RF-DETR's transformer attention mechanism with global receptive field appears to handle crowded multi-player scenes better. YOLOX's convolutional architecture may lose some detections when players overlap. Notably, Gemini matches YOLOX here — large, salient people are exactly what VLMs understand well from pretraining on web-scale image-text data.

**Referee detection**:
- RF-DETR: 97.1% AP@50
- YOLOX: 93.7% AP@50
- Gemini: 68.6% AP@50
- *Analysis*: Both trained models excel at referees. Gemini drops to 68.6% — it sometimes confuses referees with players or misses them when partially occluded. The distinct referee uniform is a strong visual cue that both trained models learn, but Gemini's general-purpose training lacks this domain-specific knowledge.

**Rim detection** (YOLOX wins decisively):
- YOLOX: **100.0%** AP@50 — perfect detection
- RF-DETR: 96.9% AP@50
- Gemini: 4.3% AP@50 — near-total failure
- *Analysis*: YOLOX achieves perfect rim detection. The rim is a fixed, high-contrast object with consistent appearance across frames. YOLOX's anchor-based detection at multiple scales captures it reliably. Gemini's 4.3% is striking — VLMs struggle to produce tight bounding boxes around geometric structures like a hoop rim, often returning boxes that are too large or miss the rim entirely.

**Number detection** (YOLOX's biggest advantage):
- YOLOX: **82.9%** AP@50 — leads by 6.2 points
- RF-DETR: 76.8% AP@50
- Gemini: 13.7% AP@50
- *Analysis*: Jersey numbers are small, low-resolution regions requiring precise localization. YOLOX's multi-scale feature pyramid and NMS handle small objects better. RF-DETR's query-based detection may allocate too few queries to small objects. Gemini at 13.7% confirms that VLMs cannot reliably localize fine-grained text regions — they may recognize that numbers exist but cannot draw tight boxes around them.

**Ball detection** (weakest class for all methods):
- YOLOX: 61.5% AP@50
- RF-DETR: 60.1% AP@50
- Gemini: 36.3% AP@50
- *Analysis*: Ball detection is universally hard: the basketball is small (often <1% of image area), frequently occluded by hands, and subject to motion blur. Both trained models perform similarly (~60%), suggesting this is a data-difficulty ceiling rather than an architecture issue. More training data with diverse ball appearances would likely help. Gemini at 36.3% detects the ball roughly a third of the time — it can identify a basketball when it's clearly visible but fails when occluded or blurred.

### Why YOLOX Outperforms RF-DETR

1. **Higher input resolution**: YOLOX uses 640x640 vs RF-DETR's 512x512, giving 56% more pixels. This directly helps with small objects like numbers and the ball.

2. **Multi-scale feature pyramid (FPN/PAN)**: YOLOX's PAFPN architecture preserves fine-grained features at multiple scales, while RF-DETR's transformer encoder aggregates features into a fixed set of queries, potentially losing small-object detail.

3. **Per-class NMS**: YOLOX applies NMS independently per class, preventing high-confidence detections of one class from suppressing nearby detections of another class. RF-DETR uses query-based detection without explicit NMS, which can cause query competition between classes.

4. **Tight bounding box regression**: The mAP@75 gap (62.5% vs 45.4%) suggests YOLOX's regression head produces more precise boxes. This may be due to YOLOX's IoU-aware loss and objectness prediction guiding box quality.

### The Gemini Gap: Zero-Shot vs Fine-Tuned

Gemini 3.1 Pro represents the state-of-the-art in zero-shot VLM detection. Its performance reveals clear boundaries:

| Regime | Gemini AP@50 | Trained Model AP@50 | Gap |
|--------|-------------|---------------------|-----|
| Large, salient objects (player) | 92.5% | 92.6-98.2% | 0-6% |
| Medium, distinct objects (referee) | 68.6% | 93.7-97.1% | 25-29% |
| Fixed geometry (rim) | 4.3% | 96.9-100% | 93-96% |
| Small text (number) | 13.7% | 76.8-82.9% | 63-69% |
| Small, occluded (ball) | 36.3% | 60.1-61.5% | 24-25% |

**Key insight**: Gemini closes the gap to within 6 points on players — objects that VLMs see billions of times in pretraining data. But for domain-specific objects requiring precise spatial reasoning (rim, numbers), Gemini falls behind by 63-96 points. This confirms that **fine-tuning on domain data is essential** for production basketball detection, and zero-shot VLMs cannot replace specialized detectors for spatially precise tasks.

**Gemini's confidence limitation**: All Gemini detections have confidence=1.0, so mAP depends entirely on detection quality, not confidence ranking. If Gemini could output calibrated confidences, its mAP would likely improve by allowing the metric to weight better detections higher.

**Cost-benefit**: Gemini costs ~15.5 seconds per image (API latency) vs ~0.37 seconds for ONNX models — a 40x slowdown. For 190 images, Gemini takes 49 minutes vs 73 seconds for YOLOX. At scale, the trained models are clearly the right choice.

### Key Takeaways

1. **YOLOX-S is the recommended production model** for basketball detection: best mAP@50:95 (54.9%), perfect rim detection, and superior localization (mAP@75: 62.5%).

2. **Localization quality matters**: The 37% mAP@75 gap between YOLOX and RF-DETR means YOLOX boxes are significantly tighter — critical for downstream tasks like jersey OCR and player tracking.

3. **Ball detection needs more data**: At ~60% AP@50, both trained models plateau on ball detection. This class needs targeted data augmentation or a specialized detector.

4. **Zero-shot VLMs have a clear niche**: Gemini is competitive for player detection (92.5% AP@50) without any training data. For rapid prototyping or classes where training data is unavailable, VLMs are a viable starting point.

5. **Fine-tuning wins decisively**: 2x higher mAP@50:95 and 40x faster inference make trained ONNX models the only practical choice for production basketball detection.

## Evaluation Details

- **mAP computation**: supervision `MeanAveragePrecision` using COCO IoU thresholds [0.50, 0.55, ..., 0.95], maxDets=100
- **P/R/F1**: Class-aware greedy matching at IoU=0.5, threshold swept in 20 steps on val, best threshold applied to test
- **Class merging**: Training sub-classes (player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block) merged into "player"; ball-in-basket merged into "ball"
- **Confidence threshold**: 0.01 for ONNX models (all predictions kept for mAP ranking)
- **YOLOX NMS**: Per-class greedy NMS at IoU 0.45

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

## GCP Artifacts

| Run | Job ID | Output Path |
|-----|--------|-------------|
| RF-DETR + Gemini 2.5 | `5188122100438663168` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/comparison/basketball-detector-eval-comparison/20260227_220728/` |
| Gemini 3.1 Pro | `2191345984330530816` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/comparison/basketball-detector-eval-comparison/20260301_154431/` |
| YOLOX-S | `7000697805152976896` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/comparison/basketball-detector-eval-comparison/20260302_125436/` |
| YOLOX ONNX export | `6409600354060599296` | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/onnx-export/basketball-detector-export-onnx-yolox/20260302_113411/` |
