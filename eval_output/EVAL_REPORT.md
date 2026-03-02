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

## Key Findings

1. **YOLOX-S is the best overall model**: 54.9% mAP@50:95 on test — 9% higher than RF-DETR (50.3%). Strongest at rim detection (100% AP@50) and number detection (82.9% AP@50).

2. **YOLOX-S excels at precise localization**: 62.5% mAP@75 vs RF-DETR's 45.4% — a 37% relative improvement. YOLOX bounding boxes are tighter, which matters for downstream tasks.

3. **RF-DETR leads on player detection**: 98.2% AP@50 vs YOLOX's 92.6%. RF-DETR's transformer attention may help with crowded player scenes.

4. **Both trained models far outperform zero-shot Gemini**: ~2x higher mAP@50:95 across the board. Fine-tuning on domain data remains essential.

5. **Gemini 3.1 Pro is surprisingly competent for players**: 92.5% AP@50 — only 6 points behind RF-DETR. Zero-shot VLMs can detect large, salient objects well.

6. **Gemini struggles with small/specific objects**: rim (4.3% AP@50), number (13.7%), and ball (36.3%) are far behind trained models. Precise localization remains a VLM weakness.

7. **ONNX inference is ~40x faster than Gemini API**: Both ONNX models process ~2.6 img/s end-to-end vs Gemini's 0.064 img/s.

8. **All models struggle with ball detection**: Ball AP@50 is the weakest class across all methods (61-70% for trained models, 36% for Gemini), likely due to small size, motion blur, and occlusion.

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
