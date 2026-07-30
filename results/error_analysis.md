# Error Analysis — Anthropometry v11

## Test Summary

Testing conducted on 3 subjects using v11 (single-posture, chest-paper method).
Camera: Laptop webcam at 1280×720, calibrated (RMS 0.43 px).
Camera position: 110 cm height, 230 cm from subject.

---

## Raw Results

| Subject | Actual Height | Detected | Error | Actual MUAC | Detected | Error | Actual HC | Detected | Error |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Person 1 | 160.0 cm | 160.5 cm | +0.5 cm | 30.0 cm | 29.2 cm | -0.8 cm | 54.5 cm | 45.4 cm | -9.1 cm |
| Person 2 | 173.0 cm | 173.9 cm | +0.9 cm | 33.0 cm | 31.8 cm | -1.2 cm | 56.5 cm | 45.9 cm | -10.6 cm |
| Person 3 | 189.0 cm | 189.2 cm | +0.2 cm | 37.0 cm | 35.6 cm | -1.4 cm | 57.0 cm | 46.1 cm | -10.9 cm |

---

## HEIGHT Analysis

Mean error: +0.5 cm (within ±3-6 cm target) ✅

All three readings were within 1 cm of the actual value.
The v11 ankle-floor correction (+3.5 cm) and BODY_PAPER_OFFSET_CM (15 cm)
were the key improvements over v10.

**Remaining sources of error:**
- Crown detection from segmentation mask can be 0–5 px inaccurate depending on hair
- BODY_PAPER_OFFSET_CM=15 cm is fixed; actual arm extension varies per subject

---

## MUAC Analysis

Mean error: -1.1 cm (within ±2-3 cm target) ✅

Readings were consistently slightly under actual tape measurement.
This is expected: the camera measures the arm at rest (fabric + pose),
while tape measurement is taken on bare skin with slight compression.

**Remaining sources of error:**
- Clothing: even thin fabric adds 1-2 mm per layer
- The arm is not perfectly cylindrical; MUAC_DEPTH_RATIO=0.80 is a population mean
- Slight arm movement during the 20-sample window causes variability

**Recommended improvement:**
Bare-arm measurement (sleeve rolled up) for clinical use.

---

## HC Analysis

Mean error: -10.2 cm (OUTSIDE ±2-4 cm target) ❌

**Root cause — depth axis not observable:**
The frontal camera can only measure the left-right width of the head.
The front-to-back depth (b-axis) is estimated as b = a × 1.26 (from Farkas data).
However, this ratio gives b ≈ 0.62 × 2a = 0.62 × head width.

If the actual head depth is different from the population mean, or if
the mask slightly underestimates skull width, the error compounds.

**Why always UNDER not over:**
1. The segmentation mask boundary is slightly inside the skull edge.
   At 2-3 m distance, each pixel = ~2-3 mm, so 2-3 px error = 4-9 mm per side.
2. The depth estimate b = 1.26a assumes head width = 15.5 cm.
   If the actual width is slightly smaller than the mask shows, b is underestimated.

**Planned fixes:**
1. Side-view capture: photograph the subject from the side to directly
   measure the b-axis from ear to back of skull.
2. HC_DEPTH_RATIO calibration: ask subject to hold two markers at known positions
   on their head to calibrate the ratio per-session.
3. Dedicated head segmentation model instead of full-body MediaPipe mask.

---

## Limitations Summary

| Factor | Impact | Severity |
|---|---|---|
| Paper depth offset (BODY_PAPER_OFFSET_CM) | Height ±2-3 cm | Low |
| Hair volume in mask | HC ±2-5 cm | Medium |
| Depth axis not measurable (frontal only) | HC -8-12 cm | High |
| Clothing thickness | MUAC ±1-2 cm | Low |
| Camera height variation | Height ±1-3 cm | Medium |
| Arm not cylindrical (MUAC_DEPTH_RATIO) | MUAC ±1-2 cm | Low |

---

## Best/Worst Cases

**Best:** Height estimation — 0.2 cm error on Person 3 (189 cm tall).
The tall subject provided a larger pixel span, reducing relative pixel quantisation error.

**Worst:** HC estimation — systematic 10 cm under-estimation across all subjects.
Requires architectural change (side view) to fix, not just constant tuning.
