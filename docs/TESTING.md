# 🧪 Testing Protocol

This document describes how to run accuracy tests and interpret the results.

---

## Hardware Setup

```
Camera placement:
  - Camera at CHEST/WAIST height (roughly 100-120 cm above floor)
  - 2.0 to 3.0 metres from subject
  - Camera level (not tilted up or down)
  - Background: plain light-coloured wall (minimises detection noise)
  - Lighting: uniform, avoid strong directional shadows

Subject:
  - Stand on a flat, level floor
  - Wear form-fitting clothing (no thick coats)
  - Hair: tied back or flat for HC measurement
  - A4 paper: hold portrait, flat against chest, both hands
  - Full body must be in frame (crown to sole)
```

---

## Ground Truth Measurements

Before running the system, measure each subject with standard tools:

| Measurement | Tool | Method |
|---|---|---|
| Height | Stadiometer or wall + tape | Barefoot, standing straight, Frankfurt plane |
| MUAC | Flexible tape measure | Left arm, midpoint between acromion and olecranon |
| HC | Flexible tape measure | Widest horizontal circumference above the ears |

Record these values in `results/accuracy_sheet.xlsx` (template provided).

---

## Running a Test Session

```bash
# Single subject session
python src/anthropometry_v11.py

# Batch test on saved frames (no camera required)
python scripts/batch_accuracy_test.py \
    --frames data/samples/ \
    --ground-truth data/ground_truth.csv \
    --output results/batch_results.xlsx
```

---

## Interpreting Results

The system prints a tuning guide after each session:

```
Height too HIGH  -> increase BODY_PAPER_OFFSET_CM  (currently 15.0)
Height too LOW   -> decrease BODY_PAPER_OFFSET_CM
MUAC too HIGH    -> decrease MUAC_DEPTH_RATIO       (currently 0.80)
HC too HIGH      -> increase HC_HAIR_CM             (currently 1.5)
HC too LOW       -> decrease HC_HAIR_CM
```

### Typical Error Sources

| Symptom | Most Likely Cause |
|---|---|
| Height 5+ cm too low | Camera too high, or BODY_PAPER_OFFSET_CM too small |
| Height 5+ cm too high | ANKLE_FLOOR_CM too large, or camera too low |
| MUAC consistently high | Thick sleeves, or torso bleed into arm scan |
| MUAC varies widely | Arm moving during capture; wait for stability ring |
| HC 8-12 cm too low | HC_DEPTH_RATIO needs increase, or head not fully in frame |
| HC varies widely | Hair occlusion; tie back hair before session |

---

## Accuracy Targets (from project brief)

| Measurement | Target Error Range |
|---|---|
| Height | ±3–6 cm |
| MUAC | ±2–3 cm |
| HC | ±2–4 cm |

---

## Test Results (v11, 3 subjects)

| Subject | Act. Height | Det. | Err. | Act. MUAC | Det. | Err. | Act. HC | Det. | Err. |
|---|---|---|---|---|---|---|---|---|---|
| P1 (160 cm) | 160.0 | 160.5 | +0.5 | 30.0 | 29.2 | -0.8 | 54.5 | 45.4 | -9.1 |
| P2 (173 cm) | 173.0 | 173.9 | +0.9 | 33.0 | 31.8 | -1.2 | 56.5 | 45.9 | -10.6 |
| P3 (189 cm) | 189.0 | 189.2 | +0.2 | 37.0 | 35.6 | -1.4 | 57.0 | 46.1 | -10.9 |

**Height:** ✅ Mean error 0.5 cm — well within ±3-6 cm target.
**MUAC:** ✅ Mean error -1.1 cm — within ±2-3 cm target.
**HC:** ❌ Mean error -10.2 cm — outside ±2-4 cm target.

HC systematic under-estimation is a known limitation. The frontal-only depth
estimation (HC_DEPTH_RATIO=1.26) is insufficient. A side-view capture to measure
the b-axis directly would resolve this. See METHODS.md for details.
