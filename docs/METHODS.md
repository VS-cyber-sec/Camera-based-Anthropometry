# 🔬 Methods & Mathematical Reference

This document details all formulas, constants, and scientific references used in the system.

---

## Pinhole Camera Model

All pixel-to-centimetre conversions use:

```
real_cm = (pixel_span × Z_cm) / focal_length_px
```

Where Z_cm is the depth to the measurement plane and focal_length_px
is the calibrated focal length (fx for horizontal, fy for vertical).

---

## Scale Recovery — A4 Paper Reference

```
Z_paper = average(
    fx × 21.0 / paper_width_px,
    fy × 29.7 / paper_height_px
)

Z_body = Z_paper + 15.0 cm
```

The 15 cm offset accounts for the paper being held in front of the body.
The person's body plane (spine) is approximately 15 cm behind the paper plane.

**Detection pipeline:**
1. CLAHE (clipLimit=2.5, tileGrid=8x8)
2. Gaussian blur (7x7)
3. Adaptive Canny (Otsu-based lo/hi thresholds)
4. Contour detection + minAreaRect aspect test
5. A4 portrait ratio = 21.0/29.7 = 0.7071 ± 0.20 tolerance
6. Z smoother: exponential moving average alpha=0.25

---

## Height

```
height_cm = (floor_y - crown_y) × Z_body / fy
```

**Crown detection:**
- Scan segmentation mask top-down
- First row with >= 6 foreground pixels (threshold 0.50)
- Fallback: brow landmarks - 15% of brow-to-heel span

**Floor correction:**
```
floor_y = heel_y + (ANKLE_FLOOR_CM × fy / Z_body)
ANKLE_FLOOR_CM = 3.5 cm
```
The MediaPipe heel landmark sits at the ankle bone, ~3.5 cm above the floor.
Reference: Clauser et al. (1969). AMRL-TR-69-70.

---

## MUAC

Ramanujan second approximation for ellipse perimeter:
```
P = pi × [3(a+b) - sqrt((3a+b)(a+3b))]

a = (arm_width_px / 2) × Z_body / fx    (horizontal semi-axis)
b = a × MUAC_DEPTH_RATIO                 (depth semi-axis)
MUAC_DEPTH_RATIO = 0.80
```

**Measurement site:** MUAC_SITE_FRAC=0.40 from shoulder to elbow.
This matches the clinical acromion-olecranon midpoint site.

**Torso exclusion:** Pixels between shoulder x-coordinates zeroed out
to prevent torso bleeding into arm width measurement.

Reference: Clarys & Marfell-Jones (1986). Ergonomics 29(11).
Reference: Norton & Olds (1996). Anthropometrica, UNSW Press. Depth/width approx 0.80.

---

## Head Circumference

```
HC = pi × [3(a+b) - sqrt((3a+b)(a+3b))]

head_w_cm  = widest_skull_row_px × Z_body / fx
skull_w_cm = head_w_cm - HC_HAIR_CM          (hair correction)
a          = skull_w_cm / 2                   (horizontal semi-axis)
b          = a × HC_DEPTH_RATIO               (depth semi-axis)

HC_DEPTH_RATIO = 1.26    (= 19.5 cm depth / 15.5 cm width)
HC_HAIR_CM     = 1.5     (total hair correction, both sides)
```

**Depth ratio derivation:**
- Adult mean head width (L-R): 15.5 cm
- Adult mean head depth (F-B): 19.5 cm
- HC_DEPTH_RATIO = 19.5 / 15.5 = 1.258 ≈ 1.26
Reference: Farkas (1994). Anthropometry of the Head and Face. Raven Press.

**Adaptive threshold search:**
Tested from 0.50 to 0.75 in steps of 0.05.
Best threshold = one giving the most valid-width skull rows.
Reference: Luijkx & Velders (2021). Medical Image Analysis 72: 102103.
Recommended threshold range: 0.55-0.65.

**Measurement plane:**
Widest row in the upper 60% of the head region (85th percentile).
This corresponds to the parietal boss level — the correct HC measurement plane.

---

## IQR Filter + Median

For each measurement, 20 samples are collected and then:

1. Compute Q1 (25th percentile) and Q3 (75th percentile)
2. IQR = Q3 - Q1
3. Remove values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
4. Take the median of remaining values

This removes outlier frames (posture drift, occlusion spikes)
and gives a robust final estimate.

---

## Ramanujan Approximation Accuracy

The second Ramanujan approximation for ellipse perimeter:

```
P ≈ pi × (a+b) × [1 + 3h / (10 + sqrt(4 - 3h))]
where h = ((a-b)/(a+b))^2
```

Accuracy: better than 0.04% for any ellipse.
Reference: Ramanujan (1914). Q.J. Math 45: 350-372.
