# 📷 Camera Calibration Guide

Accurate calibration is the single most important step for measurement accuracy.
Without calibration, the system falls back to approximate focal lengths and
**all measurements will have higher error**.

---

## What Calibration Does

Camera calibration computes:

| Parameter | Symbol | Purpose |
|---|---|---|
| Horizontal focal length | `fx` | Converts horizontal pixel spans to real-world cm |
| Vertical focal length | `fy` | Converts vertical pixel spans to real-world cm |
| Principal point X | `cx` | Optical centre offset X |
| Principal point Y | `cy` | Optical centre offset Y |
| Distortion coefficients | `k1,k2,p1,p2,k3` | Corrects lens barrel/pincushion distortion |

The system calls `cv2.undistort()` every frame using these values.

---

## Step 1 — Print the Checkerboard

Print `docs/checkerboard_9x6.pdf` on A4 paper. Glue flat on rigid cardboard.

> ⚠️ The board **must be flat**. A curved board produces wrong calibration.

---

## Step 2 — Take Calibration Photos

```bash
python scripts/capture_calibration_images.py --output data/calibration_images/
```
Press **SPACE** to save a frame, **Q** to quit.

Take **15–20 photos** covering:
- Different tilt angles (±30°)
- Different rotation angles
- Different distances
- All four corners of the frame

---

## Step 3 — Run Calibration

```bash
python scripts/calibrate_camera.py \
    --images data/calibration_images/ \
    --output data/camera_params.json \
    --pattern 9x6
```

**Expected output:**
```
Found 18 / 20 valid calibration images
RMS reprojection error: 0.43 px     <- Good if < 1.0
fx = 934.2  fy = 935.7
cx = 634.8  cy = 361.2
Saved -> data/camera_params.json
```

> RMS error < 1.0 px is good. If > 1.5, retake the photos.

---

## camera_params.json Format

```json
{
  "camera_matrix": [
    [934.2,   0.0, 634.8],
    [  0.0, 935.7, 361.2],
    [  0.0,   0.0,   1.0]
  ],
  "dist_coeff": [0.042, -0.118, 0.001, -0.002, 0.091],
  "image_size": [1280, 720],
  "rms_error": 0.43,
  "calibration_date": "2026-01-15"
}
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| RMS > 1.5 px | Board not flat, too few images | Re-print flat, add more photos |
| Very few images detected | Checkerboard not fully in frame | Move closer, improve lighting |
| Height consistently off | Calibration at wrong resolution | Re-calibrate at 1280x720 |
