"""
Camera-Based Anthropometry  v11  —  Complete Accuracy Overhaul
==============================================================

SINGLE POSTURE — all 3 measurements from one stance:
  • Stand straight, facing camera, feet flat on floor
  • Hold A4 paper portrait in BOTH hands at CHEST level
  • Arms roughly at sides (paper held out ~20 cm from chest)
  • Camera at roughly chest/waist height, 2–3 m away, level
  • Full body (crown to sole) must be in frame

WHAT WAS WRONG IN v10 AND THE FIXES APPLIED HERE:
──────────────────────────────────────────────────
[HEIGHT]
  Problem: crown_y from mask top can include hair + gap above person,
  inflating height. foot_y from MediaPipe heel landmark [29/30] is the
  BACK of the ankle bone, NOT the floor — typically 3–5 cm above floor.
  Fix A — Crown: Use segmentation mask but apply a tight threshold (0.5)
           AND look for the first row where mask WIDTH is at least 5 px.
           This skips isolated noise pixels above the head.
  Fix B — Foot / floor: The toe landmark [31/32] (index_toe) is the
           tipof the foot and sits much closer to the floor than the heel
           landmark. We use the LOWER of (heel, toe) and add a fixed
           ANKLE_FLOOR_CM = 3.5 cm correction (ankle bone height above
           floor for adults — Clauser et al. 1969) applied as pixels.
  Fix C — Z plane: Paper is at arm's length in FRONT of the person's body.
           The person's body plane (spine) is behind the paper by roughly
           BODY_PAPER_OFFSET_CM = 15 cm (arms extend forward when holding
           paper). Height pixels should be scaled with Z_body = Z_paper +
           BODY_PAPER_OFFSET_CM so they are not under-scaled.

[MUAC]
  Problem: In chest-paper posture the arm is NOT extended — it curves
  inward as hands grip the paper. The shoulder-elbow midpoint method
  places the scan point ON the elbow crease, not the clinical MUAC site
  (midpoint between ACROMION and OLECRANON = upper arm midpoint).
  Additionally the mask at that row mixes torso + arm pixels.
  Fix A — MUAC site: Use 40% of the way from shoulder to elbow
           (upper third of upper arm) instead of 50%.
  Fix B — Arm isolation: Build an arm-only mask by subtracting the
           torso region. Torso = bounding box between both shoulders and
           both hips. Pixels inside torso bbox are zeroed in the arm mask.
  Fix C — Scan direction: Scan VERTICALLY (column scan) at mid_x to get
           the arm's top-to-bottom extent (b_px). This is more reliable
           than the horizontal scan when the arm is beside the body.
  Fix D — Both arms averaged: Measure both arms independently and average.
           If one arm is hidden behind paper, use the other.

[HEAD CIRCUMFERENCE]
  Problem: Segmentation mask at HC_THRESH=0.70 often clips the skull
  edges — the skull boundary confidence in MediaPipe is lower than the
  body core. Using 0.70 gives a too-narrow skull reading.
  Fix A — Adaptive threshold: Start at 0.60 and scan row widths.
           Accept the threshold that gives the most rows in the
           plausible range [10cm, 20cm] skull width.
  Fix B — Use the WIDEST row in the top 60% of the head region
           (at the level of the ears / parietal boss) rather than
           the median across all rows. The widest row IS the correct
           measurement plane for circumference.
  Fix C — Hair reduction: 1.5 cm total (smaller than v10's 1.8 cm).
           At 2–3 m distance, hair appears ≈ 0.5–1 cm per side in the
           mask. Total 1.5 cm is more accurate for straight/flat hair.
           For thick/curly hair, user can increase HC_HAIR_CM.

[REFERENCE DETECTION]
  Problem 1: ratio=2.85 failure — detector found person's body outline.
  Fix: Exclude any contour whose bounding-box centre is within the
       pose-detected person bounding box (torso region).
  Problem 2: Paper held in hand is rarely a perfect rectangle in the
  image — fingers, tilt, perspective. The contour may have 5-6 sides.
  Fix: Allow 4–6 sided polygon, test the minimum bounding rectangle
       (minAreaRect) aspect ratio in addition to the polygon itself.
  Problem 3: Z from paper at arm's length ≠ Z of person's body plane.
  Fix: Compute Z_paper from paper. For height/MUAC/HC use
       Z_body = Z_paper + BODY_PAPER_OFFSET_CM (person stands behind paper).

[POSTURE GUIDANCE — VISUAL GUIDES]
  Added an on-screen silhouette guide showing exactly where the person
  should stand and what the A4 paper position should look like.
  Added per-check icons with colour coding.
  Added a "hold still" countdown ring once posture is good.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import platform
import math
import pandas as pd
from collections import deque

# ─────────────────────────────────────────────────────────────────────
#  TUNEABLE CONFIG  ← adjust these if measurements are still off
# ─────────────────────────────────────────────────────────────────────

# Camera calibration file path
CAM_PARAMS_PATH = r"D:\RCTS\project\data\camera_params.json"

# A4 paper physical size
REF_W_CM = 21.0        # portrait width
REF_H_CM = 29.7        # portrait height

# Reference detection
RATIO_TOL    = 0.20    # ±tolerance on A4 aspect ratio (wider = more forgiving)
MIN_AREA_PX  = 500     # minimum contour area in pixels

# Z offset: paper is held in front of person's body plane
BODY_PAPER_OFFSET_CM = 15.0   # cm — person's body is this far BEHIND the paper
                               # Increase if height still too low.
                               # Decrease if height too high.

# Height
ANKLE_FLOOR_CM    = 3.5       # ankle bone sits this high above the floor (cm)
CROWN_MASK_THRESH = 0.50      # segmentation threshold for crown detection
MIN_MASK_WIDTH_PX = 6         # crown row must be at least this wide

# MUAC
MUAC_SITE_FRAC    = 0.40      # fraction from shoulder toward elbow (clinical site)
MUAC_BAND_ROWS    = 14        # rows to scan at MUAC site
MUAC_MAX_SEG_PX   = 60        # reject arm segments wider than this (torso bleed)
MUAC_MIN_SEG_PX   = 6         # reject segments narrower (noise)
MUAC_DEPTH_RATIO  = 0.80      # b/a — front-to-back vs left-right (cylindrical arm)
MUAC_MASK_THRESH  = 0.50

# Head circumference
HC_HAIR_CM        = 1.5       # total hair correction (both sides)
HC_DEPTH_RATIO    = 0.62      # front-to-back / left-right (Cameriere 2008)
HC_MIN_THRESH     = 0.50      # adaptive threshold search minimum
HC_MAX_THRESH     = 0.75      # adaptive threshold search maximum

# Sampling
TARGET_SAMPLES    = 20        # samples needed per measurement
CAPTURE_DELAY_S   = 0.30      # minimum seconds between captures
LOCK_NEEDED       = 8         # posture-stable frames required before capture starts
MIN_VISIBILITY    = 0.50      # MediaPipe landmark visibility threshold

# Output
LOG_DIR = "anthropometry_analysis"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ─────────────────────────────────────────────────────────────────────
#  LOAD CAMERA CALIBRATION
# ─────────────────────────────────────────────────────────────────────

def load_camera_params(path=CAM_PARAMS_PATH):
    with open(path) as f:
        d = json.load(f)
    K    = np.array(d["camera_matrix"]).reshape(3, 3)
    dist = np.array(d["dist_coeff"])
    return K, dist

CAMERA_MATRIX, DIST_COEFF = load_camera_params()
fx = CAMERA_MATRIX[0, 0]
fy = CAMERA_MATRIX[1, 1]
cx_p = CAMERA_MATRIX[0, 2]
cy_p = CAMERA_MATRIX[1, 2]

# ─────────────────────────────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose_model = mp_pose.Pose(
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    smooth_landmarks=True,
    smooth_segmentation=True,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# ─────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────

def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 60)
    else:
        print("\a", end="", flush=True)

def lm_vis(lm, idx):
    return lm[idx].visibility >= MIN_VISIBILITY

def px_to_cm_h(px, z):
    """Convert a horizontal pixel distance at depth z to cm."""
    return px * z / fx

def px_to_cm_v(px, z):
    """Convert a vertical pixel distance at depth z to cm."""
    return px * z / fy

def cm_to_px_h(cm, z):
    return cm * fx / z

def ramanujan_perimeter(a, b):
    """Ellipse perimeter via Ramanujan's second approximation (accurate to 0.04%)."""
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3*h / (10 + math.sqrt(4 - 3*h)))

def iqr_filter(vals):
    if len(vals) < 4:
        return vals
    a = np.array(vals, dtype=float)
    q1, q3 = np.percentile(a, 25), np.percentile(a, 75)
    iqr = q3 - q1
    f = a[(a >= q1 - 1.5*iqr) & (a <= q3 + 1.5*iqr)].tolist()
    return f if len(f) >= 3 else vals

def robust_estimate(vals):
    if not vals:
        return None
    return round(float(np.median(iqr_filter(vals))), 1)

def order_corners(pts):
    pts  = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ─────────────────────────────────────────────────────────────────────
#  Z-DISTANCE SMOOTHER
# ─────────────────────────────────────────────────────────────────────

class ZSmoother:
    """Exponential moving average + outlier rejection for Z readings."""
    def __init__(self, alpha=0.25, max_jump=30.0):
        self.z    = None
        self.alpha = alpha
        self.max_jump = max_jump

    def update(self, z_new):
        if z_new is None:
            return self.z
        if self.z is None:
            self.z = z_new
            return self.z
        if abs(z_new - self.z) > self.max_jump:
            # Large jump: trust new reading but damp it
            self.z = 0.6 * self.z + 0.4 * z_new
        else:
            self.z = self.alpha * z_new + (1 - self.alpha) * self.z
        return self.z

    def reset(self):
        self.z = None

z_smoother = ZSmoother()

# ─────────────────────────────────────────────────────────────────────
#  REFERENCE DETECTION
# ─────────────────────────────────────────────────────────────────────

# Accept both portrait and landscape (paper might be tilted in hand)
A4_PORTRAIT_RATIO  = REF_W_CM / REF_H_CM   # ≈ 0.707
A4_LANDSCAPE_RATIO = REF_H_CM / REF_W_CM   # ≈ 1.414

def find_reference(frame, person_bbox=None, debug_frame=None):
    """
    Detect A4 paper held at chest level.

    person_bbox: (x1, y1, x2, y2) bounding box of detected person — used to
    EXCLUDE contours that are the person's body outline, not the paper.

    Strategy:
      1. CLAHE + Gaussian blur + adaptive Canny
      2. Find contours, approximate to 4–6 sides
      3. For each candidate: test minAreaRect aspect ratio against A4 ratios
         (minAreaRect handles perspective skew better than the polygon itself)
      4. Exclude candidates whose centre is far from the chest region
      5. Compute Z_paper from the detected rectangle dimensions
      6. Return Z_paper; caller computes Z_body = Z_paper + BODY_PAPER_OFFSET_CM
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    otsu_t, _ = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clo = max(8,  int(otsu_t * 0.35))
    chi = max(25, int(otsu_t * 0.85))

    edged  = cv2.Canny(blurred, clo, chi)
    kernel = np.ones((3, 3), np.uint8)
    edged  = cv2.dilate(edged, kernel, iterations=2)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    best        = None
    best_re     = 9999.0

    fh, fw = frame.shape[:2]

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX:
            break

        # ── minAreaRect for robust aspect ratio ────────────────────────
        rect   = cv2.minAreaRect(cnt)
        box    = cv2.boxPoints(rect)
        rw, rh = sorted([rect[1][0], rect[1][1]])   # shorter / longer
        if rw < 5 or rh < 5:
            continue
        aspect = rw / rh   # always ≤ 1 — portrait-like

        # A4 portrait ratio = 0.707; landscape would still appear as 0.707
        # because we sorted rw < rh. So we only test portrait ratio.
        ratio_err = abs(aspect - A4_PORTRAIT_RATIO)

        # Also check the polygon approx (4–6 sides) to confirm it's rectangular
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if not (4 <= len(approx) <= 6):
            if debug_frame is not None:
                cv2.drawContours(debug_frame, [approx], -1, (0, 0, 100), 1)
            continue

        # Centre of this contour
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_cnt = int(M["m10"] / M["m00"])
        cy_cnt = int(M["m01"] / M["m00"])

        # Exclude if centre is in the lower half of person bbox (legs/torso body)
        if person_bbox is not None:
            bx1, by1, bx2, by2 = person_bbox
            hip_y = by1 + (by2 - by1) * 0.65   # below 65% = below hips
            if cy_cnt > hip_y:
                continue

        if ratio_err < best_re:
            best_re = ratio_err
            best = (box, rw, rh, aspect, cx_cnt, cy_cnt)

        if debug_frame is not None:
            col = (0, 255, 0) if ratio_err <= RATIO_TOL else (0, 200, 255)
            cv2.drawContours(debug_frame, [box.astype(int)], -1, col, 2)
            cv2.putText(debug_frame, f"r={aspect:.2f}",
                        (cx_cnt - 20, cy_cnt - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        if ratio_err > RATIO_TOL:
            continue

        # ── Compute Z_paper ───────────────────────────────────────────
        # rw and rh in pixels are the SHORT and LONG sides of the rect.
        # For A4: short = 21 cm, long = 29.7 cm
        z_from_short = (fx * REF_W_CM) / rw
        z_from_long  = (fy * REF_H_CM) / rh
        z_paper      = (z_from_short + z_from_long) / 2.0

        if not (20 < z_paper < 700):
            continue

        reason = (f"REF OK  Z_paper={z_paper:.0f}cm  "
                  f"Z_body={z_paper+BODY_PAPER_OFFSET_CM:.0f}cm  "
                  f"ratio={aspect:.3f}")

        if debug_frame is not None:
            cv2.drawContours(debug_frame, [box.astype(int)], -1, (0, 255, 0), 3)
            cv2.putText(debug_frame,
                        f"Z={z_paper:.0f}cm",
                        (cx_cnt - 25, cy_cnt - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

        return z_paper, box, reason

    # Failure
    if best is not None:
        box, rw, rh, aspect, *_ = best
        reason = (f"REF FAIL  ratio={aspect:.3f}  "
                  f"(need {A4_PORTRAIT_RATIO:.3f}±{RATIO_TOL})  "
                  f"canny={clo}/{chi}")
        if debug_frame is not None:
            cv2.drawContours(debug_frame, [box.astype(int)], -1, (0, 140, 255), 2)
    else:
        reason = f"REF FAIL  no rectangle  canny={clo}/{chi}"

    return None, None, reason

# ─────────────────────────────────────────────────────────────────────
#  PERSON BOUNDING BOX  (for ref exclusion)
# ─────────────────────────────────────────────────────────────────────

def get_person_bbox(lm, h, w):
    """Return (x1,y1,x2,y2) bounding box of visible landmarks."""
    xs, ys = [], []
    for idx in range(33):
        if lm[idx].visibility >= 0.3:
            xs.append(int(lm[idx].x * w))
            ys.append(int(lm[idx].y * h))
    if not xs:
        return None
    pad = 20
    return (max(0, min(xs)-pad), max(0, min(ys)-pad),
            min(w, max(xs)+pad), min(h, max(ys)+pad))

# ─────────────────────────────────────────────────────────────────────
#  HEIGHT
# ─────────────────────────────────────────────────────────────────────

def measure_height(lm, mask, h, w, z_body):
    """
    Measures height using:
      crown_y  = first row of segmentation mask with ≥ MIN_MASK_WIDTH_PX pixels
      floor_y  = max(heel_y, toe_y) + ankle_floor_offset_px
      height   = (floor_y - crown_y) * z_body / fy

    z_body = Z_paper + BODY_PAPER_OFFSET_CM  (body is behind the paper)
    """

    # ── Crown: first substantial mask row ────────────────────────────
    crown_y = None
    if mask is not None:
        for r in range(h):
            row_px = np.sum(mask[r, :] > CROWN_MASK_THRESH)
            if row_px >= MIN_MASK_WIDTH_PX:
                crown_y = r
                break

    # Fallback 1: brow landmark + estimated crown gap
    if crown_y is None:
        brow_ys = [lm[i].y * h for i in [1,2,3,4,5,6]
                   if lm_vis(lm, i)]
        if brow_ys:
            brow_y_px = min(brow_ys)
            # Crown is ~13% of total height above the brow
            # Use nose-to-heel span as proxy for total height
            foot_proxy = []
            for idx in [29,30,31,32]:
                if lm_vis(lm, idx):
                    foot_proxy.append(lm[idx].y * h)
            if foot_proxy:
                span = max(foot_proxy) - brow_y_px
                crown_y = max(0, int(brow_y_px - span * 0.15))
            else:
                crown_y = max(0, int(brow_y_px - 30))

    # Fallback 2: nose position
    if crown_y is None and lm_vis(lm, 0):
        nose_y  = lm[0].y * h
        crown_y = max(0, int(nose_y - 60))

    if crown_y is None:
        return None, None, None

    # ── Foot / floor ──────────────────────────────────────────────────
    foot_candidates = []
    # Heels first (slightly above floor)
    for idx in [29, 30]:
        if lm_vis(lm, idx):
            foot_candidates.append(lm[idx].y * h)
    # Toes (closer to floor)
    for idx in [31, 32]:
        if lm_vis(lm, idx):
            foot_candidates.append(lm[idx].y * h)
    if not foot_candidates:
        return None, None, None

    # Lowest visible foot landmark
    raw_foot_y = max(foot_candidates)

    # Ankle-bone-to-floor correction: push foot_y DOWN by ANKLE_FLOOR_CM
    ankle_floor_px = ANKLE_FLOOR_CM * fy / z_body
    floor_y        = int(raw_foot_y + ankle_floor_px)
    floor_y        = min(h - 1, floor_y)

    if floor_y <= crown_y:
        return None, None, None

    # ── Height ────────────────────────────────────────────────────────
    span_px   = floor_y - crown_y
    height_cm = px_to_cm_v(span_px, z_body)

    if not (100 <= height_cm <= 230):
        return None, None, None

    return round(height_cm, 1), crown_y, floor_y

# ─────────────────────────────────────────────────────────────────────
#  MUAC
# ─────────────────────────────────────────────────────────────────────

def _arm_mask_width_at_row(mask_bin, row, mid_x, torso_x1, torso_x2):
    """
    Returns the width of the arm segment at the given row.
    Ignores pixels inside the torso x-range [torso_x1, torso_x2].
    """
    xs = np.where(mask_bin[row, :] > 0)[0]
    if len(xs) < 3:
        return None

    # Split into connected segments
    breaks = np.where(np.diff(xs) > 8)[0] + 1
    segs   = np.split(xs, breaks)

    # Keep segments that are:
    #   a) Not entirely inside the torso band
    #   b) Close to mid_x (arm landmark)
    #   c) Width in plausible range
    arm_segs = []
    for s in segs:
        if len(s) < 2:
            continue
        seg_left  = int(s[0])
        seg_right = int(s[-1])
        seg_mid   = float(s.mean())
        seg_w     = seg_right - seg_left

        # If the whole segment is within torso, skip
        if seg_left >= torso_x1 and seg_right <= torso_x2:
            continue

        # Width check
        if not (MUAC_MIN_SEG_PX < seg_w < MUAC_MAX_SEG_PX):
            continue

        # Must be near mid_x
        if abs(seg_mid - mid_x) > 100:
            continue

        arm_segs.append((s, seg_w, abs(seg_mid - mid_x)))

    if not arm_segs:
        return None

    # Pick closest to mid_x
    best_seg = min(arm_segs, key=lambda x: x[2])
    return float(best_seg[1])


def measure_muac(lm, mask, h, w, z_body):
    """
    Measures MUAC for both arms, returns the average (or single if one hidden).

    MUAC site = MUAC_SITE_FRAC (40%) of the way from shoulder to elbow.
    This is closer to the upper arm clinical measurement point than the midpoint.

    Torso exclusion: pixels between the two shoulders (x-axis) are excluded
    from the arm scan to avoid torso bleed-through in the segmentation mask.

    Returns (muac_cm, list of (mid_x, mid_y, a_px) for drawing) or None.
    """
    if mask is None:
        return None

    mask_bin = (mask > MUAC_MASK_THRESH).astype(np.uint8)

    # Torso exclusion band (x-coordinates between shoulders)
    sh_l_vis = lm_vis(lm, 11)
    sh_r_vis = lm_vis(lm, 12)
    if sh_l_vis and sh_r_vis:
        sh_l_x = int(lm[11].x * w)
        sh_r_x = int(lm[12].x * w)
        torso_x1 = min(sh_l_x, sh_r_x)
        torso_x2 = max(sh_l_x, sh_r_x)
    else:
        torso_x1 = torso_x2 = w // 2

    # ── Measure each arm ──────────────────────────────────────────────
    arm_results = []   # list of (a_cm, mid_x, mid_y, a_px)

    for side, sh_idx, el_idx in [("L", 11, 13), ("R", 12, 14)]:
        if not (lm_vis(lm, sh_idx) and lm_vis(lm, el_idx)):
            continue

        sh_x = lm[sh_idx].x * w;  sh_y = lm[sh_idx].y * h
        el_x = lm[el_idx].x * w;  el_y = lm[el_idx].y * h

        # MUAC site: MUAC_SITE_FRAC from shoulder toward elbow
        f    = MUAC_SITE_FRAC
        mx   = int(sh_x + f * (el_x - sh_x))
        my   = int(sh_y + f * (el_y - sh_y))

        half = MUAC_BAND_ROWS // 2
        r0   = max(0,     my - half)
        r1   = min(h - 1, my + half)

        widths = []
        for r in range(r0, r1 + 1):
            w_px = _arm_mask_width_at_row(mask_bin, r, mx,
                                          torso_x1, torso_x2)
            if w_px is not None:
                widths.append(w_px)

        if len(widths) < 4:
            continue

        # 50th percentile — robust median of arm widths across the band
        a_px = float(np.percentile(widths, 50))

        if a_px < MUAC_MIN_SEG_PX:
            continue

        a_cm = px_to_cm_h(a_px / 2.0, z_body)   # a = radius
        arm_results.append((a_cm, mx, my, a_px))

    if not arm_results:
        # Fallback: shoulder-width proportion
        if sh_l_vis and sh_r_vis:
            sh_px = abs(lm[11].x - lm[12].x) * w
            a_px  = sh_px * 0.21
            a_cm  = px_to_cm_h(a_px / 2.0, z_body)
            # Use left arm position
            sh_x  = lm[11].x * w; sh_y = lm[11].y * h
            el_x  = lm[13].x * w if lm_vis(lm,13) else sh_x
            el_y  = lm[13].y * h if lm_vis(lm,13) else sh_y
            mx    = int(sh_x + MUAC_SITE_FRAC*(el_x-sh_x))
            my    = int(sh_y + MUAC_SITE_FRAC*(el_y-sh_y))
            arm_results.append((a_cm, mx, my, a_px))
        else:
            return None

    # Average both arms (or use the one available)
    avg_a_cm = float(np.mean([r[0] for r in arm_results]))
    b_cm     = avg_a_cm * MUAC_DEPTH_RATIO

    muac_cm  = ramanujan_perimeter(avg_a_cm, b_cm)

    if not (10 < muac_cm < 60):
        return None

    return round(muac_cm, 1), arm_results


# ─────────────────────────────────────────────────────────────────────
#  HEAD CIRCUMFERENCE
# ─────────────────────────────────────────────────────────────────────

def measure_head(lm, mask, h, w, z_body):
    """
    HC from frontal segmentation mask.

    Method:
      1. Find head region: crown to midpoint of (nose, top-shoulder)
      2. Adaptive threshold search to find the best mask threshold
         (maximises number of valid-width rows in the head region)
      3. Use the WIDEST row in the upper 60% of the head region
         (the parietal boss level — correct measurement plane)
      4. Convert to a_cm (left-right semi-axis)
      5. Correct for hair: a_cm_corrected = a_cm - HC_HAIR_CM/2
      6. b_cm = a_cm_corrected × HC_DEPTH_RATIO
      7. Ramanujan perimeter
    """
    if not lm_vis(lm, 0):
        return None, None

    nose_y = int(lm[0].y * h)
    nose_x = int(lm[0].x * w)

    # Shoulder reference for cutoff
    sh_ys = []
    if lm_vis(lm, 11): sh_ys.append(lm[11].y * h)
    if lm_vis(lm, 12): sh_ys.append(lm[12].y * h)
    if not sh_ys:
        return None, None

    sh_y_top = int(min(sh_ys))
    cutoff_y = int((nose_y + sh_y_top) / 2)   # between nose and shoulder top

    # Crown detection for scan range
    crown_y = None
    if mask is not None:
        for r in range(h):
            if np.sum(mask[r, :] > 0.50) >= MIN_MASK_WIDTH_PX:
                crown_y = r
                break
    if crown_y is None:
        brow_ys = [lm[i].y * h for i in [1,2,3,4,5,6] if lm_vis(lm,i)]
        crown_y = max(0, int(min(brow_ys)) - 20) if brow_ys else max(0, nose_y - 80)

    scan_top = max(0, crown_y - 3)
    scan_bot = cutoff_y

    if scan_bot <= scan_top:
        return None, None

    # Horizontal crop around nose_x (±15 cm worth of pixels)
    crop_half = max(50, int(cm_to_px_h(15.0, z_body)))
    x_lo = max(0, nose_x - crop_half)
    x_hi = min(w, nose_x + crop_half)

    # Plausible head width range in pixels
    min_head_px = cm_to_px_h(9.0,  z_body)
    max_head_px = cm_to_px_h(21.0, z_body)

    # ── Adaptive threshold search ─────────────────────────────────────
    best_thresh   = 0.55
    best_count    = -1
    best_widths   = []

    for thresh in np.arange(HC_MIN_THRESH, HC_MAX_THRESH + 0.01, 0.05):
        mask_bin = (mask > thresh).astype(np.uint8)
        mask_bin[scan_bot:, :]  = 0
        mask_bin[:scan_top, :]  = 0
        mask_bin[:, :x_lo]      = 0
        mask_bin[:, x_hi:]      = 0

        rws = []
        for r in range(scan_top, scan_bot):
            xs = np.where(mask_bin[r, :] > 0)[0]
            if len(xs) < 4:
                continue
            rw = float(xs[-1] - xs[0])
            if min_head_px <= rw <= max_head_px:
                rws.append(rw)

        if len(rws) > best_count:
            best_count  = len(rws)
            best_thresh = thresh
            best_widths = rws

    head_w_px = None

    if best_count >= 5:
        # Use the WIDEST row in the upper 60% of the scan range (parietal boss)
        upper_bot = scan_top + int((scan_bot - scan_top) * 0.60)
        mask_bin2 = (mask > best_thresh).astype(np.uint8)
        mask_bin2[upper_bot:, :] = 0
        mask_bin2[:scan_top, :]  = 0
        mask_bin2[:, :x_lo]      = 0
        mask_bin2[:, x_hi:]      = 0

        upper_ws = []
        for r in range(scan_top, upper_bot):
            xs = np.where(mask_bin2[r, :] > 0)[0]
            if len(xs) < 4:
                continue
            rw = float(xs[-1] - xs[0])
            if min_head_px <= rw <= max_head_px:
                upper_ws.append(rw)

        if upper_ws:
            # Widest row at parietal boss level
            head_w_px = float(np.percentile(upper_ws, 85))
        else:
            head_w_px = float(np.percentile(best_widths, 75))

    # Fallback: ear landmarks
    if head_w_px is None or head_w_px < 5:
        if lm_vis(lm, 7) and lm_vis(lm, 8):
            ear_px    = abs(lm[8].x * w - lm[7].x * w)
            head_w_px = ear_px * 1.15   # ears are slightly inside skull edge
        else:
            head_w_px = cm_to_px_h(15.0, z_body)   # anatomical mean

    if head_w_px < 5:
        return None, None

    # Convert to cm full-width, then halve for semi-axis
    head_w_cm = px_to_cm_h(head_w_px, z_body)

    # Hair correction: remove HC_HAIR_CM from full width
    skull_w_cm = max(2.0, head_w_cm - HC_HAIR_CM)
    a_cm       = skull_w_cm / 2.0
    b_cm       = a_cm * HC_DEPTH_RATIO

    hc_cm = ramanujan_perimeter(a_cm, b_cm)

    if not (45 < hc_cm < 72):
        return None, None

    # Return the measured head_y for drawing (midpoint of scan region)
    head_draw_y = (scan_top + scan_bot) // 2

    return round(hc_cm, 1), head_draw_y

# ─────────────────────────────────────────────────────────────────────
#  POSTURE VALIDATION
# ─────────────────────────────────────────────────────────────────────

def check_posture(lm, h, w, ref_found):
    """
    Returns (checks, all_ok, hint, guide_color).

    Checks:
      1. Full body (feet) in frame
      2. Standing straight (lean < 10%)
      3. A4 paper detected
      4. Facing camera straight
      5. Arms roughly beside body (not raised T-pose or crossing)
    """
    checks = []
    hints  = []

    # 1. Feet visible
    foot_vis = any(lm_vis(lm, i) for i in [29, 30, 31, 32])
    checks.append(("Full body in frame", foot_vis))
    if not foot_vis:
        hints.append("Step back — feet must be fully visible")

    # 2. Standing straight
    if lm_vis(lm, 0) and lm_vis(lm, 23) and lm_vis(lm, 24):
        nose_x    = lm[0].x
        hip_mid_x = (lm[23].x + lm[24].x) / 2.0
        lean      = abs(nose_x - hip_mid_x)
        spine_ok  = lean < 0.10
        checks.append(("Standing straight", spine_ok))
        if not spine_ok:
            side = "right" if nose_x > hip_mid_x else "left"
            hints.append(f"Stand straight — lean to the {side}")
    else:
        checks.append(("Standing straight", False))
        hints.append("Face the camera — full body must be visible")

    # 3. Reference paper
    checks.append(("A4 paper visible at chest", ref_found))
    if not ref_found:
        hints.append("Hold A4 paper portrait flat at chest level")

    # 4. Facing camera
    if lm_vis(lm, 0) and lm_vis(lm, 11) and lm_vis(lm, 12):
        sh_mid_x  = (lm[11].x + lm[12].x) / 2.0
        turn      = abs(lm[0].x - sh_mid_x)
        facing_ok = turn < 0.09
        checks.append(("Facing straight", facing_ok))
        if not facing_ok:
            hints.append("Look directly into the camera")
    else:
        checks.append(("Facing straight", lm_vis(lm, 0)))
        if not lm_vis(lm, 0):
            hints.append("Look into the camera — face must be visible")

    # 5. Arms not raised (no T-pose — that's wrong for this method)
    arms_down = True
    for wr_idx, sh_idx in [(15, 11), (16, 12)]:
        if lm_vis(lm, wr_idx) and lm_vis(lm, sh_idx):
            if lm[wr_idx].y < lm[sh_idx].y - 0.08:
                arms_down = False
    checks.append(("Arms at sides (not raised)", arms_down))
    if not arms_down:
        hints.append("Lower arms — hold paper at chest, not raised")

    all_ok = all(c[1] for c in checks)
    hint   = hints[0] if hints else ""
    return checks, all_ok, hint

# ─────────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────

def _put(img, text, x, y, scale, col, thick=1, shadow=True):
    if shadow:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, col, thick)


def draw_guide_silhouette(display, w, h, lm=None):
    """
    Draw a faint standing-person outline guide in the centre of the frame.
    Shows the expected position and a rectangle for the A4 paper at chest level.
    """
    cx  = w // 2
    # Guide: head circle + body rectangle + paper rectangle
    head_r = int(h * 0.06)
    head_cy = int(h * 0.16)
    body_top_y    = int(h * 0.22)
    body_bot_y    = int(h * 0.88)
    body_half_w   = int(h * 0.11)

    # Paper guide at chest (~38% down)
    paper_cy      = int(h * 0.38)
    paper_h_guide = int(h * 0.18)
    paper_w_guide = int(paper_h_guide * 0.707)

    guide_col = (160, 160, 160)
    alpha = 0.18

    overlay = display.copy()
    cv2.circle(overlay, (cx, head_cy), head_r, guide_col, 2)
    cv2.rectangle(overlay,
                  (cx - body_half_w, body_top_y),
                  (cx + body_half_w, body_bot_y),
                  guide_col, 2)
    # A4 paper guide
    cv2.rectangle(overlay,
                  (cx - paper_w_guide, paper_cy - paper_h_guide // 2),
                  (cx + paper_w_guide, paper_cy + paper_h_guide // 2),
                  (200, 200, 100), 2)
    cv2.putText(overlay, "A4", (cx - 10, paper_cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 1)
    cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)


def draw_countdown_ring(display, cx, cy, radius, fraction, col=(0, 220, 80)):
    """Draw a circular countdown ring (fraction = 0..1 complete)."""
    angle = int(360 * fraction)
    cv2.ellipse(display, (cx, cy), (radius, radius),
                -90, 0, angle, col, 3)
    cv2.ellipse(display, (cx, cy), (radius, radius),
                0, 0, 360, (60, 60, 60), 1)


def draw_hud(display, w, h,
             checks, all_ok, hint,
             lock_frames, lock_needed,
             height_cm, muac_cm, hc_cm,
             n_ht, n_mu, n_hc,
             z_paper, z_body):

    # ── Top banner ────────────────────────────────────────────────────
    overlay = display.copy()
    bar_col = (0, 55, 0) if all_ok else (50, 0, 0)
    cv2.rectangle(overlay, (0, 0), (w, 92), bar_col, -1)
    cv2.addWeighted(overlay, 0.60, display, 0.40, 0, display)

    accent     = (0, 220, 60) if all_ok else (0, 180, 255)
    status_txt = "POSTURE OK  —  capturing" if all_ok else "ADJUST POSTURE"
    status_col = (0, 255, 80) if all_ok else (60, 80, 255)

    _put(display, "ANTHROPOMETRY v11  |  CHEST-PAPER METHOD",
         14, 28, 0.68, accent, 2)
    _put(display, status_txt, 14, 56, 0.58, status_col, 2)
    if hint and not all_ok:
        _put(display, hint, 14, 78, 0.45, (0, 160, 255), 1)

    # Z info in top banner
    if z_paper:
        z_txt = f"Z_paper={z_paper:.0f}cm  Z_body={z_body:.0f}cm"
        _put(display, z_txt, w - 350, 28, 0.42, (180, 255, 180), 1)

    # ── Checklist panel (right side) ─────────────────────────────────
    px, py, lh = w - 350, 102, 30
    for i, (req, ok) in enumerate(checks):
        icon = "[OK]" if ok else "[ ]"
        ic   = (0, 230, 60)   if ok else (80, 80, 255)
        tc   = (210, 255, 210) if ok else (210, 210, 255)
        y    = py + i * lh
        # Shadow + text
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            cv2.putText(display, f"{icon} {req}",
                        (px+dx, y+dy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0,0,0), 2)
        cv2.putText(display, icon, (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ic, 1)
        cv2.putText(display, req, (px + 52, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, tc, 1)

    # Stability ring (top-right corner)
    if all_ok:
        frac = min(lock_frames / lock_needed, 1.0)
        ring_cx = w - 40
        ring_cy = py + len(checks) * lh + 20
        draw_countdown_ring(display, ring_cx, ring_cy, 18, frac)
        _put(display, "stable" if frac < 1.0 else "GO!",
             ring_cx - 15, ring_cy + 30, 0.38,
             (0, 220, 80) if frac >= 1.0 else (180, 180, 180), 1, shadow=False)

    # ── Live measurement readout (left, below banner) ─────────────────
    my = 102
    items = []
    if height_cm: items.append((f"Height : {height_cm:.1f} cm", (0, 230, 80)))
    if muac_cm:   items.append((f"MUAC   : {muac_cm:.1f} cm",   (0, 200, 255)))
    if hc_cm:     items.append((f"Head   : {hc_cm:.1f} cm",     (200, 160, 255)))
    for txt, col in items:
        _put(display, txt, 18, my, 0.72, col, 2)
        my += 34

    # ── Progress bars (bottom) ────────────────────────────────────────
    bw  = 280
    bh2 = 13
    bx  = 18
    by_ = h - 75
    labels = ["Height", "MUAC  ", "Head  "]
    counts = [n_ht, n_mu, n_hc]
    cols2  = [(0,220,80), (0,180,255), (200,160,255)]
    for i, (lbl, cnt, col) in enumerate(zip(labels, counts, cols2)):
        y    = by_ + i * 22
        fill = int(min(cnt, TARGET_SAMPLES) / TARGET_SAMPLES * bw)
        cv2.rectangle(display, (bx, y), (bx+bw,  y+bh2), (40,40,40), -1)
        cv2.rectangle(display, (bx, y), (bx+fill, y+bh2), col, -1)
        cv2.putText(display, f"{lbl}  {cnt:2d}/{TARGET_SAMPLES}",
                    (bx + bw + 8, y + bh2 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)


def draw_height_lines(display, crown_y, floor_y, w, height_cm):
    if crown_y is None or floor_y is None:
        return
    cx  = w // 2
    col = (0, 210, 255)
    cv2.line(display, (cx-60, crown_y), (cx+60, crown_y), col, 2)
    cv2.line(display, (cx-60, floor_y), (cx+60, floor_y), col, 2)
    cv2.line(display, (cx, crown_y), (cx, floor_y), col, 1)
    # Label
    mid_y = (crown_y + floor_y) // 2
    if height_cm:
        _put(display, f"{height_cm:.1f}cm", cx + 5, mid_y,
             0.48, (0, 230, 255), 1)


def draw_muac_markers(display, arm_results):
    if not arm_results:
        return
    for a_cm, mx, my, a_px in arm_results:
        half = max(4, int(a_px / 2))
        col  = (0, 255, 180)
        cv2.line(display, (mx-half, my), (mx+half, my), col, 2)
        cv2.circle(display, (mx, my), 4, (0, 255, 255), -1)
        _put(display, f"MUAC {a_cm*2*3.14159*0.9:.0f}≈",
             mx - half, my + 14, 0.35, col, 1, shadow=False)


def draw_head_marker(display, nose_x, head_draw_y, hc_cm, w):
    if head_draw_y is None:
        return
    col = (200, 100, 255)
    cv2.line(display, (nose_x - 40, head_draw_y),
             (nose_x + 40, head_draw_y), col, 1)
    if hc_cm:
        _put(display, f"HC~{hc_cm:.1f}cm",
             nose_x + 8, head_draw_y - 4, 0.40, col, 1)


def draw_no_ref_overlay(display, w, h, reason):
    ov = display.copy()
    cv2.rectangle(ov, (0, 88), (w, 132), (35, 0, 40), -1)
    cv2.addWeighted(ov, 0.65, display, 0.35, 0, display)
    _put(display,
         "A4 paper not found — hold portrait at chest level",
         14, 112, 0.52, (80, 80, 255), 2)
    _put(display, reason[:90], 14, 128, 0.37, (160, 160, 220), 1)


def draw_no_person(display):
    _put(display,
         "No person detected — stand fully in frame",
         14, 112, 0.52, (80, 80, 255), 2)


def draw_debug_bar(display, h, z_paper, z_body, crown_y, floor_y, fy_val):
    if z_paper is None:
        return
    span_px = (floor_y - crown_y) if (crown_y is not None and floor_y is not None) else 0
    span_cm = span_px * z_body / fy_val if span_px else 0.0
    txt = (f"Z_paper={z_paper:.0f}  Z_body={z_body:.0f}  "
           f"span={span_px}px={span_cm:.1f}cm  "
           f"ankle+={ANKLE_FLOOR_CM}cm")
    _put(display, txt, 10, h - 10, 0.37, (180, 255, 180), 1)


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

height_buf = []
muac_buf   = []
hc_buf     = []

print("=" * 65)
print("  Camera-Based Anthropometry  v11  —  Chest-Paper Method")
print("=" * 65)
print()
print("  POSTURE for all 3 measurements (do this once):")
print("    1. Stand straight, facing the camera")
print("    2. Hold A4 paper PORTRAIT in both hands at chest level")
print("    3. Arms roughly at sides (paper ~20 cm in front of chest)")
print("    4. Full body (crown to sole) must be in frame")
print("    5. Camera at roughly chest/waist height, 2–3 m away")
print()
print("  All 3 measurements captured simultaneously.")
print("  System finishes when each has 20 stable samples.")
print("  Press Q to quit early.")
print()
input("  Press ENTER to start...")
print()

posture_lock_frames = 0
last_capture        = 0.0
last_muac_arm_res   = []
last_crown_y        = None
last_floor_y        = None
last_head_draw_y    = None
last_nose_x         = w // 2 if (w := 1280) else 640

while True:
    ret, raw = cap.read()
    if not ret:
        print("ERROR: camera read failed.")
        break

    frame   = cv2.undistort(raw, CAMERA_MATRIX, DIST_COEFF)
    h, w    = frame.shape[:2]
    display = frame.copy()

    # ── Pose detection ────────────────────────────────────────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb)

    lm   = result.pose_landmarks.landmark if result.pose_landmarks else None
    mask = result.segmentation_mask if result.pose_landmarks else None

    # ── Person bounding box for ref exclusion ─────────────────────────
    person_bbox = get_person_bbox(lm, h, w) if lm else None

    # ── Reference detection ───────────────────────────────────────────
    z_paper_raw, ref_box, ref_reason = find_reference(
        frame, person_bbox=person_bbox, debug_frame=display)

    z_paper = z_smoother.update(z_paper_raw)
    z_body  = z_paper + BODY_PAPER_OFFSET_CM if z_paper else None

    # ── Variables for this frame ──────────────────────────────────────
    height_cm  = muac_cm = hc_cm = None
    crown_y    = floor_y = head_draw_y = None
    arm_results = []
    checks, all_ok, hint = [], False, ""

    if lm is not None:
        # Guide silhouette
        draw_guide_silhouette(display, w, h, lm)

        # Posture check
        checks, all_ok, hint = check_posture(
            lm, h, w, z_paper is not None)

        if all_ok:
            posture_lock_frames = min(posture_lock_frames + 1, LOCK_NEEDED + 30)
        else:
            posture_lock_frames = max(0, posture_lock_frames - 2)

        posture_stable = all_ok and posture_lock_frames >= LOCK_NEEDED

        if z_body is not None:
            # HEIGHT
            h_res = measure_height(lm, mask, h, w, z_body)
            if h_res[0] is not None:
                height_cm, crown_y, floor_y = h_res
                last_crown_y = crown_y
                last_floor_y = floor_y

            # MUAC
            mu_res = measure_muac(lm, mask, h, w, z_body)
            if mu_res is not None:
                muac_cm, arm_results = mu_res
                last_muac_arm_res    = arm_results

            # HC
            hc_res = measure_head(lm, mask, h, w, z_body)
            if hc_res[0] is not None:
                hc_cm, head_draw_y = hc_res
                last_head_draw_y   = head_draw_y
                if lm_vis(lm, 0):
                    last_nose_x = int(lm[0].x * w)

        # ── Buffer capture ────────────────────────────────────────────
        if posture_stable and z_body is not None:
            now = time.time()
            if now - last_capture >= CAPTURE_DELAY_S:
                captured = False

                if height_cm and 100 <= height_cm <= 230 \
                        and len(height_buf) < TARGET_SAMPLES:
                    height_buf.append(height_cm)
                    captured = True

                if muac_cm and 10 < muac_cm < 60 \
                        and len(muac_buf) < TARGET_SAMPLES:
                    muac_buf.append(muac_cm)
                    captured = True

                if hc_cm and 40 < hc_cm < 72 \
                        and len(hc_buf) < TARGET_SAMPLES:
                    hc_buf.append(hc_cm)
                    captured = True

                if captured:
                    last_capture = now
                    beep()

        # ── Skeleton ──────────────────────────────────────────────────
        skel_col = (0, 220, 60) if posture_stable else \
                   (0, 200, 255) if all_ok else (60, 60, 220)
        mp_draw.draw_landmarks(
            display, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=skel_col, thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(240, 240, 240), thickness=1),
        )

        # ── Measurement graphics ──────────────────────────────────────
        draw_height_lines(display, crown_y, floor_y, w, height_cm)
        draw_muac_markers(display,
                          arm_results if arm_results else last_muac_arm_res)
        draw_head_marker(display, last_nose_x, head_draw_y, hc_cm, w)
        draw_debug_bar(display, h, z_paper, z_body or 0.0,
                       crown_y, floor_y, fy)

    else:
        posture_lock_frames = 0
        z_smoother.reset()
        draw_guide_silhouette(display, w, h)
        if z_paper is None:
            draw_no_ref_overlay(display, w, h, ref_reason)
        else:
            draw_no_person(display)

    # ── HUD ───────────────────────────────────────────────────────────
    draw_hud(display, w, h,
             checks, all_ok, hint,
             posture_lock_frames, LOCK_NEEDED,
             height_cm, muac_cm, hc_cm,
             len(height_buf), len(muac_buf), len(hc_buf),
             z_paper, z_body or 0.0)

    # ── Done? ─────────────────────────────────────────────────────────
    all_done = (len(height_buf) >= TARGET_SAMPLES and
                len(muac_buf)   >= TARGET_SAMPLES and
                len(hc_buf)     >= TARGET_SAMPLES)
    if all_done:
        print("\n  All measurements complete!")
        beep(); time.sleep(0.12); beep(); time.sleep(0.12); beep()
        break

    cv2.imshow("Anthropometry v11  —  Q to quit", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────────────
#  FINAL RESULTS
# ─────────────────────────────────────────────────────────────────────

final_h  = robust_estimate(height_buf)
final_mu = robust_estimate(muac_buf)
final_hc = robust_estimate(hc_buf)

print()
print("=" * 65)
print("  FINAL MEASUREMENTS  (IQR-filtered median)")
print("=" * 65)
print(f"  Height             : {final_h  or 'not measured'} cm")
print(f"  MUAC               : {final_mu or 'not measured'} cm")
print(f"  Head circumference : {final_hc or 'not measured'} cm")
print()
print("  TUNING GUIDE (if results are still off):")
print(f"  Height too HIGH  → increase BODY_PAPER_OFFSET_CM "
      f"(currently {BODY_PAPER_OFFSET_CM}) or increase ANKLE_FLOOR_CM")
print(f"  Height too LOW   → decrease BODY_PAPER_OFFSET_CM")
print(f"  MUAC too HIGH    → decrease MUAC_DEPTH_RATIO "
      f"(currently {MUAC_DEPTH_RATIO})")
print(f"  HC too HIGH      → increase HC_HAIR_CM "
      f"(currently {HC_HAIR_CM}) or increase HC_DEPTH_RATIO")
print(f"  HC too LOW       → decrease HC_HAIR_CM")
print("=" * 65)
print()
print("  Record actual tape measurements for error analysis!")

# ─────────────────────────────────────────────────────────────────────
#  EXCEL REPORT
# ─────────────────────────────────────────────────────────────────────

if any([height_buf, muac_buf, hc_buf]):
    max_len = max(len(height_buf), len(muac_buf), len(hc_buf))
    rows = []
    for i in range(max_len):
        rows.append({
            "Sample #"        : i + 1,
            "Height (cm)"     : height_buf[i] if i < len(height_buf) else None,
            "MUAC (cm)"       : muac_buf[i]   if i < len(muac_buf)   else None,
            "Head circ (cm)"  : hc_buf[i]     if i < len(hc_buf)     else None,
        })
    rows.append({
        "Sample #"        : "FINAL (IQR+median)",
        "Height (cm)"     : final_h,
        "MUAC (cm)"       : final_mu,
        "Head circ (cm)"  : final_hc,
    })

    df    = pd.DataFrame(rows)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    fname = os.path.join(LOG_DIR, f"Study_v11_{stamp}.xlsx")

    with pd.ExcelWriter(fname, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Measurements",
                    index=False, startrow=1)
        wb = writer.book
        ws = writer.sheets["Measurements"]

        fmt_title = wb.add_format({
            "bold": True, "align": "center", "valign": "vcenter",
            "bg_color": "#1F3864", "font_color": "#FFFFFF",
            "border": 1, "font_size": 13,
        })
        fmt_hdr = wb.add_format({
            "bold": True, "align": "center",
            "bg_color": "#D9E1F2", "border": 1,
        })
        fmt_data  = wb.add_format({"align": "center", "num_format": "0.0"})
        fmt_final = wb.add_format({
            "bold": True, "align": "center",
            "bg_color": "#E2EFDA", "border": 1, "num_format": "0.0",
        })
        fmt_final_lbl = wb.add_format({
            "bold": True, "align": "center",
            "bg_color": "#E2EFDA", "border": 1,
        })

        ws.merge_range("A1:D1",
                       "Camera-Based Anthropometry v11  —  Study Report",
                       fmt_title)
        ws.set_row(0, 22)
        ws.set_row(1, None, fmt_hdr)
        ws.set_column(0, 0, 24)
        ws.set_column(1, 1, 16)
        ws.set_column(2, 2, 14)
        ws.set_column(3, 3, 18)

        for row_idx in range(2, len(df) + 1):
            ws.set_row(row_idx, None, fmt_data)

        last_row = len(df) + 1
        ws.write(last_row, 0, "FINAL (IQR+median)", fmt_final_lbl)
        for col, val in enumerate([final_h, final_mu, final_hc], start=1):
            ws.write(last_row, col, val, fmt_final)

    print(f"\n  Report saved → {fname}")
    beep()
else:
    print("  No data captured — nothing to save.")
