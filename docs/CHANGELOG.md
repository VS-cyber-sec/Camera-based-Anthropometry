# Changelog

All notable changes to the Camera-Based Anthropometry System.

---

## [v11] — 2026-04-07 (Current)

### Architecture
- **Single-posture**: all 3 measurements in one stance (hold A4 paper at chest)
- Removed 3-phase architecture (height → MUAC → HC separately)

### Reference Detection
- Added `minAreaRect` aspect ratio test (handles perspective skew better)
- Added person bounding-box exclusion to reject body-outline false positives
- Added Z exponential moving average smoother (alpha=0.25)
- Wider ratio tolerance: ±0.20 (was ±0.15)

### Height
- Fixed crown detection: scan for first row with ≥6 mask pixels (was first any-pixel row)
- Added ankle-floor correction: +3.5 cm to heel landmark
- Added Z_body = Z_paper + BODY_PAPER_OFFSET_CM (15 cm) correction

### MUAC
- Moved measurement site from 50% to 40% shoulder→elbow (clinical site)
- Added torso x-band exclusion to prevent torso bleeding into arm scan
- Both arms measured and averaged
- MUAC_DEPTH_RATIO: 0.80 (was 0.75)

### Head Circumference
- Adaptive threshold search [0.50–0.75] instead of fixed 0.70
- Use widest row at parietal boss level (upper 60%, 85th percentile)
- HC_HAIR_CM: 1.5 cm (was 1.8 cm)
- HC_DEPTH_RATIO: 1.26 (derived from Farkas anthropometric data)

### UX
- Added guide silhouette overlay showing expected person position and paper location
- Added countdown ring (circular progress) for posture stability
- Added "arms at sides" posture check (rejects T-pose)
- Tuning guide printed to console after session

---

## [v10] — 2026-03-15

### Architecture
- First single-posture attempt (previously 3-phase)
- Accepts both A4 portrait (0.707) and landscape (1.414) ratios
- Crown from segmentation mask top row (first version)

### Known Issues Fixed in v11
- ratio=2.851 false-positive detection (body outline detected as paper)
- Crown picking up floating noise pixels above head
- MUAC scanning at wrong site (50%, should be 40%)
- HC threshold too high (0.70) — clips skull edges

---

## [v9] — 2026-02-20

### Architecture
- 3-phase system: Phase 1 (height), Phase 2 (MUAC, requires T-pose), Phase 3 (HC)
- 3-second countdown between phases
- 5-frame posture lock before capture

### Height
- Nose-to-height ratio method: height_cm = nose_height_cm / 0.862
- WHO (2006) + Pheasant (1996) population constant
- Brow-to-crown offset: 2.6% of height

### MUAC
- Required T-pose (arms horizontal at 90°) — separate phase
- Both horizontal (a) and vertical (b) axes measured from mask directly

### Head Circumference
- Fixed threshold 0.60 (Luijkx & Velders 2021 recommendation)
- Ear-landmark fallback

---

## [v8] — 2026-01-20

- First A4 paper reference detection
- Canny edge + polygon aspect ratio test
- Basic height-only measurement
- No segmentation mask usage
