"""
test_reference_detection.py
============================
Debug tool for A4 paper reference detection.
Shows live view with all detected contours coloured by their
aspect ratio and the best-match bounding rectangle highlighted.

Usage:
    python scripts/test_reference_detection.py
    python scripts/test_reference_detection.py --image data/samples/sample_01.jpg

Controls:
    Q — quit
"""

import cv2
import numpy as np
import argparse
import json
import os

A4_PORTRAIT_RATIO = 21.0 / 29.7   # 0.7071
RATIO_TOL         = 0.20


def detect_reference_debug(frame, fx, fy):
    """Run reference detection with full debug drawing."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray    = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    otsu_t, _ = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clo = max(8, int(otsu_t * 0.35))
    chi = max(25, int(otsu_t * 0.85))
    edged  = cv2.Canny(blurred, clo, chi)
    edged  = cv2.dilate(edged, np.ones((3, 3)), iterations=2)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

    debug = frame.copy()
    result_text = f"Canny: {clo}/{chi}  |  A4 target ratio: {A4_PORTRAIT_RATIO:.3f} +-{RATIO_TOL}"
    cv2.putText(debug, result_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    best_z = None

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 400:
            break

        rect    = cv2.minAreaRect(cnt)
        box     = cv2.boxPoints(rect)
        rw, rh  = sorted([rect[1][0], rect[1][1]])
        if rw < 5 or rh < 5:
            continue

        aspect    = rw / rh
        ratio_err = abs(aspect - A4_PORTRAIT_RATIO)

        M  = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        sides = len(approx)

        if ratio_err <= RATIO_TOL and 4 <= sides <= 6:
            col = (0, 255, 0)
            z_w = (fx * 21.0) / rw
            z_h = (fy * 29.7) / rh
            z   = (z_w + z_h) / 2.0
            label = f"A4! r={aspect:.3f} Z={z:.0f}cm"
            best_z = z
        elif ratio_err <= RATIO_TOL:
            col   = (0, 255, 200)
            label = f"r={aspect:.3f} ({sides}sides)"
        elif ratio_err <= RATIO_TOL * 1.5:
            col   = (0, 200, 255)
            label = f"r={aspect:.3f}"
        else:
            col   = (0, 0, 180)
            label = f"r={aspect:.3f}"

        cv2.drawContours(debug, [box.astype(int)], -1, col, 2)
        cv2.putText(debug, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

    if best_z:
        cv2.putText(debug, f"DETECTED: Z_paper={best_z:.0f}cm  Z_body={best_z+15:.0f}cm",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)
    else:
        cv2.putText(debug, "NOT DETECTED — show A4 paper to camera",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 255), 2)

    return debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to a single image (omit for live camera)")
    parser.add_argument("--params", default="data/camera_params.json")
    args = parser.parse_args()

    if os.path.exists(args.params):
        with open(args.params) as f:
            d = json.load(f)
        K  = np.array(d["camera_matrix"]).reshape(3, 3)
        fx = K[0, 0]; fy = K[1, 1]
    else:
        print(f"WARNING: {args.params} not found — using estimated focal lengths")
        fx = fy = 900.0

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Cannot read {args.image}")
            return
        debug = detect_reference_debug(frame, fx, fy)
        cv2.imshow("Reference Detection Debug", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Press Q to quit")
        while True:
            ret, frame = cap.read()
            if not ret: break
            debug = detect_reference_debug(frame, fx, fy)
            cv2.imshow("Reference Detection Debug", debug)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
