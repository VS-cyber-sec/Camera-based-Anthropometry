"""
calibrate_camera.py
====================
Compute camera intrinsic matrix and distortion coefficients
from a folder of checkerboard calibration images.

Usage:
    python scripts/calibrate_camera.py \
        --images data/calibration_images/ \
        --output data/camera_params.json \
        --pattern 9x6

Output camera_params.json format:
    {
        "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
        "dist_coeff": [k1, k2, p1, p2, k3],
        "image_size": [width, height],
        "rms_error": 0.43,
        "calibration_date": "YYYY-MM-DD"
    }
"""

import cv2
import numpy as np
import json
import os
import glob
import argparse
from datetime import date


def parse_args():
    parser = argparse.ArgumentParser(description="Camera calibration from checkerboard images")
    parser.add_argument("--images",  required=True, help="Directory containing calibration images")
    parser.add_argument("--output",  default="data/camera_params.json", help="Output JSON path")
    parser.add_argument("--pattern", default="9x6", help="Checkerboard inner corners, e.g. 9x6")
    parser.add_argument("--square",  type=float, default=25.0, help="Square size in mm (for scale, not critical)")
    parser.add_argument("--show",    action="store_true", help="Show detected corners as images are processed")
    return parser.parse_args()


def main():
    args = parse_args()

    cols, rows = [int(x) for x in args.pattern.split("x")]
    print(f"\nCheckerboard: {cols} × {rows} inner corners")

    # Prepare object points (0,0,0), (1,0,0), ..., (cols-1, rows-1, 0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= args.square

    obj_points = []   # 3D points in world space
    img_points = []   # 2D points in image plane

    pattern = os.path.join(args.images, "*.jpg")
    files   = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(args.images, "*.png")
        files   = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: No .jpg or .png images found in {args.images}")
        return

    print(f"Found {len(files)} images. Processing...")
    img_size = None
    found = 0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fpath in files:
        img  = cv2.imread(fpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            found += 1
            if args.show:
                cv2.drawChessboardCorners(img, (cols, rows), corners_refined, ret)
                cv2.imshow("Calibration", cv2.resize(img, (960, 540)))
                cv2.waitKey(400)
            print(f"  [OK] {os.path.basename(fpath)}")
        else:
            print(f"  [--] {os.path.basename(fpath)}  (corners not found)")

    if args.show:
        cv2.destroyAllWindows()

    print(f"\nUsing {found} / {len(files)} images for calibration.")

    if found < 6:
        print("ERROR: Need at least 6 valid images. Retake photos and try again.")
        return

    rms, cam_mat, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    print(f"\nRMS reprojection error : {rms:.4f} px")
    print(f"  {'Good' if rms < 1.0 else 'Acceptable' if rms < 1.5 else 'POOR — retake photos'}")
    print(f"\nCamera matrix:")
    print(f"  fx = {cam_mat[0,0]:.2f}   fy = {cam_mat[1,1]:.2f}")
    print(f"  cx = {cam_mat[0,2]:.2f}   cy = {cam_mat[1,2]:.2f}")
    print(f"\nDistortion: {dist_coeff.ravel().tolist()}")

    output = {
        "camera_matrix"   : cam_mat.tolist(),
        "dist_coeff"      : dist_coeff.ravel().tolist(),
        "image_size"      : list(img_size),
        "rms_error"       : round(float(rms), 4),
        "calibration_date": str(date.today()),
        "pattern"         : args.pattern,
        "images_used"     : found,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved -> {args.output}")
    print("\nCalibration complete. Update CAM_PARAMS_PATH in src/anthropometry_v11.py if needed.")


if __name__ == "__main__":
    main()
