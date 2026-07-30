"""
capture_calibration_images.py
==============================
Interactive tool to capture calibration images from your webcam.

Usage:
    python scripts/capture_calibration_images.py --output data/calibration_images/

Controls:
    SPACE — save current frame
    Q     — quit
"""

import cv2
import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/calibration_images/",
                        help="Directory to save images")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--target", type=int, default=20, help="Target number of images")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    count = 0
    last_save = 0

    print(f"\nCapturing calibration images to: {args.output}")
    print(f"Target: {args.target} images")
    print(f"  SPACE — save frame   |   Q — quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Instructions overlay
        cv2.putText(display,
                    f"Calibration Capture  [{count}/{args.target}]",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 60), 2)
        cv2.putText(display,
                    "SPACE = save   |   Q = quit",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(display,
                    "Hold checkerboard at different angles, distances, positions",
                    (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)

        if count >= args.target:
            cv2.putText(display,
                        "TARGET REACHED — press Q to finish",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)

        cv2.imshow("Calibration Image Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            now = time.time()
            if now - last_save > 0.5:   # debounce
                fname = os.path.join(args.output, f"calib_{count+1:03d}.jpg")
                cv2.imwrite(fname, frame)
                count += 1
                last_save = now
                print(f"  Saved [{count}]: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {count} images saved to {args.output}")
    print(f"Next step: python scripts/calibrate_camera.py --images {args.output}")


if __name__ == "__main__":
    main()
