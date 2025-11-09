import argparse
import cv2

"""
Minimal script to open a video, let you drag a rectangle with the mouse, and print the region:
--region X Y W H
Press ENTER or SPACE to confirm, ESC to cancel/exit.
"""

def parse_args():
    p = argparse.ArgumentParser(description="Select a region (x,y,w,h) from a video frame")
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--frame", type=int, default=0, help="Frame index to pick (default 0)")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[error] Could not open video: {args.video}")
        return

    # Jump to requested frame
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("[error] Could not read frame")
        return

    # Let user select ROI
    # selectROI returns (x, y, w, h)
    r = cv2.selectROI("select-region", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("select-region")

    x, y, w, h = [int(v) for v in r]
    if w <= 0 or h <= 0:
        print("[info] No region selected.")
        return

    print(f"--region {x} {y} {w} {h}")


if __name__ == "__main__":
    main()
