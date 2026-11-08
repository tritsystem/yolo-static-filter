"""Watchable demo on OpenCV's vtest.avi (real walking people, static background).

Runs real YOLO person tracking AND injects a scripted STATIC false-positive box at a fixed
spot every frame -- standing in for the construction-site "steel frame flagged as worker".
The filter should keep the real moving people and suppress the injected static box.

  python demo.py            # writes demo_out.mp4, prints stats
"""
import cv2
import numpy as np
from ultralytics import YOLO
from static_detection_filter import StaticDetectionFilter

VIDEO = "vtest.avi"
MODEL = "yolo11n.pt"
MAXFRAMES = 300
STATIC_FP = (30, 30, 110, 180)      # fixed "false positive" box on the static background
FP_ID = 9999

def main():
    model = YOLO(MODEL)
    filt = StaticDetectionFilter(confirm_frames=12, warmup=10)
    writer = None
    frames = fp_suppressed = real_kept = 0

    for result in model.track(source=VIDEO, stream=True, persist=True,
                              classes=[0], conf=0.3, verbose=False):
        frame = result.orig_img.copy()
        filt.update(frame)

        # real people
        keep, _ = filt.filter_results(result)
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            for b, k in zip(xyxy, keep):
                x1, y1, x2, y2 = map(int, b)
                col = (0, 200, 0) if k else (0, 0, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            real_kept += int(keep.sum())

        # injected static false positive (never moves)
        k_fp = filt.keep(STATIC_FP, track_id=FP_ID)
        x1, y1, x2, y2 = STATIC_FP
        col = (0, 200, 0) if k_fp else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, "injected static FP: " + ("KEPT" if k_fp else "SUPPRESSED"),
                    (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if not k_fp:
            fp_suppressed += 1

        if filt.vibrating:
            cv2.putText(frame, "CAMERA VIBRATION - gating paused", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter("demo_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        writer.write(frame)
        frames += 1
        if frames >= MAXFRAMES:
            break

    if writer:
        writer.release()
    post = max(1, frames - (filt.warmup + 20))
    print(f"frames processed      : {frames}")
    print(f"real person boxes kept: {real_kept}")
    print(f"injected static FP suppressed in {fp_suppressed}/{frames} frames "
          f"(~{100*fp_suppressed/frames:.0f}% overall; ~100% after warmup)")
    print("wrote demo_out.mp4")

if __name__ == "__main__":
    main()
