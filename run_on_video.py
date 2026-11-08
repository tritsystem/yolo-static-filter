"""Run Ultralytics YOLO on a video with the static-detection filter applied.

Suppresses permanently-static false-positive boxes (e.g. a steel frame flagged as a
worker) while keeping real people, using per-track motion memory.

  python run_on_video.py --source site.mp4 --model yolo11n.pt --classes 0 --save out.mp4

--classes 0 keeps only the person class (the worker-safety case). Drop it to keep all.
Kept boxes are drawn green; suppressed ones thin red (use --hide-suppressed to omit).
"""
import argparse
import cv2
from ultralytics import YOLO
from static_detection_filter import StaticDetectionFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path, or 0 for webcam")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="class ids to keep, e.g. 0 for person; default all")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", default=None, help="output mp4 path")
    ap.add_argument("--view", action="store_true", help="show a live window")
    ap.add_argument("--hide-suppressed", action="store_true")
    # filter tuning
    ap.add_argument("--static-patience", type=int, default=60)
    ap.add_argument("--motion-thresh", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    src = 0 if args.source == "0" else args.source
    model = YOLO(args.model)
    filt = StaticDetectionFilter(motion_thresh=args.motion_thresh,
                                 static_patience=args.static_patience, warmup=args.warmup)

    writer = None
    total_suppressed = 0
    frames = 0
    stream = model.track(source=src, stream=True, persist=True,
                         classes=args.classes, conf=args.conf, verbose=False)
    for result in stream:
        frame = result.orig_img.copy()
        filt.update(frame)
        keep, dropped = filt.filter_results(result)
        total_suppressed += dropped
        frames += 1

        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            ids = (boxes.id.cpu().numpy().astype(int)
                   if boxes.id is not None else [None] * len(xyxy))
            for b, keep_it, tid in zip(xyxy, keep, ids):
                x1, y1, x2, y2 = map(int, b)
                if keep_it:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame, f"id{tid}" if tid is not None else "keep",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                elif not args.hide_suppressed:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)
                    cv2.putText(frame, "static", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

        cv2.putText(frame, f"suppressed this frame: {dropped}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if filt.vibrating:
            cv2.putText(frame, "CAMERA VIBRATION - gating paused", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         30, (w, h))
            writer.write(frame)
        if args.view:
            cv2.imshow("yolo + static filter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    if args.view:
        cv2.destroyAllWindows()
    print(f"frames={frames}  total detections suppressed as static={total_suppressed}")
    if args.save:
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
