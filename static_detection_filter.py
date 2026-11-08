"""Static-detection GATE for Ultralytics YOLO video inference.

Problem (construction-site safety demo, M. R. Munawar): YOLO flags permanent static scene
structures -- a bolted steel frame, a block -- as "worker" every frame. A real worker, even
when briefly still, moved at some point (they walked into frame).

This is a GATE, not a late post-filter: it learns where static false positives live and
rejects detections there at the source, before they are accepted as boxes.

How the gate decides, per detection (keyed on the detection's OWN motion, so a person
walking THROUGH a static structure's box does not fool it):

  1. If the detection's track centre has ever moved > eps from where it first appeared ->
     KEEP forever (a real mover; protects workers who pass through a static zone).
  2. If the track has stayed pinned for `confirm_frames` -> SUPPRESS and record its footprint
     as a learned STATIC ZONE.
  3. A brand-new detection landing inside an already-learned static zone, showing no motion
     yet, is GATED OUT immediately -- even if YOLO gave it a fresh track id (flicker).
  4. Otherwise keep provisionally (a genuinely new detection gets `confirm_frames` to prove
     it moves before being judged static).

No-track fallback (bare predict, no ids): a box over a region with no foreground motion for
more than `static_patience` frames is suppressed (MOG2 background subtractor).

Limits: does not recover MISSED detections (recall). A worker who appears already pinned
inside a learned static zone and never moves is the documented edge case -- use model.track.

Dependencies: opencv-python, numpy. No YOLO dependency here, so it is unit-testable; see
selftest().
"""
from __future__ import annotations
import numpy as np
import cv2

CELL = 16          # static-zone grid cell size (px)


class StaticDetectionFilter:
    def __init__(self, confirm_frames=12, warmup=10, static_patience=60,
                 move_eps_px=10.0, move_eps_frac=0.25, zone_hits=8, zone_cover=0.6,
                 motion_thresh=0.02, history=500, var_threshold=16, global_motion_cap=0.5,
                 vib_shift_px=1.5, vib_response=0.15):
        """
        confirm_frames   frames a track may stay pinned before it is judged static/suppressed
        warmup           frames to observe before the gate starts suppressing
        move_eps_px      a track has "moved" once its box centre displaces more than
        move_eps_frac    max(move_eps_px, move_eps_frac * box_diagonal) from its first seen
                         position (keyed on the detection's own centre -> passer-by immune)
        zone_hits        grid-cell hit count at which a cell counts as a learned static zone
        zone_cover       fraction of a box's cells that must be in learned static zones for
                         the proactive gate (rule 3) to reject a brand-new detection
        static_patience  no-track fallback only: motionless frames before a box is dropped
        motion_thresh / history / var_threshold / global_motion_cap   MOG2 + camera guard
        """
        self.confirm_frames = int(confirm_frames)
        self.warmup = int(warmup)
        self.static_patience = int(static_patience)
        self.move_eps_px = float(move_eps_px)
        self.move_eps_frac = float(move_eps_frac)
        self.zone_hits = int(zone_hits)
        self.zone_cover = float(zone_cover)
        self.motion_thresh = float(motion_thresh)
        self.global_motion_cap = float(global_motion_cap)
        self.vib_shift_px = float(vib_shift_px)
        self.vib_response = float(vib_response)
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self._frame = 0
        self._fg = None
        self._last_motion = None            # no-track fallback: last frame each px moved
        self._global_motion = 0.0
        self._moved = set()                 # track ids whose centre has moved -> keep
        self._track = {}                    # id -> [origin_cx, origin_cy, first_frame]
        self._zone = None                   # learned static-zone hit grid (float)
        self._prev_gray = None              # vibration detector: previous grayscale frame
        self._shift = 0.0                   # last global frame-to-frame shift magnitude (px)
        self.vibrating = False              # True when the whole picture is shaking

    # ---- per-frame update (call once, before keep/gate) --------------------------------
    def update(self, frame_bgr, learning_rate=-1):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # whole-picture vibration: sub-pixel global shift vs the previous frame
        gf = np.float32(gray)
        resp = 0.0
        if self._prev_gray is not None:
            (dx, dy), resp = cv2.phaseCorrelate(self._prev_gray, gf)
            self._shift = float((dx * dx + dy * dy) ** 0.5)
        else:
            self._shift = 0.0
        self._prev_gray = gf

        mask = self._bg.apply(frame_bgr, learningRate=learning_rate)
        fg = mask > 0
        h, w = fg.shape
        if self._last_motion is None:
            self._last_motion = np.full((h, w), -10 ** 9, dtype=np.int64)
            self._zone = np.zeros(((h + CELL - 1) // CELL, (w + CELL - 1) // CELL), np.float32)
        self._global_motion = float(fg.mean())
        # vibrating if the whole frame coherently shifted (high-confidence phase-corr peak),
        # or a huge share of the frame is foreground (gross global change)
        self.vibrating = ((self._shift > self.vib_shift_px and resp > self.vib_response)
                          or self._global_motion > self.global_motion_cap)
        if not self.vibrating:
            self._last_motion[fg] = self._frame     # don't trust motion while shaking
        self._fg = fg
        self._frame += 1
        return mask

    # ---- helpers -----------------------------------------------------------------------
    def _cells(self, xyxy):
        gh, gw = self._zone.shape
        x1, y1, x2, y2 = xyxy
        cx1 = max(0, int(x1) // CELL); cy1 = max(0, int(y1) // CELL)
        cx2 = min(gw, int(x2) // CELL + 1); cy2 = min(gh, int(y2) // CELL + 1)
        return cx1, cy1, cx2, cy2

    def _mark_zone(self, xyxy):
        cx1, cy1, cx2, cy2 = self._cells(xyxy)
        if cx2 > cx1 and cy2 > cy1:
            np.minimum(self._zone[cy1:cy2, cx1:cx2] + 1.0, 4 * self.zone_hits,
                       out=self._zone[cy1:cy2, cx1:cx2])

    def _in_zone(self, xyxy):
        cx1, cy1, cx2, cy2 = self._cells(xyxy)
        if cx2 <= cx1 or cy2 <= cy1:
            return False
        sub = self._zone[cy1:cy2, cx1:cx2]
        return float((sub >= self.zone_hits).mean()) >= self.zone_cover

    # ---- the gate ----------------------------------------------------------------------
    def keep(self, xyxy, track_id=None):
        """True = keep, False = gate out as a static false positive. Call after update()."""
        if self.vibrating:
            return True        # whole picture shaking -> acknowledge, suppress nothing, freeze state

        if track_id is not None:
            tid = int(track_id)
            x1, y1, x2, y2 = xyxy
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            eps = max(self.move_eps_px, self.move_eps_frac * diag)
            rec = self._track.get(tid)
            if rec is None:
                self._track[tid] = [cx, cy, self._frame - 1]
                # rule 3: new detection dropped into a known static zone, not moving -> gate
                if (self._frame - 1) > self.warmup and self._in_zone(xyxy):
                    return False
                return True
            ox, oy, first = rec
            if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 > eps:
                self._moved.add(tid)                      # its OWN centre moved
            if tid in self._moved:
                return True                               # rule 1: ever moved -> keep
            if (self._frame - 1) <= self.warmup:
                return True
            life = (self._frame - 1) - first
            if life >= self.confirm_frames:               # rule 2: pinned -> static
                self._mark_zone(xyxy)
                return False
            if self._in_zone(xyxy):                       # rule 3: inside a learned zone
                return False
            return True                                   # rule 4: provisional keep

        # no track id: spatial foreground fallback
        if self._fg is None or self._frame <= self.warmup:
            return True
        h, w = self._fg.shape
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return False
        if float(self._fg[y1:y2, x1:x2].mean()) >= self.motion_thresh:
            return True
        frames_since = (self._frame - 1) - int(self._last_motion[y1:y2, x1:x2].max())
        return frames_since <= self.static_patience

    def filter_results(self, result):
        """Keep-mask aligned to result.boxes (Ultralytics Results for one frame), plus the
        drop count. update() must already have run on this frame."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros(0, dtype=bool), 0
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        keep = np.array([self.keep(b, t) for b, t in zip(xyxy, ids)], dtype=bool)
        return keep, int((~keep).sum())


# ----------------------------------------------------------------------------------------
def selftest():
    """Planted-case validation: a STATIC box (false positive) next to a MOVING blob (real
    worker). The gate must suppress the static box and keep the moving one."""
    H, W = 240, 320
    STATIC_BOX = (40, 40, 90, 90)
    f = StaticDetectionFilter(confirm_frames=12, warmup=10, history=50)
    rng = np.random.default_rng(1)
    bg = rng.integers(0, 60, (H, W, 3), dtype=np.uint8)   # fixed static texture (like a real scene)
    ks, km = [], []
    N = 120
    for i in range(N):
        frame = bg.copy()
        cv2.rectangle(frame, STATIC_BOX[:2], STATIC_BOX[2:], (255, 255, 255), -1)
        cx = 30 + int((W - 60) * (i / N)); cy = 170
        cv2.circle(frame, (cx, cy), 18, (255, 255, 255), -1)
        moving_box = (cx - 20, cy - 20, cx + 20, cy + 20)
        f.update(frame)
        s = f.keep(STATIC_BOX, track_id=1)
        m = f.keep(moving_box, track_id=2)
        if i > 30:
            ks.append(s); km.append(m)
    sr, mr = float(np.mean(ks)), float(np.mean(km))
    print(f"[selftest] static false-box kept rate = {sr:.2f}  (want ~0.00)")
    print(f"[selftest] moving worker  kept rate = {mr:.2f}  (want ~1.00)")
    ok = sr < 0.1 and mr > 0.9
    print("[selftest]", "PASS" if ok else "FAIL")
    return ok


def selftest_vibration():
    """The gate must ACKNOWLEDGE whole-picture vibration: when the frame jitters, flag
    `vibrating` and suppress nothing (a shaking static box must not be gated as static)."""
    H, W = 240, 320
    f = StaticDetectionFilter(confirm_frames=12, warmup=10, history=50)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 60, (H, W, 3), dtype=np.uint8)   # textured scene (features for phase-corr)
    cv2.rectangle(base, (120, 90), (180, 160), (255, 255, 255), -1)
    for _ in range(40):                                   # settle: static camera
        f.update(base); f.keep((120, 90, 180, 160), track_id=1)
    flagged, kept_during_shake = [], []
    for _ in range(30):                                   # now shake the whole picture
        dx, dy = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shaken = cv2.warpAffine(base, M, (W, H))
        f.update(shaken)
        flagged.append(f.vibrating)
        kept_during_shake.append(f.keep((120 + dx, 90 + dy, 180 + dx, 160 + dy), track_id=1))
    vib_rate = float(np.mean(flagged)); keep_rate = float(np.mean(kept_during_shake))
    print(f"[vibration] frames flagged vibrating = {vib_rate:.2f}  (want high)")
    print(f"[vibration] detections kept during shake = {keep_rate:.2f}  (want 1.00)")
    ok = vib_rate > 0.7 and keep_rate > 0.99
    print("[vibration]", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    a = selftest()
    b = selftest_vibration()
    sys.exit(0 if (a and b) else 1)
