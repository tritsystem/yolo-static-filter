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
                 vib_shift_px=1.5, vib_response=0.15,
                 conf_keep=0.55, live_thresh=3.0, live_window=16,
                 verifier=None, verify_every=15):
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
        # positive false-positive evidence required before suppressing (fixes "removes valid
        # static objects"): only drop a pinned detection that is BOTH low-confidence AND
        # pixel-frozen. A confidently-detected or micro-moving (living) object is kept.
        self.conf_keep = float(conf_keep)        # max conf at/above which a pinned det is kept
        self.live_thresh = float(live_thresh)    # temporal per-pixel std above which = "alive"
        self.live_window = int(live_window)
        # optional appearance verifier (e.g. ClipAppearanceVerifier): the pattern-based,
        # motion-independent check. .verify(crop_bgr) -> ("person"|"structure"|"unsure", margin)
        self.verifier = verifier
        self.verify_every = int(verify_every)
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
        self._gray = None                   # current grayscale frame (for liveness crops)
        self._bgr = None                    # current colour frame (for appearance crops)
        self._maxconf = {}                  # id -> max detection confidence ever seen
        self._live = {}                     # id -> recent downscaled gray crops (liveness)
        self._appear = {}                   # id -> (frame_checked, verdict) appearance cache

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
        self._gray = gray
        self._bgr = frame_bgr
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

    def _alive(self, tid, xyxy):
        """A living subject is never pixel-frozen (micro-motion, sway, lighting); a rigid
        structure under a fixed camera is. Track temporal per-pixel std of a small gray crop;
        'alive' once it rises above live_thresh. Too little history -> assume alive (keep)."""
        if self._gray is None:
            return True
        h, w = self._gray.shape
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        x1 = max(0, min(x1, w - 2)); x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 2)); y2 = max(y1 + 1, min(y2, h))
        crop = cv2.resize(self._gray[y1:y2, x1:x2], (12, 12)).astype(np.float32)
        buf = self._live.setdefault(tid, [])
        buf.append(crop)
        if len(buf) > self.live_window:
            del buf[0]
        if len(buf) < 6:
            return True                                   # not enough history -> keep
        return float(np.stack(buf).std(axis=0).mean()) > self.live_thresh

    def _is_false_positive(self, tid, xyxy):
        """Positive evidence this pinned detection is a false positive, not a valid static
        object: it must be BOTH low-confidence AND pixel-frozen. Any doubt -> not an FP (keep)."""
        low_conf = self._maxconf.get(tid, 0.0) < self.conf_keep
        frozen = not self._alive(tid, xyxy)
        return low_conf and frozen

    def _appearance(self, tid, xyxy):
        """Pattern-based, motion-independent verdict from the plugged-in verifier (CLIP), cached
        per track and refreshed every verify_every frames. Returns 'person' | 'structure' | None."""
        if self.verifier is None or self._bgr is None:
            return None
        cached = self._appear.get(tid)
        if cached is not None and (self._frame - 1) - cached[0] < self.verify_every:
            return cached[1]
        h, w = self._bgr.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
        verdict, _ = self.verifier.verify(self._bgr[y1:y2, x1:x2])
        verdict = verdict if verdict in ("person", "structure") else None
        self._appear[tid] = ((self._frame - 1), verdict)
        return verdict

    # ---- the gate ----------------------------------------------------------------------
    def keep(self, xyxy, track_id=None, conf=None):
        """True = keep, False = gate out as a static false positive. Call after update().
        conf = this detection's confidence (0-1); lets the gate keep confidently-detected
        valid static objects instead of removing them."""
        if self.vibrating:
            return True        # whole picture shaking -> acknowledge, suppress nothing, freeze state

        if track_id is not None:
            tid = int(track_id)
            if conf is not None:
                self._maxconf[tid] = max(self._maxconf.get(tid, 0.0), float(conf))
            x1, y1, x2, y2 = xyxy
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            eps = max(self.move_eps_px, self.move_eps_frac * diag)
            rec = self._track.get(tid)
            if rec is None:
                self._track[tid] = [cx, cy, self._frame - 1]
                # rule 3: new det in a known static zone, not moving, AND proven FP -> gate
                if (self._frame - 1) > self.warmup and self._in_zone(xyxy) \
                        and self._is_false_positive(tid, xyxy):
                    return False
                return True
            ox, oy, first = rec
            if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 > eps:
                self._moved.add(tid)                      # its OWN centre moved
            if tid in self._moved:
                return True                               # rule 1: ever moved -> keep
            if (self._frame - 1) <= self.warmup:
                return True
            pinned = (self._frame - 1) - first >= self.confirm_frames
            if pinned or self._in_zone(xyxy):             # candidate static...
                app = self._appearance(tid, xyxy)         # pattern-based check (if a verifier is set)
                if app == "person":
                    return True                           # looks like a person -> keep (even a
                                                          # worker stationary since camera start)
                # CLIP can veto a removal but never remove on its own: an occluded still person
                # can look like "pole + sign" to CLIP. Dropping always requires pixel-frozen
                # (a living person is not), plus low confidence or a CLIP "structure" verdict.
                frozen = not self._alive(tid, xyxy)
                low_conf = self._maxconf.get(tid, 0.0) < self.conf_keep
                if frozen and (low_conf or app == "structure"):
                    self._mark_zone(xyxy)
                    return False
            return True                                   # doubt / valid static -> keep

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
        cfs = boxes.conf.cpu().numpy() if boxes.conf is not None else [None] * len(xyxy)
        keep = np.array([self.keep(b, t, c) for b, t, c in zip(xyxy, ids, cfs)], dtype=bool)
        return keep, int((~keep).sum())


# ----------------------------------------------------------------------------------------
def selftest():
    """Planted-case validation including the valid-static FIX. Four objects, all static or
    moving, each with a confidence:
      FP         low-conf, pixel-frozen static  -> must be SUPPRESSED (a real false positive)
      VALID_HI   high-conf, pixel-frozen static -> must be KEPT (confidence = real object)
      VALID_LIVE low-conf, micro-moving static  -> must be KEPT (liveness = real object)
      MOVER      moving                          -> must be KEPT
    Only positive FP evidence (low-conf AND frozen) may drop a detection."""
    H, W = 240, 360
    FP = (30, 30, 80, 90); VHI = (140, 30, 190, 90); VLIVE = (260, 30, 310, 90)
    f = StaticDetectionFilter(confirm_frames=12, warmup=10, history=50)
    rng = np.random.default_rng(1)
    bg = rng.integers(0, 60, (H, W, 3), dtype=np.uint8)    # fixed static texture
    k = {"FP": [], "VHI": [], "VLIVE": [], "MOVER": []}
    N = 130
    for i in range(N):
        frame = bg.copy()
        cv2.rectangle(frame, FP[:2], FP[2:], (230, 230, 230), -1)       # frozen
        cv2.rectangle(frame, VHI[:2], VHI[2:], (210, 210, 210), -1)     # frozen
        live = bg[VLIVE[1]:VLIVE[3], VLIVE[0]:VLIVE[2]].astype(np.int16)
        live = np.clip(live + rng.integers(-18, 19, live.shape), 0, 255).astype(np.uint8)
        frame[VLIVE[1]:VLIVE[3], VLIVE[0]:VLIVE[2]] = live              # micro-moving
        cx = 30 + int((W - 60) * (i / N)); cy = 180
        cv2.circle(frame, (cx, cy), 18, (255, 255, 255), -1)
        mover = (cx - 20, cy - 20, cx + 20, cy + 20)
        f.update(frame)
        r = {"FP": f.keep(FP, 1, conf=0.30), "VHI": f.keep(VHI, 2, conf=0.85),
             "VLIVE": f.keep(VLIVE, 3, conf=0.35), "MOVER": f.keep(mover, 4, conf=0.80)}
        if i > 35:
            for key in k: k[key].append(r[key])
    rate = {key: float(np.mean(v)) for key, v in k.items()}
    print(f"[selftest] FP (low-conf frozen) kept   = {rate['FP']:.2f}  (want ~0.00 suppress)")
    print(f"[selftest] VALID_HI   (high-conf) kept = {rate['VHI']:.2f}  (want ~1.00 KEEP)")
    print(f"[selftest] VALID_LIVE (micro-move) kept= {rate['VLIVE']:.2f}  (want ~1.00 KEEP)")
    print(f"[selftest] MOVER kept                  = {rate['MOVER']:.2f}  (want ~1.00 KEEP)")
    ok = rate["FP"] < 0.15 and rate["VHI"] > 0.9 and rate["VLIVE"] > 0.9 and rate["MOVER"] > 0.9
    print("[selftest]", "PASS" if ok else "FAIL",
          "- suppresses only the real FP; keeps valid static objects (confidence/liveness)")
    return ok


class _MockVerifier:
    """Stand-in for CLIP in the integration self-test: bright crop -> person, mid-grey -> structure."""
    def verify(self, crop_bgr):
        m = float(crop_bgr.mean())
        if m > 185: return "person", 0.5
        if m < 150: return "structure", 0.5
        return "unsure", 0.0


def selftest_appearance():
    """The cold-start case: a worker STATIONARY SINCE FRAME 0 (never moved, low confidence,
    pixel-frozen) must be KEPT once the appearance verifier says 'person' -- while a structure
    in the same situation is dropped. Without the verifier the person would be suppressed, so
    this isolates the pattern-based stage."""
    H, W = 240, 360
    PERSON = (40, 60, 95, 170); STRUCT = (240, 60, 300, 170)   # both static from the first frame
    f = StaticDetectionFilter(confirm_frames=12, warmup=10, history=50,
                              verifier=_MockVerifier(), verify_every=5)
    rng = np.random.default_rng(2)
    bg = rng.integers(0, 60, (H, W, 3), dtype=np.uint8)
    # regression (found on vtest.avi): an OCCLUDED still person whose crop CLIP reads as
    # "structure" (pole + sign) but who is alive -> must be KEPT; CLIP alone may never remove.
    OCCL = (140, 60, 200, 170)
    kp, ks, ko = [], [], []
    for i in range(70):
        frame = bg.copy()
        cv2.rectangle(frame, PERSON[:2], PERSON[2:], (210, 210, 210), -1)   # bright -> "person"
        cv2.rectangle(frame, STRUCT[:2], STRUCT[2:], (120, 120, 120), -1)   # mid   -> "structure"
        occ = np.full((OCCL[3] - OCCL[1], OCCL[2] - OCCL[0], 3), 120, np.int16)
        occ = np.clip(occ + rng.integers(-18, 19, occ.shape), 0, 255).astype(np.uint8)
        frame[OCCL[1]:OCCL[3], OCCL[0]:OCCL[2]] = occ                        # "structure" but alive
        f.update(frame)
        p = f.keep(PERSON, 1, conf=0.30)      # low conf, never moves, frozen
        s = f.keep(STRUCT, 2, conf=0.30)
        o = f.keep(OCCL, 3, conf=0.30)
        if i > 30:
            kp.append(p); ks.append(s); ko.append(o)
    pr, sr, orr = float(np.mean(kp)), float(np.mean(ks)), float(np.mean(ko))
    print(f"[appearance] cold-start PERSON kept  = {pr:.2f}  (want ~1.00 KEEP via 'person')")
    print(f"[appearance] STRUCTURE kept          = {sr:.2f}  (want ~0.00 drop via 'structure')")
    print(f"[appearance] OCCLUDED alive kept     = {orr:.2f}  (want ~1.00 KEEP: CLIP can't remove alone)")
    ok = pr > 0.9 and sr < 0.1 and orr > 0.9
    print("[appearance]", "PASS" if ok else "FAIL",
          "- appearance keeps a worker stationary since camera start; drops the structure")
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
    c = selftest_appearance()
    b = selftest_vibration()
    sys.exit(0 if (a and b and c) else 1)
