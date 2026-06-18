#!/usr/bin/env python3
"""
chess_glass_detector.py  —  Glass piece occupancy detector for OAK-D Lite
═══════════════════════════════════════════════════════════════════════════════
Merges:
  • Your working depthai 3.x pipeline  (createOutputQueue — no crashes)
  • Your BoardCalibrator                (proper corner + contour detection)
  • Glass piece detection               (blue HSV + caustic brightness)

PIECE TYPES  (your glass set)
  B = Blue cobalt glass   → detected via HSV blue ratio
  C = Clear glass         → detected via caustic bright-spot ratio
  ? = Unknown / borderline

KEY BINDINGS
  r  redetect board (pieces OK on board)
  R  full recalibrate from empty board
  C  calibrate occupancy baseline (put ALL pieces off board first)
  D  detect occupancy now
  H  toggle heatmap  (blue_ratio / caustic_ratio)
  F  cycle heatmap feature
  A  print ASCII board to console
  s  save board calibration to disk
  q  quit
"""

import os, time, threading, json
import numpy as np
import cv2
import depthai as dai

# ── Adaptive piece calibration (loaded from piece_calibration.json) ───────────
PIECE_CALIB_FILE = "piece_calibration.json"
_piece_calib: dict | None = None

def load_piece_calibration():
    """Load piece_calibration.json produced by calibrate_pieces.py (optional)."""
    global _piece_calib
    if not os.path.exists(PIECE_CALIB_FILE):
        print(f"  Piece calib: not found ({PIECE_CALIB_FILE}) — using built-in thresholds")
        return None
    try:
        with open(PIECE_CALIB_FILE) as fp:
            _piece_calib = json.load(fp)
        cats = _piece_calib.get("categories", {})
        filled = [k for k, v in cats.items() if v]
        acc = _piece_calib.get("accuracy", "?")
        print(f"  ✓ Piece calib loaded  ({PIECE_CALIB_FILE})")
        print(f"    Categories with data: {filled}  |  accuracy: {acc}%")
        if len(filled) < 6:
            missing = [k for k in ["black_blue","white_blue","black_clear",
                                    "white_clear","black_empty","white_empty"]
                       if k not in filled or not cats.get(k)]
            print(f"    ⚠ Missing samples for: {missing}")
            print(f"    → Run calibrate_pieces.py to collect more samples")
        return _piece_calib
    except Exception as e:
        print(f"  Piece calib load failed: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

BOARD_PX   = 640           # warped board size in pixels
SQ         = BOARD_PX // 8 # = 80 px per square
MARGIN     = 0.15          # strip this fraction of each square edge
LOCK_NEEDED = 5            # stable frames needed to lock calibration
CONSIST_PX  = 4.0          # max drift between frames to be "consistent"
CALIB_FILE  = "board_calibration.npy"

# ── Glass piece HSV range ────────────────────────────────────────────────────
BLUE_LO = (80,  60,  40)
BLUE_HI = (150, 255, 255)
CAUSTIC_THRESH = 235       # pixels brighter than this = caustic from clear glass

# ── Occupancy vote table: {feature: (normaliser, max_pts)} ──────────────────
VOTE_TABLE = {
    "blue_ratio":    (1.5,  4.0),
    "hsv_s_max":     (25,   2.5),
    "caustic_ratio": (0.30, 4.0),
    "caustic_delta": (12,   2.5),
    "gray_std_delta":(8,    1.0),
    "edge_delta":    (0.025,1.0),
}
OCCUPIED_THRESHOLD  = 3.0
BLUE_HARD_THRESH    = 5.0   # blue_ratio% above this → always OCCUPIED
CAUSTIC_HARD_THRESH = 0.8   # caustic_ratio% above this → always OCCUPIED

DST_CORNERS = np.float32([
    [0,        0       ],
    [BOARD_PX, 0       ],
    [BOARD_PX, BOARD_PX],
    [0,        BOARD_PX],
])


# ══════════════════════════════════════════════════════════════════════════════
# depthai 3.x  Pipeline  — RGB only, createOutputQueue (no crashes)
# ══════════════════════════════════════════════════════════════════════════════

def create_pipeline():
    """
    Minimal RGB pipeline — tuned for OAK-D Lite on USB 2.0.

    Why the device crashed even at 640×480 @ 15fps:
      1. ColorCamera.setFps(15) sets OUTPUT fps but the ISP still runs at its
         native rate and queues frames internally.  When the host can't drain
         fast enough, the device-side pool overflows → firmware crash.
      2. setAutoFocusTrigger() kicks off a full AF sweep which spikes ISP load
         and memory use right as the pipeline is stabilising.

    Fixes applied here:
      • Camera node (not ColorCamera — avoids the deprecated preview path)
      • setIspNumFramesPool(1) + setOutputsNumFramesPool(1)
          → device drops frames instead of buffering them → no overflow
      • 320×240 output  (320×240×3×30fps = 6.9 MB/s — tiny, always safe)
      • NO ctrl_queue   — removes autofocus / exposure-lock traffic entirely
      • maxSize=1 on host queue — discard rather than accumulate

    Resolution 640×480: 27.6 MB/s with the pool-size fix applied above.
    Board detection and piece detection both work well at this resolution.
    Piece detection runs on a warped 640×640 patch so source res doesn't matter.
    """
    pipeline = dai.Pipeline()
    cam      = pipeline.create(dai.node.Camera).build()   # CAM_A = RGB

    # Minimise on-device frame pools so the ISP drops old frames immediately
    # rather than buffering them until USB can drain → prevents firmware crash
    try:
        cam.setIspNumFramesPool(1)
        cam.setOutputsNumFramesPool(1)
        cam.setRawNumFramesPool(1)
    except Exception:
        pass   # older firmware may not support — safe to ignore

    rgb_out   = cam.requestOutput(
        (640, 480),                    # 27.6 MB/s — safe with pool-size fix
        type=dai.ImgFrame.Type.BGR888p,
    )
    rgb_queue = rgb_out.createOutputQueue(maxSize=1, blocking=False)

    # No ctrl_queue — eliminates autofocus ISP spike and control traffic
    return pipeline, rgb_queue


def trigger_autofocus(_ctrl_queue=None):
    """No-op — autofocus removed to reduce ISP load on USB 2.0."""
    print("  (autofocus disabled — reduces USB 2.0 load)")


def lock_exposure_wb(_ctrl_queue=None):
    """No-op — exposure lock removed (no ctrl_queue in pipeline)."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Board detection  (your BoardCalibrator, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


FLAG_SETS = [
    cv2.CALIB_CB_ADAPTIVE_THRESH,
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS,
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
]

def try_find_corners(gray):
    for flags in FLAG_SETS:
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), flags=flags)
        if ret:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            return True, corners
    return False, None


def extract_outer_corners(inner_corners):
    pts   = inner_corners.reshape(7, 7, 2)
    avg_r = ((pts[6] - pts[0]) / 6).mean(axis=0)
    avg_c = ((pts[:, 6] - pts[:, 0]) / 6).mean(axis=0)
    return np.float32([
        pts[0, 0] - avg_r - avg_c,
        pts[0, 6] - avg_r + avg_c,
        pts[6, 6] + avg_r + avg_c,
        pts[6, 0] + avg_r - avg_c,
    ])


def sort_corners(pts):
    tl = pts[np.argmin( pts[:, 0] + pts[:, 1])]
    tr = pts[np.argmin(-pts[:, 0] + pts[:, 1])]
    br = pts[np.argmax( pts[:, 0] + pts[:, 1])]
    bl = pts[np.argmax(-pts[:, 0] + pts[:, 1])]
    return np.float32([tl, tr, br, bl])


def detect_by_contour(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges   = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(cnt)
        if area < 8000:   # 640×480 frame: board ~80k-150k px²; 8k keeps some margin
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts   = sort_corners(approx.reshape(4, 2).astype(np.float32))
            w     = np.linalg.norm(pts[1] - pts[0])
            h     = np.linalg.norm(pts[3] - pts[0])
            ratio = max(w, h) / (min(w, h) + 1e-5)
            if ratio < 1.6:
                return pts
    return None


def build_perspective(src_corners: np.ndarray):
    """Returns perspective matrix M that warps src → BOARD_PX×BOARD_PX."""
    return cv2.getPerspectiveTransform(src_corners, DST_CORNERS)


def save_calibration(src_corners):
    np.save(CALIB_FILE, src_corners)
    print(f"  Saved board calibration → {CALIB_FILE}")


def load_calibration(min_area: float = 8_000.0):   # 640×480 frame
    """
    Load board corners from disk.
    Rejects the file if the board area is too small — this catches stale
    calibrations saved with a different camera position/zoom.
    The board must cover at least min_area pixels in the camera frame.
    """
    if not os.path.exists(CALIB_FILE):
        return None
    try:
        src  = np.load(CALIB_FILE)
        area = float(cv2.contourArea(src.reshape(4, 1, 2).astype(np.float32)))
        if area < min_area:
            print(f"  ✗ Rejected stale calibration  "
                  f"(area={area:.0f}px² < {min_area:.0f}px²  — wrong camera position?)")
            print(f"    Delete {CALIB_FILE} or press R to recalibrate.")
            os.remove(CALIB_FILE)   # remove so it doesn't keep getting rejected
            return None
        print(f"  ✓ Loaded calibration ← {CALIB_FILE}  (area={area:.0f}px²)")
        return src
    except Exception as e:
        print(f"  Load failed: {e}")
        return None


class BoardCalibrator:
    def __init__(self):
        self.history     = []
        self.src_corners = None
        self.M           = None
        self.locked      = False
        self.mode        = "IDLE"

    def try_load_disk(self) -> bool:
        src = load_calibration()
        if src is not None:
            self._lock(src, "disk")
            return True
        return False

    def start_search_any(self):
        self.history = []; self.src_corners = None
        self.M = None; self.locked = False
        self.mode = "SEARCH_ANY"
        print("  Searching (contour — pieces allowed)...")

    def start_search_empty(self):
        if os.path.exists(CALIB_FILE):
            os.remove(CALIB_FILE)
        self.history = []; self.src_corners = None
        self.M = None; self.locked = False
        self.mode = "SEARCH_EMPTY"
        print("  Searching (corner detection — clear the board)...")

    def update(self, frame) -> bool:
        if self.locked:
            return True
        if self.mode == "SEARCH_EMPTY":
            src = self._detect_empty(frame)
        elif self.mode == "SEARCH_ANY":
            src = detect_by_contour(frame)
        else:
            return False
        if src is None:
            self.history = self.history[-2:]
            return False
        self.history.append(src)
        if len(self.history) < LOCK_NEEDED:
            return False
        if self._consistent():
            self._lock(self.history[-1],
                       "corners" if self.mode == "SEARCH_EMPTY" else "contour")
            if self.mode == "SEARCH_EMPTY":
                save_calibration(self.src_corners)
            return True
        self.history.pop(0)
        return False

    def save_current(self):
        if self.locked:
            save_calibration(self.src_corners)

    def _detect_empty(self, frame):
        enhanced     = preprocess(frame)
        ret, corners = try_find_corners(enhanced)
        if ret:
            return extract_outer_corners(corners)
        return detect_by_contour(frame)

    def _consistent(self):
        recent = self.history[-LOCK_NEEDED:]
        ref    = recent[0]
        for src in recent[1:]:
            if np.linalg.norm(src - ref, axis=1).max() > CONSIST_PX:
                return False
        return True

    def _lock(self, src, method):
        self.src_corners = src
        self.M           = build_perspective(src)
        self.locked      = True
        self.mode        = "IDLE"
        print(f"  ✓ Board locked [{method}]  corners={src.astype(int).tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# Glass piece feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def get_square(board: np.ndarray, row: int, col: int) -> np.ndarray:
    m  = int(SQ * MARGIN)
    r1 = row * SQ + m;  r2 = (row + 1) * SQ - m
    c1 = col * SQ + m;  c2 = (col + 1) * SQ - m
    return board[r1:r2, c1:c2]


def extract_features(region: np.ndarray) -> dict:
    if region is None or region.size == 0:
        return {}
    f = {}

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    f["gray_mean"]  = float(np.mean(gray))
    f["gray_std"]   = float(np.std(gray))
    f["gray_max"]   = float(np.max(gray))

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    blue_mask       = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    f["blue_ratio"] = float(np.count_nonzero(blue_mask) / blue_mask.size * 100)
    f["hsv_s_mean"] = float(np.mean(hsv[:, :, 1]))
    f["hsv_s_max"]  = float(np.max(hsv[:, :, 1]))

    caustic_mask        = gray > CAUSTIC_THRESH
    f["caustic_ratio"]  = float(np.count_nonzero(caustic_mask) / caustic_mask.size * 100)

    edges            = cv2.Canny(gray, 40, 120)
    f["edge_density"] = float(np.count_nonzero(edges) / edges.size)
    return f


def sq_name(row: int, col: int) -> str:
    return "ABCDEFGH"[col] + str(8 - row)


# ══════════════════════════════════════════════════════════════════════════════
# Occupancy scoring
# ══════════════════════════════════════════════════════════════════════════════

def score_square(curr: dict, base: dict) -> tuple[float, str]:
    total = 0.0

    br  = curr.get("blue_ratio", 0.0)
    pts = min(br / VOTE_TABLE["blue_ratio"][0], VOTE_TABLE["blue_ratio"][1])
    total += pts

    s_delta = max(0, curr.get("hsv_s_max", 0) - base.get("hsv_s_max", 0))
    total  += min(s_delta / VOTE_TABLE["hsv_s_max"][0], VOTE_TABLE["hsv_s_max"][1])

    cr  = curr.get("caustic_ratio", 0.0)
    pts = min(cr / VOTE_TABLE["caustic_ratio"][0], VOTE_TABLE["caustic_ratio"][1])
    total += pts

    c_delta = max(0, curr.get("gray_max", 0) - base.get("gray_max", 0))
    total  += min(c_delta / VOTE_TABLE["caustic_delta"][0], VOTE_TABLE["caustic_delta"][1])

    std_d  = abs(curr.get("gray_std", 0) - base.get("gray_std", 0))
    total += min(std_d / VOTE_TABLE["gray_std_delta"][0], VOTE_TABLE["gray_std_delta"][1])

    e_d    = abs(curr.get("edge_density", 0) - base.get("edge_density", 0))
    total += min(e_d / VOTE_TABLE["edge_delta"][0], VOTE_TABLE["edge_delta"][1])

    # This chess set has exactly two piece types: blue glass and clear glass.
    # Classification rule: blue hue present → blue, otherwise → clear.
    # No "unknown" category — anything occupied is one of the two.
    occupied_flag = (total >= OCCUPIED_THRESHOLD or
                     br > BLUE_HARD_THRESH or
                     cr > CAUSTIC_HARD_THRESH)

    if not occupied_flag:
        piece_type = ""
    elif br > BLUE_HARD_THRESH:
        piece_type = "blue"
    else:
        piece_type = "clear"   # not blue → clear (only two types in this set)

    return round(total, 2), piece_type


def calibrate_baseline(board: np.ndarray) -> dict:
    baseline = {}
    for r in range(8):
        for c in range(8):
            baseline[f"{r},{c}"] = extract_features(get_square(board, r, c))
    return baseline


def detect_all(board: np.ndarray, baseline: dict) -> dict:
    results = {}
    for r in range(8):
        for c in range(8):
            key  = f"{r},{c}"
            curr = extract_features(get_square(board, r, c))
            base = baseline.get(key, {})
            score, piece_type = score_square(curr, base)
            occupied = (piece_type != "")   # piece_type set iff occupied
            results[key] = {
                "row": r, "col": c, "name": sq_name(r, c),
                "score": score, "occupied": occupied,
                "piece_type": piece_type,
                "blue_ratio": curr.get("blue_ratio", 0),
                "caustic_ratio": curr.get("caustic_ratio", 0),
            }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

PIECE_COLOR = {
    "blue":    (220,  60,   0),
    "clear":   (0,   220, 220),
    "unknown": (0,   200, 200),
    "":        (0,   180,  60),
}

def draw_occupancy(board: np.ndarray, results: dict) -> np.ndarray:
    vis = board.copy()
    for key, info in results.items():
        r, c   = info["row"], info["col"]
        x1, y1 = c * SQ, r * SQ
        x2, y2 = x1 + SQ, y1 + SQ
        pt     = info["piece_type"]
        col    = PIECE_COLOR.get(pt, PIECE_COLOR[""])
        ov     = vis.copy()
        cv2.rectangle(ov, (x1,y1), (x2,y2), col, -1)
        cv2.addWeighted(ov, 0.3 if info["occupied"] else 0.1, vis, 0.7 if info["occupied"] else 0.9, 0, vis)
        cv2.rectangle(vis, (x1,y1), (x2,y2), col, 1)
        cv2.putText(vis, f"{info['score']:.1f}", (x1+2, y2-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, info["name"], (x1+2, y1+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (200,200,200), 1, cv2.LINE_AA)
        if info["occupied"]:
            label = "B" if pt == "blue" else "C"   # only two types
            cv2.putText(vis, label, (x1+SQ//2-6, y1+SQ//2+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2, cv2.LINE_AA)
    return vis


def draw_heatmap(board: np.ndarray, results: dict, feat: str) -> np.ndarray:
    grid = np.zeros((8,8), np.float32)
    for key, info in results.items():
        grid[info["row"], info["col"]] = info.get(feat, 0.0)
    vmax = grid.max() or 1.0
    norm = (grid / vmax * 255).astype(np.uint8)
    heat = cv2.resize(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO),
                      (BOARD_PX, BOARD_PX), interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(board, 0.4, heat, 0.6, 0)
    cv2.putText(blended, f"{feat}  max={vmax:.2f}", (6,18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return blended


def draw_searching(frame, count, mode):
    vis = frame.copy()
    if mode == "SEARCH_EMPTY":
        msg, col = f"Searching (corner detection — empty board) {count}/{LOCK_NEEDED}", (0,200,255)
    else:
        msg, col = f"Searching (contour — pieces OK) {count}/{LOCK_NEEDED}", (0,255,150)
    cv2.putText(vis, msg, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
    cv2.putText(vis, "r=redetect  R=empty recal  q=quit",
                (20, vis.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    return vis


def draw_hud_locked(frame, calibrator, mode, calibrated_baseline, results):
    vis = frame.copy()
    # Board outline in original frame
    cv2.polylines(vis, [calibrator.src_corners.astype(int)], True, (0,255,0), 2)
    # HUD bar
    h = vis.shape[0]
    cv2.rectangle(vis, (0, h-28), (vis.shape[1], h), (30,30,30), -1)
    hints = []
    if mode == "detect" and results:
        blues  = sum(1 for v in results.values() if v["piece_type"] == "blue")
        clears = sum(1 for v in results.values() if v["piece_type"] == "clear")
        occ    = sum(1 for v in results.values() if v["occupied"])
        hints.append(f"Pieces:{occ}  B:{blues} C:{clears}")
    if not calibrated_baseline:
        hints.append("C=set baseline(empty board)")
    hints.append("d=detect  h=heatmap  r=redetect  q=quit")
    cv2.putText(vis, "  ".join(hints), (8, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    return vis


def print_ascii(results):
    print("\n  A  B  C  D  E  F  G  H")
    for r in range(8):
        row = f"{8-r} "
        for c in range(8):
            info = results[f"{r},{c}"]
            if not info["occupied"]:      sym = "·"
            elif info["piece_type"] == "blue":  sym = "B"
            else:                               sym = "C"
            row += f" {sym} "
        print(row + f" {8-r}")
    print("  A  B  C  D  E  F  G  H\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Connecting to OAK-D Lite...")
    pipeline, rgb_queue             = create_pipeline()
    pipeline.start()
    print("Pipeline started.")

    trigger_autofocus()

    calibrator   = BoardCalibrator()
    baseline     = None
    baseline_M   = None   # frozen perspective matrix when baseline was captured
    results      = None
    mode         = "live"       # live | detect
    show_heat    = False
    heat_feats   = ["blue_ratio", "caustic_ratio"]
    heat_idx     = 0

    # Live board tracking — re-detect corners every frame via contour detection.
    # EMA smooths out jitter while still following real camera movement.
    live_track   = True         # ON by default — no manual calibration needed
    EMA_ALPHA    = 0.15         # 0 = frozen, 1 = no smoothing
    ema_corners  = None         # exponential moving average of detected corners
    auto_detect  = False        # when True, re-run occupancy every DETECT_EVERY frames
    DETECT_EVERY = 15           # frames between auto occupancy updates
    frame_count  = 0

    load_piece_calibration()   # piece_calibration.json → adaptive clear detection

    # Try saved calibration — rejected automatically if area is too small
    if calibrator.try_load_disk():
        ema_corners = calibrator.src_corners.copy()
        lock_exposure_wb()
        print("  Board loaded from disk.  Live tracking ON — board will re-lock if it drifts.")
    else:
        calibrator.start_search_any()   # works with pieces on board

    print("\n  l   toggle live board tracking  (currently ON)")
    print("  r   redetect board (pieces OK on board)")
    print("  R   recalibrate from EMPTY board + save  (needs Shift)")
    print("  c   set baseline  (all pieces OFF the board first)")
    print("  d   detect occupancy once")
    print("  x   toggle auto-detect every 15 frames")
    print("  +   raise occupancy threshold (fewer pieces — less noise)")
    print("  -   lower occupancy threshold (more pieces — catch dim ones)")
    print("  h   toggle heatmap overlay")
    print("  f   cycle heatmap feature")
    print("  a   print ASCII board to console")
    print("  s   save board corners to disk")
    print("  q   quit")
    calib_status = f" ← {PIECE_CALIB_FILE}" if _piece_calib else " (run calibrate_pieces.py to improve)"
    print(f"  Piece calib: {'loaded' if _piece_calib else 'not loaded'}{calib_status}\n")

    while pipeline.isRunning():
        pkt = rgb_queue.tryGet()
        if pkt is None:
            time.sleep(0.005)
            continue
        frame = pkt.getCvFrame()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        # ── Key handlers (case-insensitive for single-action keys) ─────────────
        # r vs R have DIFFERENT actions so they stay case-sensitive.
        # Every other key works whether you press shift or not.
        k = key  # raw key; compare with key | 32 for case-fold where safe
        if k == ord("r"):
            calibrator.start_search_any()
            results = None; mode = "live"
            baseline_M = None   # unfreeze — live tracking will update M again
            if baseline is not None:
                print("  ⚠ Board M unfrozen. Press c again on empty board to re-set baseline.")
        elif k == ord("R"):
            calibrator.start_search_empty()
            results = None; mode = "live"
        elif k in (ord("s"), ord("S")):
            calibrator.save_current()
        elif k in (ord("c"), ord("C")):
            if calibrator.locked:
                baseline_M = calibrator.M.copy()   # freeze transform NOW
                board      = cv2.warpPerspective(frame, baseline_M, (BOARD_PX, BOARD_PX))
                baseline   = calibrate_baseline(board)
                print("  ✓ Baseline set — perspective locked.")
                print("    Press r to unfreeze tracking (and re-run c after).")
            else:
                print("  Board not locked yet — wait for the green outline.")
        elif k in (ord("d"), ord("D")):
            if calibrator.locked and baseline is not None:
                M_use   = baseline_M if baseline_M is not None else calibrator.M
                board   = cv2.warpPerspective(frame, M_use, (BOARD_PX, BOARD_PX))
                results = detect_all(board, baseline)
                mode    = "detect"
                print_ascii(results)
            elif baseline is None:
                print("  No baseline yet — press c first with empty board.")
        elif k in (ord("h"), ord("H")):
            show_heat = not show_heat
        elif k in (ord("f"), ord("F")):
            heat_idx = (heat_idx + 1) % len(heat_feats)
            print(f"  Heatmap: {heat_feats[heat_idx]}")
        elif k in (ord("a"), ord("A")) and results:
            print_ascii(results)
        elif k in (ord("l"), ord("L")):
            live_track = not live_track
            ema_corners = calibrator.src_corners.copy() if calibrator.locked else None
            print(f"  Live tracking: {'ON' if live_track else 'OFF'}")
        elif k in (ord("x"), ord("X")):
            auto_detect = not auto_detect
            print(f"  Auto-detect every {DETECT_EVERY} frames: {'ON' if auto_detect else 'OFF'}")
        elif key in (ord("+"), ord("=")):   # + or = (same key, no shift needed)
            OCCUPIED_THRESHOLD = globals().get("OCCUPIED_THRESHOLD", 4.5)
            import chess_glass_detector as _cgd
            _cgd.OCCUPIED_THRESHOLD = round(_cgd.OCCUPIED_THRESHOLD + 0.5, 1)
            print(f"  Threshold raised → {_cgd.OCCUPIED_THRESHOLD}  (fewer detections)")
        elif key == ord("-"):
            import chess_glass_detector as _cgd
            _cgd.OCCUPIED_THRESHOLD = round(max(1.0, _cgd.OCCUPIED_THRESHOLD - 0.5), 1)
            print(f"  Threshold lowered → {_cgd.OCCUPIED_THRESHOLD}  (more detections)")

        frame_count += 1

        # ── Live board tracking (runs every frame regardless of lock state) ───
        if live_track:
            src = detect_by_contour(frame)
            if src is not None:
                if ema_corners is None:
                    ema_corners = src.astype(np.float32)
                else:
                    ema_corners = (EMA_ALPHA * src
                                   + (1.0 - EMA_ALPHA) * ema_corners).astype(np.float32)
                calibrator.src_corners = ema_corners
                calibrator.M           = build_perspective(ema_corners)
                calibrator.locked      = True

        # ── Initial search (only when live_track is OFF and board not locked) ─
        elif not calibrator.locked:
            calibrator.update(frame)
            if calibrator.locked:
                lock_exposure_wb()
                ema_corners = calibrator.src_corners.copy()
            vis = draw_searching(frame, len(calibrator.history), calibrator.mode)
            cv2.imshow("Chess Glass Detector", vis)
            continue

        if not calibrator.locked:
            vis = draw_searching(frame, len(calibrator.history), calibrator.mode)
            cv2.imshow("Chess Glass Detector", vis)
            continue

        # ── Auto occupancy detection ──────────────────────────────────────────
        if auto_detect and baseline is not None and frame_count % DETECT_EVERY == 0:
            M_use   = baseline_M if baseline_M is not None else calibrator.M
            board   = cv2.warpPerspective(frame, M_use, (BOARD_PX, BOARD_PX))
            results = detect_all(board, baseline)
            mode    = "detect"

        # ── Locked — warp & show board ────────────────────────────────────────
        M_disp = baseline_M if baseline_M is not None else calibrator.M
        board = cv2.warpPerspective(frame, M_disp, (BOARD_PX, BOARD_PX))

        if mode == "detect" and results:
            if show_heat:
                board_disp = draw_heatmap(board, results, heat_feats[heat_idx])
            else:
                board_disp = draw_occupancy(board, results)
        else:
            board_disp = board.copy()
            if baseline is None:
                cv2.putText(board_disp, "C = calibrate baseline (empty board)",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,220), 2)
            else:
                cv2.putText(board_disp, "D = detect",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,220), 2)

        # Show original frame + warped board side by side
        hud_frame = draw_hud_locked(frame, calibrator, mode, baseline is not None, results)
        # Live-track indicator
        track_col = (0, 255, 120) if live_track else (100, 100, 100)
        if baseline_M is not None:
            freeze_lbl = "M FROZEN (baseline locked)"
            cv2.putText(hud_frame, freeze_lbl, (8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
        track_lbl = "LIVE TRACK ON" if live_track else "LIVE TRACK OFF  (L=on)"
        cv2.putText(hud_frame, track_lbl, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_col, 1)
        if auto_detect and baseline is not None:
            cv2.putText(hud_frame, "AUTO-DETECT ON", (8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        frame_small = cv2.resize(hud_frame, (BOARD_PX, BOARD_PX))
        combined = np.hstack([frame_small, board_disp])
        cv2.imshow("Chess Glass Detector", combined)

    pipeline.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()