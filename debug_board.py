#!/usr/bin/env python3 -u
"""
debug_board.py — Real-time board square detection debugger.

Two windows:
  "raw + grid"    : camera feed with the 8×8 grid projected back onto the board.
                    Confirms square boundaries are in the right place.
  "warped + diff" : perspective-corrected board with per-square diff overlay.
                    Green = stable, orange = near threshold, red = triggered.
                    In Blue mode (press B): cyan = blue piece detected.

Controls
────────
  S       — re-capture baseline snapshot ("before" image)
  B       — toggle blue-piece detection mode (shows HSV blue squares)
  +  / =  — raise diff threshold by 2
  -       — lower diff threshold by 2
  Q / Esc — quit

Workflow for tuning
───────────────────
1. Diff mode: run with empty board → press S → place pieces → squares with
   pieces turn red.  Adjust threshold until only piece squares are red.
2. Blue mode (press B): blue glass pieces should show CYAN overlays.
   If pieces aren't detected, tune BLUE_HUE_LO/HI/SAT_MIN/VAL_MIN/BLUE_FRAC
   in constants.py and restart.
3. Run with pieces in starting position → press S → move a piece →
   only the two changed squares should turn red.
"""

import time, cv2, numpy as np
import constants as const
from servo_pick import init_camera, get_frame

BOARD_PX    = 640
SQ          = BOARD_PX // 8
DST_CORNERS = np.float32([[0, 0], [BOARD_PX, 0],
                           [BOARD_PX, BOARD_PX], [0, BOARD_PX]])
THRESH_INIT = 15


def load_corners(path="board_calibration.npy"):
    try:
        return np.load(path).astype(np.float32)
    except FileNotFoundError:
        raise FileNotFoundError(
            "board_calibration.npy not found.\n"
            "  Run: python chess_glass_detector.py  →  r  →  s  →  q"
        )


def sq_name(r, c):
    """Warped-image (row, col) → chess square name.  r=0 → rank 1 (white at top, blue/black at bottom)."""
    return "abcdefgh"[c] + str(r + 1)


def draw_blue_overlay(warped: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Shows which squares contain a blue piece (HSV hue detection).
    Cyan overlay = blue piece detected.  Grey outline = not detected.
    Also prints the per-square blue-pixel fraction so you can tune BLUE_FRAC.
    """
    hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lo   = np.array([const.BLUE_HUE_LO,  const.BLUE_SAT_MIN, const.BLUE_VAL_MIN], np.uint8)
    hi   = np.array([const.BLUE_HUE_HI,  255,                 255],                np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    out  = warped.copy()
    detected = []

    I    = const.SQ_INSET
    area = float((SQ - 2*I) ** 2)
    for r in range(8):
        for c in range(8):
            y1, x1 = r * SQ, c * SQ
            y2, x2 = y1 + SQ, x1 + SQ
            sq_mask = mask[y1+I:y2-I, x1+I:x2-I]
            frac    = np.count_nonzero(sq_mask) / area
            name    = sq_name(r, c)
            has_blue = frac >= const.BLUE_FRAC

            if has_blue:
                bgr, alpha = (255, 200, 0), 0.40   # cyan
                detected.append(name)
            else:
                bgr, alpha = (80, 80, 80), 0.10    # dark grey

            roi  = out[y1:y2, x1:x2]
            fill = np.full_like(roi, bgr)
            cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, roi)
            cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)
            # Show the inset boundary so you can see what's being sampled
            cv2.rectangle(out, (x1+I, y1+I), (x2-I, y2-I), (0, 180, 255), 1)

            # Fraction text
            cv2.putText(out, f"{frac:.2f}", (x1 + 2, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        (0, 255, 180) if has_blue else (180, 180, 180), 1)
            cv2.putText(out, name, (x1 + 3, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (210, 210, 210), 1)

    return out, detected


def draw_warped_overlay(warped, baseline, threshold):
    """
    Returns (annotated_image, list_of_triggered_square_names).
    Each square is colour-coded by mean pixel diff vs baseline.
    """
    out = warped.copy()
    triggered = []

    I = const.SQ_INSET
    for r in range(8):
        for c in range(8):
            y1, x1 = r * SQ, c * SQ
            y2, x2 = y1 + SQ, x1 + SQ

            s_now  = warped[y1+I:y2-I, x1+I:x2-I].astype(np.float32)
            s_base = baseline[y1+I:y2-I, x1+I:x2-I].astype(np.float32)
            diff   = float(np.mean(np.abs(s_now - s_base)))

            name = sq_name(r, c)
            if diff > threshold:
                bgr, alpha = (0, 0, 210), 0.45      # red
                triggered.append(name)
            elif diff > threshold * 0.6:
                bgr, alpha = (0, 130, 255), 0.32    # orange
            else:
                bgr, alpha = (0, 200, 0), 0.12      # green

            roi  = out[y1:y2, x1:x2]
            fill = np.full_like(roi, bgr)
            cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, roi)

            # Grid border
            cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)

            # Diff value (top of square)
            cv2.putText(out, f"{diff:.0f}",
                        (x1 + 3, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Square name (bottom of square)
            cv2.putText(out, name,
                        (x1 + 3, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (210, 210, 210), 1)

    return out, triggered


def draw_raw_grid(frame, M_inv):
    """
    Project the warped 8×8 grid back onto the raw camera frame.
    Shows exactly which board region each square maps to.
    """
    out = frame.copy()

    def wpt(wx, wy):
        p = cv2.perspectiveTransform(
            np.array([[[float(wx), float(wy)]]], dtype=np.float32), M_inv)
        return tuple(p[0, 0].astype(int))

    # Grid lines
    for i in range(9):
        cv2.line(out, wpt(i * SQ, 0),       wpt(i * SQ, BOARD_PX), (0, 220, 0), 1)
        cv2.line(out, wpt(0, i * SQ),       wpt(BOARD_PX, i * SQ), (0, 220, 0), 1)

    # Corner labels (chess notation)
    for r, c, label in [(0, 0, "a8"), (0, 7, "h8"), (7, 0, "a1"), (7, 7, "h1")]:
        pt = wpt(c * SQ + SQ // 2, r * SQ + SQ // 2)
        cv2.circle(out, pt, 5, (0, 255, 255), -1)
        cv2.putText(out, label, (pt[0] + 6, pt[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    return out


def main():
    print("Loading board corners...")
    corners = load_corners()
    M     = cv2.getPerspectiveTransform(corners, DST_CORNERS)
    M_inv = np.linalg.inv(M)

    print("Starting camera...")
    device, pipeline, rgb_queue, _ = init_camera()

    cv2.namedWindow("raw + grid",    cv2.WINDOW_NORMAL)
    cv2.namedWindow("warped + diff", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("raw + grid",    700, 530)
    cv2.resizeWindow("warped + diff", 640, 700)

    threshold  = THRESH_INIT
    baseline   = None
    blue_mode  = False

    print(f"\nThreshold = {threshold}")
    print("S = new baseline   B = toggle blue mode   +/- = adjust threshold   Q = quit\n")

    while True:
        frame = get_frame(pipeline, rgb_queue)
        if frame is None:
            time.sleep(0.02)
            continue

        warped = cv2.warpPerspective(frame, M, (BOARD_PX, BOARD_PX))

        if baseline is None:
            baseline = warped.copy()
            print("Initial baseline captured. Move pieces to see diffs. Press S to reset.")

        if blue_mode:
            overlay, triggered = draw_blue_overlay(warped)
            mode_label = f"BLUE MODE  hue={const.BLUE_HUE_LO}-{const.BLUE_HUE_HI}  frac>={const.BLUE_FRAC}"
            status = f"{mode_label}  |  blue squares: {triggered if triggered else 'none'}"
        else:
            overlay, triggered = draw_warped_overlay(warped, baseline, threshold)
            status = f"DIFF MODE  thresh={threshold}  |  triggered: {triggered if triggered else 'none'}"

        # Status bar below the warped image
        bar = np.zeros((32, BOARD_PX, 3), dtype=np.uint8)
        cv2.putText(bar, status, (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
        display = np.vstack([overlay, bar])

        raw_grid = draw_raw_grid(frame, M_inv)

        cv2.imshow("raw + grid",    raw_grid)
        cv2.imshow("warped + diff", display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            baseline = warped.copy()
            print("Baseline re-captured!")
        elif key == ord('b'):
            blue_mode = not blue_mode
            print(f"{'Blue detection mode' if blue_mode else 'Diff mode'}")
        elif key in (ord('+'), ord('=')):
            threshold += 2
            print(f"Threshold = {threshold}")
        elif key == ord('-'):
            threshold = max(2, threshold - 2)
            print(f"Threshold = {threshold}")

    device.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
