#!/usr/bin/env python3 -u
"""
chessgame.py — ARIA chess robot full pipeline
══════════════════════════════════════════════
Run from chessbotV1/:
  python chessgame.py

Flow
────
1. quick_cal.py must have been run this session (fresh T_cam_to_arm).
2. Script asks for player colour (w/b) and engine strength.
3. Clear the board → camera takes empty baseline.
4. Set up pieces → camera verifies starting position.
5. Game loop:
     Human turn  → watch camera for piece move, validate, update state.
     Robot turn  → Stockfish picks move → arm executes → update state.
"""

import sys, select, time, cv2, chess, numpy as np

import constants as const
from engine.stockfish      import Stockfish
from servo_pick            import (init_camera, get_frame,
                                   square_center_pixel, pixel_to_arm)

STOCKFISH_DEPTH   = 15
STABLE_FRAMES     = 6
STABLE_GAP_S      = 0.25
BOARD_PX          = 640
SQ                = BOARD_PX // 8    # 80 px per square
DST_CORNERS       = np.float32([[0,0],[BOARD_PX,0],[BOARD_PX,BOARD_PX],[0,BOARD_PX]])
BASELINE_IMG      = "empty_board.png"
PIX_DIFF_THRESH   = 10    # threshold vs empty board (piece-count check)
FRAME_DIFF_THRESH = 15    # before→after comparison — original value; NOISE_GATE handles bulk rejection
# Consecutive-frame stability thresholds
MOTION_THRESH     = 10    # sq mean diff to call a square "moving" in live frame-pairs
STABLE_RUN_N      = 10    # consecutive stable frame-pairs needed to call the board "still"
MOTION_TRIGGER_N  = 3     # consecutive motion frames before declaring "move in progress"
MOTION_SQ_MIN     = 2     # minimum squares moving to count a frame as "in motion"
NOISE_GATE        = 6     # before→after: more than this = board-wide noise (castling max = 4)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def det_to_sq(r: int, c: int) -> int:
    """Warped image (row, col) → chess.Square.  Row 0 = rank 1 (white at top, blue/black at bottom)."""
    return chess.square(c, r)

def sq_to_servo(sq: int) -> tuple[int, int]:
    """chess.Square → servo_pick (col, row).  row 0 = rank 1."""
    return chess.square_file(sq), chess.square_rank(sq)


# ── Camera / board helpers ────────────────────────────────────────────────────

def warp(frame: np.ndarray, M: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(frame, M, (BOARD_PX, BOARD_PX))


def _square_img(warped: np.ndarray, r: int, c: int) -> np.ndarray:
    return warped[r*SQ:(r+1)*SQ, c*SQ:(c+1)*SQ]


def get_occupancy(warped: np.ndarray, empty_warped: np.ndarray,
                  threshold: int = PIX_DIFF_THRESH) -> dict[int, bool]:
    """
    Pixel-diff occupancy: compare each square against the saved empty board image.
    Returns {chess.Square: occupied_bool} for all 64 squares.
    """
    occ = {}
    for r in range(8):
        for c in range(8):
            curr = _square_img(warped, r, c).astype(np.float32)
            base = _square_img(empty_warped, r, c).astype(np.float32)
            diff = float(np.mean(np.abs(curr - base)))
            occ[det_to_sq(r, c)] = diff > threshold
    return occ


def get_stable_occupancy(pipeline, rgb_queue, M, empty_warped,
                         n=STABLE_FRAMES, gap=STABLE_GAP_S,
                         window: str | None = None) -> dict[int, bool]:
    """
    Poll n frames with gap seconds between them.
    Returns majority-vote occupancy (most common value per square).
    """
    counts = {sq: 0 for sq in range(64)}
    for _ in range(n):
        frame = get_frame(pipeline, rgb_queue)
        if frame is None:
            time.sleep(gap)
            continue
        w = warp(frame, M)
        occ = get_occupancy(w, empty_warped)
        for sq, val in occ.items():
            if val:
                counts[sq] += 1
        if window:
            cv2.imshow(_CAM_WIN, frame)
            cv2.waitKey(1)
        time.sleep(gap)
    return {sq: counts[sq] > n // 2 for sq in range(64)}


# ── Frame-to-frame snapshot helpers ──────────────────────────────────────────

def _median_warped(pipeline, rgb_queue, M, n=6, gap=0.12, window=None):
    """Pixel-wise median of n warped frames — low-noise board snapshot."""
    frames = []
    for _ in range(n):
        frame = get_frame(pipeline, rgb_queue)
        if frame is not None:
            frames.append(warp(frame, M).astype(np.float32))
            if window:
                cv2.imshow(_CAM_WIN, frame)
                cv2.waitKey(1)
        time.sleep(gap)
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _changed_squares(img1, img2, threshold=FRAME_DIFF_THRESH):
    """
    Squares whose mean pixel value changed significantly between two warped images.
    Global brightness offset (from hand shadow or lighting drift) is subtracted
    before comparison so a uniform dimming does not flag all 64 squares.
    """
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    # Remove global brightness shift (hand casts a shadow over the whole board)
    offset = float(np.mean(f2) - np.mean(f1))
    f2 = np.clip(f2 - offset, 0, 255)
    I = const.SQ_INSET
    changed = set()
    for r in range(8):
        for c in range(8):
            s1 = f1[r*SQ+I:(r+1)*SQ-I, c*SQ+I:(c+1)*SQ-I]
            s2 = f2[r*SQ+I:(r+1)*SQ-I, c*SQ+I:(c+1)*SQ-I]
            if np.mean(np.abs(s1 - s2)) > threshold:
                changed.add(det_to_sq(r, c))
    return changed


def _consec_motion_squares(img1, img2) -> int:
    """
    Count squares that differ between two *consecutive* live frames.
    Used to detect whether something is moving (hand/piece) right now.
    Global brightness shift is compensated the same way as _changed_squares.
    """
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    offset = float(np.mean(f2) - np.mean(f1))
    f2 = np.clip(f2 - offset, 0, 255)
    I = const.SQ_INSET
    count = 0
    for r in range(8):
        for c in range(8):
            s1 = f1[r*SQ+I:(r+1)*SQ-I, c*SQ+I:(c+1)*SQ-I]
            s2 = f2[r*SQ+I:(r+1)*SQ-I, c*SQ+I:(c+1)*SQ-I]
            if np.mean(np.abs(s1 - s2)) > MOTION_THRESH:
                count += 1
    return count


# ── Live board debug window ───────────────────────────────────────────────────

_BOARD_WIN = "board view"
_CAM_WIN   = "camera"

def _open_debug_windows():
    cv2.namedWindow(_BOARD_WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(_CAM_WIN,   cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_BOARD_WIN, 660, 700)
    cv2.resizeWindow(_CAM_WIN,   700, 530)


def _draw_board_debug(warped: np.ndarray, state: str,
                      before_blue: set[int] | None = None,
                      after_blue:  set[int] | None = None) -> None:
    """
    Show the warped board with:
      - Blue detection overlay (cyan = blue piece, fraction shown per square)
      - Inset sampling boundary (orange rectangle)
      - before/after blue squares highlighted differently when comparing
      - State label at the top
    """
    hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lo   = np.array([const.BLUE_HUE_LO,  const.BLUE_SAT_MIN, const.BLUE_VAL_MIN], np.uint8)
    hi   = np.array([const.BLUE_HUE_HI,  255,                 255],                np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    out  = warped.copy()
    I    = const.SQ_INSET
    area = float((SQ - 2*I) ** 2)

    for r in range(8):
        for c in range(8):
            y1, x1 = r * SQ, c * SQ
            y2, x2 = y1 + SQ, x1 + SQ
            sq_idx  = det_to_sq(r, c)
            sq_mask = mask[y1+I:y2-I, x1+I:x2-I]
            frac    = np.count_nonzero(sq_mask) / area
            name    = chess.square_name(sq_idx)

            # Colour by detection state
            if after_blue is not None and before_blue is not None:
                if sq_idx in (before_blue - after_blue):
                    bgr, alpha = (0, 60, 220), 0.50    # red — piece left
                elif sq_idx in (after_blue - before_blue):
                    bgr, alpha = (60, 220, 60), 0.50   # green — piece arrived
                elif sq_idx in after_blue:
                    bgr, alpha = (220, 160, 0), 0.35   # cyan — blue piece (unchanged)
                else:
                    bgr, alpha = (30, 30, 30), 0.10
            else:
                if frac >= const.BLUE_FRAC:
                    bgr, alpha = (220, 160, 0), 0.40   # cyan
                else:
                    bgr, alpha = (30, 30, 30), 0.10

            roi  = out[y1:y2, x1:x2]
            fill = np.full_like(roi, bgr)
            cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, roi)

            cv2.rectangle(out, (x1, y1), (x2, y2), (120, 120, 120), 1)
            cv2.rectangle(out, (x1+I, y1+I), (x2-I, y2-I), (0, 160, 255), 1)

            cv2.putText(out, f"{frac:.2f}", (x1+2, y1+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (0, 255, 180) if frac >= const.BLUE_FRAC else (140, 140, 140), 1)
            cv2.putText(out, name, (x1+2, y2-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

    # Status bar
    bar = np.zeros((36, BOARD_PX, 3), np.uint8)
    cv2.putText(bar, state, (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)
    display = np.vstack([out, bar])
    cv2.imshow(_BOARD_WIN, display)


# ── Blue piece detection ──────────────────────────────────────────────────────

def _blue_squares(warped: np.ndarray) -> set[int]:
    """
    Return the set of chess squares that contain a blue glass piece.

    Uses HSV hue detection — immune to auto-exposure and lighting changes
    because Hue is invariant to uniform illumination scaling.
    Thresholds live in constants.py (BLUE_HUE_LO/HI, BLUE_SAT_MIN, etc.).
    """
    hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lo   = np.array([const.BLUE_HUE_LO, const.BLUE_SAT_MIN, const.BLUE_VAL_MIN], np.uint8)
    hi   = np.array([const.BLUE_HUE_HI, 255,                255],                 np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    I    = const.SQ_INSET
    area = float((SQ - 2*I) ** 2)
    result: set[int] = set()
    for r in range(8):
        for c in range(8):
            sq_mask = mask[r*SQ+I:(r+1)*SQ-I, c*SQ+I:(c+1)*SQ-I]
            frac    = np.count_nonzero(sq_mask) / area
            if frac >= const.BLUE_FRAC:
                result.add(det_to_sq(r, c))
    return result


def _match_blue_move(board: chess.Board,
                     before_blue: set[int], after_blue: set[int],
                     human_color: chess.Color) -> chess.Move | None:
    """
    Infer the human's move from which squares lost / gained a blue piece.

    emptied = squares where a blue piece disappeared (piece was lifted).
    filled  = squares where a blue piece appeared  (piece was placed, incl. captures).

    Handles normal moves, captures, en-passant, and castling.
    Falls back to None when the detection is ambiguous.
    """
    emptied = before_blue - after_blue
    filled  = after_blue  - before_blue

    if not emptied:
        return None   # no blue piece moved

    # ── Castling: king + rook both move ───────────────────────────────────────
    if len(emptied) == 2 and len(filled) == 2:
        for move in board.legal_moves:
            if (board.color_at(move.from_square) == human_color
                    and board.is_castling(move)):
                rank = chess.square_rank(move.from_square)
                if chess.square_file(move.to_square) == 6:   # kingside
                    exp_emp  = {chess.square(4, rank), chess.square(7, rank)}
                    exp_fill = {chess.square(6, rank), chess.square(5, rank)}
                else:                                          # queenside
                    exp_emp  = {chess.square(4, rank), chess.square(0, rank)}
                    exp_fill = {chess.square(2, rank), chess.square(3, rank)}
                if emptied == exp_emp and filled == exp_fill:
                    return move
        return None

    # ── Single piece moves (normal, capture, en passant) ─────────────────────
    if len(emptied) != 1:
        return None   # more than one blue piece lifted — noise

    from_sq = next(iter(emptied))
    candidates = []

    for move in board.legal_moves:
        if board.color_at(move.from_square) != human_color:
            continue
        if move.from_square != from_sq:
            continue

        if board.is_capture(move):
            # After capture, blue piece is at to_sq → should be in filled.
            # Allow filled = {} as fallback if camera missed it.
            if move.to_square in filled or not filled:
                candidates.append(move)
        else:
            # Non-capture: to_sq must be in filled
            if move.to_square in filled:
                candidates.append(move)

    if len(candidates) == 1:
        return candidates[0]

    # Ambiguous captures from same square: use filled to disambiguate
    if len(candidates) > 1 and filled:
        to_sq = next(iter(filled))
        exact = [m for m in candidates if m.to_square == to_sq]
        if len(exact) == 1:
            return exact[0]

    return None


# ── Human move detection ──────────────────────────────────────────────────────

def _expected_diff(board: chess.Board, move: chess.Move):
    """
    Returns (emptied, filled) sets of squares for the given move.
    Capture: to_sq was occupied → stays occupied → not in 'filled'.
    En passant: captured pawn sq is added to 'emptied'.
    Castling: rook squares included.
    """
    f, t = move.from_square, move.to_square
    emptied: set[int] = {f}
    filled:  set[int] = set()

    if board.is_en_passant(move):
        captured_sq = chess.square(chess.square_file(t), chess.square_rank(f))
        emptied.add(captured_sq)
        filled.add(t)
    elif board.is_capture(move):
        pass   # to_sq stays occupied (captured piece removed, attacker arrives)
    else:
        filled.add(t)

    if board.is_castling(move):
        rank    = chess.square_rank(f)
        if chess.square_file(t) == 6:   # kingside
            emptied.add(chess.square(7, rank))
            filled.add(chess.square(5, rank))
        else:                            # queenside
            emptied.add(chess.square(0, rank))
            filled.add(chess.square(3, rank))

    return emptied, filled


def _match_move(board: chess.Board, before: dict, after: dict,
                human_color: chess.Color) -> chess.Move | None:
    """
    Identify which legal human move produced the observed occupancy change.
    Returns the Move, or None if no unique match found.
    """
    emptied = {sq for sq in range(64) if before.get(sq) and not after.get(sq)}
    filled  = {sq for sq in range(64) if not before.get(sq) and after.get(sq)}

    candidates = []
    for move in board.legal_moves:
        if board.color_at(move.from_square) != human_color:
            continue
        exp_emp, exp_fill = _expected_diff(board, move)
        if exp_emp == emptied and exp_fill == filled:
            candidates.append(move)

    if len(candidates) == 1:
        return candidates[0]

    # Ambiguous capture (multiple captures from same square): use 'filled'
    # to_sq of the capture is whichever opponent square is now missing.
    if len(candidates) > 1:
        robot_color  = not human_color
        robot_before = {sq for sq in range(64)
                        if before.get(sq) and board.color_at(sq) == robot_color}
        robot_after  = {sq for sq in range(64) if after.get(sq)
                        and board.color_at(sq) == robot_color}  # still there
        captured_sq  = robot_before - robot_after  # the one taken off
        if len(captured_sq) == 1:
            to_sq = captured_sq.pop()
            for m in candidates:
                if m.to_square == to_sq:
                    return m
    return None


def _wait_for_stable_board(pipeline, rgb_queue, M, window) -> np.ndarray:
    """
    Wait until STABLE_RUN_N consecutive frame-pairs all show < STABLE_THRESH
    per-square diff (compensated for global brightness).  Then return a clean
    median snapshot.  This handles: arm still homing, pieces still rocking,
    hand shadow that hasn't cleared yet.
    """
    prev         = None
    stable_count = 0
    while stable_count < STABLE_RUN_N:
        frame = get_frame(pipeline, rgb_queue)
        if frame is None:
            time.sleep(0.03)
            continue
        curr = warp(frame, M)
        if prev is not None:
            n_moving = _consec_motion_squares(prev, curr)
            if n_moving == 0:
                stable_count += 1
            else:
                stable_count = 0
        prev = curr
        if window:
            cv2.putText(frame, f"settling... {stable_count}/{STABLE_RUN_N}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2)
            cv2.imshow(window, frame)
            cv2.waitKey(1)
    return _median_warped(pipeline, rgb_queue, M, n=8, window=window)


def observe_human_move(pipeline, rgb_queue, M, empty_warped,
                       board: chess.Board, human_color: chess.Color,
                       window: str = "game") -> str:
    """
    Block until a legal human move is detected using consecutive-frame stability.

    State machine — no model required, immune to lighting drift:

      WAIT_STABLE  — consecutive frame-pairs all quiet → take before_img
      WAIT_MOTION  — watch for any square to start moving (piece lifted/placed)
      WAIT_SETTLE  — consecutive frame-pairs quiet again → take after_img
      COMPARE      — diff before vs after → match to legal move → done or retry

    Both before_img and after_img are taken only when the board is confirmed
    still, so arm shadows, rocking pieces, and hand presence never corrupt them.

    Global brightness shift (from hand shadow) is subtracted before comparing
    so a uniform dimming of the whole board does not flag all 64 squares.

    Manual UCI input is always accepted as an override.
    """
    human_sqs  = {sq for sq in range(64) if board.piece_at(sq) is not None
                  and board.color_at(sq) == human_color}
    robot_sqs  = {sq for sq in range(64) if board.piece_at(sq) is not None
                  and board.color_at(sq) != human_color}
    occ_before = {sq: board.piece_at(sq) is not None for sq in range(64)}

    def _typed_input_check() -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            typed = sys.stdin.readline().strip().lower()
            if len(typed) >= 4:
                uci_str = typed[:4]
                try:
                    move = board.parse_uci(uci_str)
                    if (move in board.legal_moves and
                            board.color_at(move.from_square) == human_color):
                        print(f"  Manual: {board.san(move)}  ({uci_str})")
                        return uci_str
                    else:
                        print(f"  '{uci_str}' not a legal human move — keep watching")
                except Exception:
                    print(f"  Invalid: '{typed}'")
        return None

    def _compare_and_match(before: np.ndarray, after: np.ndarray) -> str | None:
        # ── Primary: HSV blue piece tracking — immune to auto-exposure shifts ──
        before_blue = _blue_squares(before)
        after_blue  = _blue_squares(after)
        print(f"  Blue before: {sorted(chess.square_name(s) for s in before_blue)}")
        print(f"  Blue after:  {sorted(chess.square_name(s) for s in after_blue)}")

        _draw_board_debug(after, "COMPARING", before_blue, after_blue)
        cv2.waitKey(1)

        if before_blue != after_blue:
            move = _match_blue_move(board, before_blue, after_blue, human_color)
            if move is not None:
                print(f"  [blue] Detected: {board.san(move)}  ({move.uci()})")
                _draw_board_debug(after, f"DETECTED: {board.san(move)}", before_blue, after_blue)
                cv2.waitKey(1)
                return move.uci()
            print(f"  [blue] Changed: "
                  f"-{sorted(chess.square_name(s) for s in before_blue - after_blue)}"
                  f"  +{sorted(chess.square_name(s) for s in after_blue - before_blue)}"
                  f"  — no legal move matched")
        else:
            print("  [blue] No blue squares changed — falling back to pixel diff")

        # ── Fallback: pixel diff (for cases where blue HSV thresholds miss pieces) ──
        changed = _changed_squares(before, after)
        if not changed:
            print("  [diff] Board looks the same — did the piece go back?")
            return None
        if len(changed) > NOISE_GATE:
            print(f"  [diff] {len(changed)} squares changed — still noisy, ignoring")
            return None
        emptied = changed & human_sqs
        filled  = changed - (human_sqs | robot_sqs)
        print(f"  [diff] Changed: -{sorted(chess.square_name(s) for s in emptied)}"
              f"  +{sorted(chess.square_name(s) for s in filled)}")
        occ_after = dict(occ_before)
        for sq in emptied: occ_after[sq] = False
        for sq in filled:  occ_after[sq] = True
        move = _match_move(board, occ_before, occ_after, human_color)
        if move is not None:
            print(f"  [diff] Detected: {board.san(move)}  ({move.uci()})")
            return move.uci()
        print("  No legal move matched — watching again  (type UCI to override)")
        return None

    # ── WAIT_STABLE: board must be still before we take the reference snapshot ──
    print("  Waiting for board to settle...")
    before_img = _wait_for_stable_board(pipeline, rgb_queue, M, window)
    print("  Make your move... (or type UCI e.g. e7e5 + Enter)")

    while True:
        # ── WAIT_MOTION: require MOTION_TRIGGER_N consecutive frames showing ≥ MOTION_SQ_MIN
        #    squares moving.  Filters out single-frame ambient light flickers.
        prev          = None
        motion_streak = 0
        while True:
            uci = _typed_input_check()
            if uci:
                return uci

            frame = get_frame(pipeline, rgb_queue)
            if frame is None:
                time.sleep(0.03)
                continue
            curr = warp(frame, M)

            if window:
                cv2.putText(frame, "waiting for move...",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1)
                cv2.imshow(_CAM_WIN, frame)
                _draw_board_debug(curr, "WAITING FOR MOVE")
                cv2.waitKey(1)

            if prev is not None:
                n = _consec_motion_squares(prev, curr)
                motion_streak = motion_streak + 1 if n >= MOTION_SQ_MIN else 0
                if motion_streak >= MOTION_TRIGGER_N:
                    print("  Motion detected — waiting for board to settle...")
                    break
            prev = curr

        # ── WAIT_SETTLE: board must be still again before sampling after_img ───
        stable_count = 0
        while stable_count < STABLE_RUN_N:
            uci = _typed_input_check()
            if uci:
                return uci

            frame = get_frame(pipeline, rgb_queue)
            if frame is None:
                time.sleep(0.03)
                continue
            curr = warp(frame, M)

            n_moving = _consec_motion_squares(prev, curr) if prev is not None else 1
            stable_count = stable_count + 1 if n_moving == 0 else 0
            prev = curr

            if window:
                col = (0, 200, 80) if n_moving == 0 else (0, 80, 255)
                cv2.putText(frame, f"settling {stable_count}/{STABLE_RUN_N}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                cv2.imshow(_CAM_WIN, frame)
                _draw_board_debug(curr, f"SETTLING {stable_count}/{STABLE_RUN_N}")
                cv2.waitKey(1)

        # ── COMPARE: both snapshots taken when board was confirmed still ────────
        # Extra wait: pixel-motion stops before AE and piece reflections settle.
        print(f"  Board still — waiting {const.SETTLE_EXTRA_S:.0f}s for AE to settle...")
        t_wait = time.monotonic()
        while time.monotonic() - t_wait < const.SETTLE_EXTRA_S:
            frame = get_frame(pipeline, rgb_queue)
            if frame is not None:
                curr = warp(frame, M)
                elapsed = time.monotonic() - t_wait
                remaining = const.SETTLE_EXTRA_S - elapsed
                cv2.putText(frame, f"AE settling... {remaining:.1f}s",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                cv2.imshow(_CAM_WIN, frame)
                _draw_board_debug(curr, f"AE SETTLING {remaining:.1f}s")
                cv2.waitKey(1)
            else:
                time.sleep(0.03)
        after_img = _median_warped(pipeline, rgb_queue, M, n=8, window=window)
        uci = _compare_and_match(before_img, after_img)
        if uci:
            return uci

        # Failed — re-take before_img from current stable state and retry.
        # (The piece may have been put back, or we got a false motion.)
        before_img = _wait_for_stable_board(pipeline, rgb_queue, M, window)
        print("  Watching again...")


# ── Robot move execution ──────────────────────────────────────────────────────

def execute_robot_move(uci: str, board: chess.Board, sp) -> bool:
    """
    Execute a robot move using visual-servo pick/place.
    Handles captures (remove opponent piece first), castling, en passant.
    """
    move = board.parse_uci(uci)
    f_col, f_row = sq_to_servo(move.from_square)
    t_col, t_row = sq_to_servo(move.to_square)

    # Capture: clear destination first
    if board.is_capture(move):
        if board.is_en_passant(move):
            cap_sq  = chess.square(chess.square_file(move.to_square),
                                   chess.square_rank(move.from_square))
            cc, cr  = sq_to_servo(cap_sq)
            print(f"    en passant — removing pawn at {chess.square_name(cap_sq)}")
            sp.remove_to_bin(cc, cr)
        else:
            print(f"    capture — removing piece at {chess.square_name(move.to_square)}")
            sp.remove_to_bin(t_col, t_row)

    # Castling: move rook first
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) == 6:   # kingside
            r_from = chess.square(7, rank); r_to = chess.square(5, rank)
        else:                                         # queenside
            r_from = chess.square(0, rank); r_to = chess.square(3, rank)
        rfc, rfr = sq_to_servo(r_from)
        rtc, rtr = sq_to_servo(r_to)
        print(f"    castling — moving rook {chess.square_name(r_from)} → {chess.square_name(r_to)}")
        sp.pick(rfc, rfr)
        sp.place(rtc, rtr)

    # Main piece
    print(f"    moving {chess.square_name(move.from_square)} → {chess.square_name(move.to_square)}")
    sp.pick(f_col, f_row)
    sp.place(t_col, t_row)

    if move.promotion:
        print("    PROMOTION — please swap the piece on the board manually.")
        input("    Press Enter when done...")

    return True


# ── Board verification ────────────────────────────────────────────────────────

def verify_starting_position(pipeline, rgb_queue, M, empty_warped,
                             human_color: chess.Color, window="game") -> bool:
    """
    Scan the board and check that 32 pieces are present:
    16 on ranks 7-8 (far side = black) and 16 on ranks 1-2 (near side = white).
    Prints a warning if the count is off but does not block.
    """
    occ = get_stable_occupancy(pipeline, rgb_queue, M, empty_warped, n=8, window=window)
    far_side  = sum(1 for sq in range(64) if occ.get(sq)
                    and chess.square_rank(sq) >= 6)   # ranks 7-8
    near_side = sum(1 for sq in range(64) if occ.get(sq)
                    and chess.square_rank(sq) <= 1)   # ranks 1-2
    total = sum(occ.values())

    print(f"  Piece count — near (ranks 1-2): {near_side}  "
          f"far (ranks 7-8): {far_side}  total: {total}")

    if total != 32 or near_side != 16 or far_side != 16:
        print("  WARNING: expected 32 pieces (16 per side). Check board setup.")
        return False
    print("  Board looks correct.")
    return True


# ── Board corners ─────────────────────────────────────────────────────────────

def get_board_corners() -> np.ndarray:
    """Load board corners from disk or raise."""
    try:
        corners = np.load("board_calibration.npy").astype(np.float32)
        print("  Board corners loaded from board_calibration.npy")
        return corners
    except FileNotFoundError:
        print("ERROR: board_calibration.npy not found.")
        print("  Run: python chess_glass_detector.py → r → s → q")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ARIA Chess Robot")
    print("=" * 60)

    # ── Colour selection ──────────────────────────────────────────────────────
    while True:
        choice = input("\n  Your colour? [w=white / b=black]: ").strip().lower()
        if choice in ("w", "b"):
            break
        print("  Enter 'w' or 'b'.")
    human_color = chess.WHITE if choice == "w" else chess.BLACK
    robot_color = not human_color
    print(f"  You play {'WHITE' if human_color == chess.WHITE else 'BLACK'}.")

    # ── Engine strength ───────────────────────────────────────────────────────
    elo_str = input("  Engine ELO 100-3200 (Enter = full strength): ").strip()
    elo     = int(elo_str) if elo_str.isdigit() else None
    print(f"  Strength: {'ELO ' + elo_str if elo else 'full'}")

    # ── Load calibration ──────────────────────────────────────────────────────
    try:
        T = np.load(const.T_CAM_TO_ARM_FILE)
    except FileNotFoundError:
        print(f"ERROR: {const.T_CAM_TO_ARM_FILE} not found — run jog_cal.py first.")
        sys.exit(1)

    corners = get_board_corners()
    M       = cv2.getPerspectiveTransform(corners, DST_CORNERS)

    # ── Hardware init ─────────────────────────────────────────────────────────
    print("\nConnecting to camera...")
    device, pipeline, rgb_queue, intr = init_camera()
    print(f"  fx={intr['fx']:.1f}")

    print("Connecting to arm...")
    from uarm.wrapper import SwiftAPI
    from servo_pick   import ServoPicker
    arm = SwiftAPI(port=const.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    arm.set_position(x=const.HOME_X, y=const.HOME_Y, z=const.HOME_Z,
                     speed=const.ARM_SPEED_FAST, wait=True)
    sp  = ServoPicker(arm, pipeline, rgb_queue, intr, T, corners)

    print("Starting engine...")
    engine = Stockfish(depth=STOCKFISH_DEPTH, elo=elo)

    _open_debug_windows()

    try:
        # ── Empty-board baseline ──────────────────────────────────────────────
        import os
        empty_warped = None
        if os.path.exists(BASELINE_IMG):
            empty_warped = cv2.imread(BASELINE_IMG)
            print(f"  Loaded saved baseline from {BASELINE_IMG}  (skip board clear)")
            redo = input("  Re-take baseline? [y/N]: ").strip().lower()
            if redo == 'y':
                empty_warped = None

        if empty_warped is None:
            input("\n  CLEAR THE BOARD completely, then press Enter...")
            print("  Taking empty baseline...")
            frame = get_frame(pipeline, rgb_queue)
            while frame is None:
                frame = get_frame(pipeline, rgb_queue)
            empty_warped = warp(frame, M)
            cv2.imwrite(BASELINE_IMG, empty_warped)
            print(f"  Baseline captured and saved → {BASELINE_IMG}")

        # ── Piece setup ───────────────────────────────────────────────────────
        input("  Set up pieces in STARTING POSITION, then press Enter...")
        verify_starting_position(pipeline, rgb_queue, M, empty_warped, human_color,
                                 window="game")

        # ── Game state ────────────────────────────────────────────────────────
        board = chess.Board()   # standard starting position
        print("\n  Game started!\n")
        print(board)

        # ── Game loop ─────────────────────────────────────────────────────────
        while not board.is_game_over():
            turn = board.turn   # chess.WHITE or chess.BLACK
            print(f"\n{'─'*40}")
            print(f"  {'White' if turn == chess.WHITE else 'Black'} to move"
                  f"  ({'YOU' if turn == human_color else 'ROBOT'})")

            if turn == human_color:
                # ── Human turn ────────────────────────────────────────────────
                uci = observe_human_move(pipeline, rgb_queue, M, empty_warped,
                                         board, human_color, window="game")
                move = board.parse_uci(uci)
                san  = board.san(move)
                board.push(move)
                print(f"  Human played: {san}")

            else:
                # ── Robot turn ────────────────────────────────────────────────
                print("  Engine thinking...")
                uci = engine.best_move(board)
                if uci is None:
                    print("  Engine returned no move (game over?).")
                    break
                move = board.parse_uci(uci)
                san  = board.san(move)
                print(f"  Robot plays: {san}  ({uci})")
                execute_robot_move(uci, board, sp)
                board.push(move)
                print(f"  Done. Arm homing...")
                sp.home()
                time.sleep(1.5)   # let board lighting settle after arm clears

            print(board)

            frame = get_frame(pipeline, rgb_queue)
            if frame is not None:
                warped = warp(frame, M)
                cv2.imshow(_CAM_WIN, frame)
                _draw_board_debug(warped, "GAME LOOP")
                cv2.waitKey(1)

        # ── Game over ─────────────────────────────────────────────────────────
        print(f"\n{'═'*40}")
        result = board.result()
        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            print(f"  CHECKMATE — {winner} wins!  ({result})")
        elif board.is_stalemate():
            print("  STALEMATE — draw.")
        else:
            print(f"  Game over: {result}")

    except KeyboardInterrupt:
        print("\n  Game interrupted.")
    finally:
        print("  Shutting down...")
        try:
            sp.home()
        except Exception:
            pass
        try:
            engine.close()
        except Exception:
            pass
        try:
            device.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
