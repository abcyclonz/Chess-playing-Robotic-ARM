#!/usr/bin/env python3 -u
"""
servo_pick.py — Visual-servo pick/place for chessbotV1
═══════════════════════════════════════════════════════
No pre-computed square coordinates.  Every pick/place works like this:

  1. Detect target square centre in camera frame (real-time board detection)
  2. Move arm to rough position above it (T_cam_to_arm plane projection)
  3. Servo loop — camera sees BOTH the ArUco gripper marker AND the target
     square.  Arm nudges until they align.
  4. Dive → grip → lift  (or lower → release → lift for place)

Works even if the board or arm shifts between moves.

CLI quick-test
──────────────
  python -u servo_pick.py e2 e4        # full move
  python -u servo_pick.py --pick  e2   # pick only
  python -u servo_pick.py --place e4   # place only (gripper already holding)
  python -u servo_pick.py --home       # just home the arm
"""

import sys, time, argparse
import numpy as np
import cv2
import depthai as dai

import constants as const
from aruco_pose_pnp import detect_marker_3d, get_last_reject_reason

# ── Servo tuning ──────────────────────────────────────────────────────────────
SERVO_GAIN        = 0.6    # fraction of error to correct each step (0<g<1)
SERVO_THRESH_MM   = 3.0    # stop when XY error < this
SERVO_MAX_ITERS   = 30     # give up after this many servo steps
SERVO_SETTLE_S    = 0.20   # wait after each arm nudge before re-detecting
SERVO_Z           = 120.0  # arm Z during servo — ext tip 72mm above board, clears all pieces


# ── Camera pipeline ───────────────────────────────────────────────────────────

def init_camera():
    device   = dai.Device()
    pipeline = dai.Pipeline(device)
    cam      = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    try:
        cam.setIspNumFramesPool(1)
        cam.setOutputsNumFramesPool(1)
        cam.setRawNumFramesPool(1)
    except Exception:
        pass
    out       = cam.requestOutput((const.CAM_W, const.CAM_H),
                                   type=dai.ImgFrame.Type.BGR888p, fps=15)
    rgb_queue = out.createOutputQueue(maxSize=1, blocking=False)

    calib = device.readCalibration()
    M     = np.array(calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, const.CAM_W, const.CAM_H))
    intr  = {'fx': float(M[0][0]), 'fy': float(M[1][1]),
             'cx': float(M[0][2]), 'cy': float(M[1][2])}

    pipeline.start()
    return device, pipeline, rgb_queue, intr


def get_frame(pipeline, rgb_queue, timeout=1.0):
    deadline = time.monotonic() + timeout
    while pipeline.isRunning() and time.monotonic() < deadline:
        pkt = rgb_queue.tryGet()
        if pkt is not None:
            return pkt.getCvFrame()
        time.sleep(0.005)
    return None


# ── Board detection helpers ───────────────────────────────────────────────────

def load_board_corners(path="board_calibration.npy"):
    """Load saved board corners (4×2 float32, in camera pixel space)."""
    try:
        return np.load(path).astype(np.float32)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Board calibration not found: {path}\n"
            "  Run: python -u chess_glass_detector.py  →  r  →  s  →  q"
        )


def square_center_pixel(col: int, row: int, src_corners: np.ndarray) -> tuple[float, float]:
    """
    Return (u, v) pixel coords of the square centre in the original camera frame.
    Uses inverse perspective from the saved board corner calibration.
    """
    BOARD_PX = 640
    SQ       = BOARD_PX // 8
    DST      = np.float32([[0,0],[BOARD_PX,0],[BOARD_PX,BOARD_PX],[0,BOARD_PX]])
    M        = cv2.getPerspectiveTransform(src_corners, DST)
    M_inv    = np.linalg.inv(M)
    u_w      = col * SQ + SQ // 2
    v_w      = row * SQ + SQ // 2
    pt       = cv2.perspectiveTransform(
                   np.array([[[float(u_w), float(v_w)]]], dtype=np.float32), M_inv)
    return float(pt[0, 0, 0]), float(pt[0, 0, 1])


def pixel_to_arm(u: float, v: float, T: np.ndarray, intr: dict,
                 arm_z: float) -> np.ndarray:
    """
    Back-project pixel (u,v) onto the horizontal plane at arm Z = arm_z.
    Returns arm XYZ in mm.
    """
    R, t = T[:3, :3], T[:3, 3]
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    du = (u - cx) / fx
    dv = (v - cy) / fy
    Z_cam = (arm_z - t[2]) / (R[2,0]*du + R[2,1]*dv + R[2,2])
    pt_cam = np.array([du*Z_cam, dv*Z_cam, Z_cam, 1.0])
    return (T @ pt_cam)[:3]


# ── Visual servo ──────────────────────────────────────────────────────────────

def servo_align(arm, pipeline, rgb_queue, intr, T,
                target_u: float, target_v: float,
                servo_z: float = SERVO_Z,
                window: str | None = "servo") -> bool:
    """
    Servo the arm until the ArUco gripper marker is directly above target_u,v.

    Both target and marker positions are expressed in arm-frame mm via T, so
    the XY error vector is directly usable as an arm nudge command.

    Returns True if aligned within SERVO_THRESH_MM, False on timeout.
    """
    # Target position in arm frame (board-surface plane) — also used as reset point
    # SERVO_OFFSET shifts the marker target so the gripper tip lands on the piece centre.
    target_arm    = pixel_to_arm(target_u, target_v, T, intr, const.TABLE_Z)
    target_arm[0] += const.SERVO_OFFSET_X
    target_arm[1] += const.SERVO_OFFSET_Y
    no_mkr_streak = 0   # consecutive frames with no marker

    for it in range(SERVO_MAX_ITERS):
        frame = get_frame(pipeline, rgb_queue)
        if frame is None:
            continue

        r = detect_marker_3d(frame, intr, draw=True)
        if r is None:
            no_mkr_streak += 1
            if window:
                reason = get_last_reject_reason() or "?"
                cv2.putText(frame, f"NO MARKER [{no_mkr_streak}] — {reason}  iter={it}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.imshow(window, frame)
                cv2.waitKey(1)

            if no_mkr_streak >= 8:
                # Marker drifted out of camera FOV — reset to the computed rough position
                print(f"  servo: marker out of view for {no_mkr_streak} frames, "
                      f"resetting to rough ({target_arm[0]:+.0f},{target_arm[1]:+.0f})")
                arm.set_position(x=target_arm[0], y=target_arm[1], z=servo_z,
                                 speed=const.ARM_SPEED, wait=True)
                time.sleep(0.5)
                no_mkr_streak = 0
            else:
                time.sleep(0.05)
            continue

        no_mkr_streak = 0

        # Transform marker from camera frame (metres) → arm frame (mm)
        marker_cam = r['tvec_m'] * 1000.0
        marker_arm = (T @ np.append(marker_cam, 1.0))[:3]

        # XY error in arm frame — directly usable as arm nudge
        err_arm  = target_arm[:2] - marker_arm[:2]
        err_norm = float(np.linalg.norm(err_arm))

        if window:
            cv2.putText(frame,
                f"err=({err_arm[0]:+.1f},{err_arm[1]:+.1f}) {err_norm:.1f}mm  iter={it}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0,255,120) if err_norm < SERVO_THRESH_MM else (0,165,255), 2)
            cv2.imshow(window, frame)
            cv2.waitKey(1)

        if err_norm < SERVO_THRESH_MM:
            print(f"  servo OK  err={err_norm:.1f}mm  iters={it+1}")
            return True

        # Nudge arm toward target
        cur = _safe_pos(arm)
        nx  = cur[0] + err_arm[0] * SERVO_GAIN
        ny  = cur[1] + err_arm[1] * SERVO_GAIN
        arm.set_position(x=nx, y=ny, z=servo_z,
                         speed=const.ARM_SPEED, wait=True)
        time.sleep(SERVO_SETTLE_S)

    print(f"  servo TIMEOUT after {SERVO_MAX_ITERS} iters")
    return False


# ── Arm helpers ───────────────────────────────────────────────────────────────

def _safe_pos(arm) -> tuple[float, float, float]:
    """Read arm XYZ, stripping any trailing non-numeric chars from firmware."""
    pos = arm.get_position()
    return (float(str(pos[0]).strip().rstrip('M')),
            float(str(pos[1]).strip().rstrip('M')),
            float(str(pos[2]).strip().rstrip('M')))


# ── Pick / place primitives ───────────────────────────────────────────────────

class ServoPicker:
    """
    Visual-servo pick and place.

    Parameters
    ----------
    arm      : SwiftAPI instance (connected, mode 0)
    pipeline, rgb_queue, intr : from init_camera()
    T        : 4×4 T_cam_to_arm matrix
    corners  : board corner pixel coords (4×2) from board_calibration.npy
    """

    def __init__(self, arm, pipeline, rgb_queue, intr, T, corners):
        self.arm      = arm
        self.pipeline = pipeline
        self.queue    = rgb_queue
        self.intr     = intr
        self.T        = T
        self.corners  = corners

    def pick(self, col: int, row: int) -> bool:
        label = _sq(col, row)
        print(f"[pick] {label}")

        u, v  = square_center_pixel(col, row, self.corners)
        rough = pixel_to_arm(u, v, self.T, self.intr, const.TABLE_Z)
        print(f"  rough arm=({rough[0]:+.0f},{rough[1]:+.0f})")

        # 1. Open gripper
        self.arm.set_gripper(catch=False, wait=True)
        time.sleep(const.GRIPPER_OPEN_DELAY)

        # 2. Rise to TRANSIT_Z before any lateral movement
        cur = _safe_pos(self.arm)
        self.arm.set_position(x=cur[0], y=cur[1], z=const.TRANSIT_Z,
                               speed=const.ARM_SPEED, wait=True)
        time.sleep(0.3)

        # 3. Lateral move to rough position at SERVO_Z
        self.arm.set_position(x=rough[0], y=rough[1], z=SERVO_Z,
                               speed=const.ARM_SPEED_FAST, wait=True)
        time.sleep(0.3)

        # 4. Visual servo alignment
        aligned = servo_align(self.arm, self.pipeline, self.queue,
                               self.intr, self.T, u, v, servo_z=SERVO_Z)
        if not aligned:
            print(f"  [warn] servo timed out — proceeding open-loop")

        # 5. Read final aligned XY
        cur = _safe_pos(self.arm)
        cx, cy = cur[0], cur[1]

        # 6. Lower to hover height (just above piece tops)
        hover_z = const.TABLE_Z + const.HOVER_Z
        self.arm.set_position(x=cx, y=cy, z=hover_z,
                               speed=const.ARM_SPEED_SLOW, wait=True)
        time.sleep(const.SETTLE_S)

        # 7. Dive to grip height (mid-piece)
        grip_z = const.TABLE_Z + const.GRIP_Z_OFFSET
        self.arm.set_position(x=cx, y=cy, z=grip_z,
                               speed=const.ARM_SPEED_SLOW, wait=True)
        time.sleep(0.2)

        # 8. Close gripper and wait for full close
        self.arm.set_gripper(catch=True, wait=True)
        time.sleep(const.PICK_DELAY)

        # 9. Rise vertically to TRANSIT_Z before any lateral movement
        self.arm.set_position(x=cx, y=cy, z=const.TRANSIT_Z,
                               speed=const.ARM_SPEED_SLOW, wait=True)
        time.sleep(0.3)

        print(f"  picked {label}")
        return True

    def place(self, col: int, row: int) -> bool:
        label = _sq(col, row)
        print(f"[place] {label}")

        u, v  = square_center_pixel(col, row, self.corners)
        rough = pixel_to_arm(u, v, self.T, self.intr, const.TABLE_Z)

        # 1. Lateral move to destination area at TRANSIT_Z (arm already high from pick)
        self.arm.set_position(x=rough[0], y=rough[1], z=const.TRANSIT_Z,
                               speed=const.ARM_SPEED_FAST, wait=True)
        time.sleep(0.3)

        # 2. Drop to SERVO_Z for alignment (same height, or slight drop if SERVO_Z < TRANSIT_Z)
        self.arm.set_position(x=rough[0], y=rough[1], z=SERVO_Z,
                               speed=const.ARM_SPEED, wait=True)
        time.sleep(0.3)

        # 3. Visual servo alignment
        aligned = servo_align(self.arm, self.pipeline, self.queue,
                               self.intr, self.T, u, v, servo_z=SERVO_Z)
        if not aligned:
            print(f"  [warn] servo timed out — proceeding open-loop")

        # 4. Read final aligned XY
        cur = _safe_pos(self.arm)
        cx, cy = cur[0], cur[1]

        # 5. Lower to hover height
        hover_z = const.TABLE_Z + const.HOVER_Z
        self.arm.set_position(x=cx, y=cy, z=hover_z,
                               speed=const.ARM_SPEED_SLOW, wait=True)
        time.sleep(const.SETTLE_S)

        # 6. Lower to place height
        place_z = const.TABLE_Z + const.PLACE_Z_OFFSET
        self.arm.set_position(x=cx, y=cy, z=place_z,
                               speed=const.ARM_SPEED_SLOW, wait=True)
        time.sleep(0.2)

        # 7. Open gripper and wait for full release
        self.arm.set_gripper(catch=False, wait=True)
        time.sleep(const.PLACE_DELAY + const.GRIPPER_OPEN_DELAY)

        # 8. Rise vertically to TRANSIT_Z
        self.arm.set_position(x=cx, y=cy, z=const.TRANSIT_Z,
                               speed=const.ARM_SPEED_SLOW, wait=True)
        time.sleep(0.3)

        print(f"  placed {label}")
        return True

    def remove_to_bin(self, col: int, row: int) -> bool:
        if not self.pick(col, row):
            return False
        bx, by = const.CAPTURE_BIN_X, const.CAPTURE_BIN_Y

        self.arm.set_position(x=bx, y=by, z=const.TRANSIT_Z,
                               speed=const.ARM_SPEED_FAST, wait=True)
        time.sleep(0.3)

        hover_z = const.TABLE_Z + const.HOVER_Z
        self.arm.set_position(x=bx, y=by, z=hover_z,
                               speed=const.ARM_SPEED_SLOW, wait=True)
        time.sleep(const.SETTLE_S)

        place_z = const.TABLE_Z + const.PLACE_Z_OFFSET
        self.arm.set_position(x=bx, y=by, z=place_z,
                               speed=const.ARM_SPEED_SLOW, wait=True)
        time.sleep(0.2)

        self.arm.set_gripper(catch=False, wait=True)
        time.sleep(const.PLACE_DELAY + const.GRIPPER_OPEN_DELAY)
        self.arm.set_position(x=bx, y=by, z=const.TRANSIT_Z,
                               speed=const.ARM_SPEED_SLOW, wait=True)
        time.sleep(0.3)
        return True

    def home(self):
        self.arm.set_position(x=const.HOME_X, y=const.HOME_Y, z=const.HOME_Z,
                               speed=const.ARM_SPEED_FAST, wait=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sq(col: int, row: int) -> str:
    return "abcdefgh"[col] + str(row + 1)

def _parse_square(sq: str) -> tuple[int, int]:
    sq = sq.strip().lower()
    return ord(sq[0]) - ord('a'), int(sq[1]) - 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('squares', nargs='*',
                    help="Two squares for a move (e2 e4) or one UCI string (e2e4)")
    ap.add_argument('--pick',  metavar='SQ')
    ap.add_argument('--place', metavar='SQ')
    ap.add_argument('--home',  action='store_true')
    args = ap.parse_args()

    print("Loading board corners...")
    corners = load_board_corners()
    print("Loading T_cam_to_arm...")
    T = np.load(const.T_CAM_TO_ARM_FILE)

    print("Connecting to camera...")
    device, pipeline, rgb_queue, intr = init_camera()
    print(f"  fx={intr['fx']:.1f}  fy={intr['fy']:.1f}")

    print("Connecting to arm...")
    from uarm.wrapper import SwiftAPI
    arm = SwiftAPI(port=const.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    print("  Arm connected.")

    cv2.namedWindow("servo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("servo", 640, 480)

    sp = ServoPicker(arm, pipeline, rgb_queue, intr, T, corners)

    try:
        if args.home:
            sp.home()

        elif args.pick:
            sp.pick(*_parse_square(args.pick))

        elif args.place:
            sp.place(*_parse_square(args.place))

        elif len(args.squares) == 1:
            uci = args.squares[0]
            sp.pick(*_parse_square(uci[:2]))
            sp.place(*_parse_square(uci[2:4]))

        elif len(args.squares) == 2:
            sp.pick(*_parse_square(args.squares[0]))
            sp.place(*_parse_square(args.squares[1]))

        else:
            ap.print_help()

    finally:
        sp.home()
        try:
            device.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
