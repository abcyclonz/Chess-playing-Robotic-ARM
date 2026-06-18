#!/usr/bin/env python3 -u
"""
jog_cal.py — Interactive camera→arm calibration with keyboard arm jogging.

Jog the arm until the ArUco marker is GREEN in the camera window,
then press SPACE to capture that point.  Collect 6+ points, press Enter to solve.

Controls
────────
  W / S       +X / -X  (arm forward / back)       10 mm
  A / D       +Y / -Y  (arm left / right)          10 mm
  R / F       +Z / -Z  (arm up / down)              5 mm
  Shift+above           large step                 30 / 30 / 15 mm
  SPACE       capture current position (only when marker detected)
  Enter       solve & save  T_cam_to_arm_topdown.npy  (need ≥ 4 points)
  H           home arm
  Q / Esc     quit without saving

Tips
────
  • Marker must face UPWARD on the gripper
  • Board must be in playing position (empty — no pieces)
  • Move the arm so the marker appears in the camera window (goes GREEN)
  • Spread captures across different X, Y, Z positions for accuracy
  • Aim for at least 2 different heights
"""

import sys, time, cv2, numpy as np
import depthai as dai
import constants as const
from aruco_pose_pnp import detect_marker_3d, get_last_reject_reason

MIN_POSES     = 4      # minimum captured poses to allow solving
CAPTURE_N     = 10     # frames to median-average per capture
CAPTURE_T     = 2.0    # max seconds to collect those frames


# ── Camera init ───────────────────────────────────────────────────────────────

def init_camera():
    device   = dai.Device()
    pipeline = dai.Pipeline(device)
    cam      = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    try:
        cam.setIspNumFramesPool(1)
        cam.setOutputsNumFramesPool(1)
    except Exception:
        pass
    try:
        cam.setRawNumFramesPool(1)
    except Exception:
        pass
    out  = cam.requestOutput((const.CAM_W, const.CAM_H),
                              type=dai.ImgFrame.Type.BGR888p, fps=15)
    rgb_q = out.createOutputQueue(maxSize=1, blocking=False)
    calib = device.readCalibration()
    M     = np.array(calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, const.CAM_W, const.CAM_H))
    intr  = {'fx': float(M[0][0]), 'fy': float(M[1][1]),
             'cx': float(M[0][2]), 'cy': float(M[1][2])}
    pipeline.start()
    return device, pipeline, rgb_q, intr


def poll(pipeline, rgb_q):
    if not pipeline.isRunning():
        return None
    try:
        pkt = rgb_q.tryGet()
        return pkt.getCvFrame() if pkt is not None else None
    except Exception:
        return None


# ── Kabsch SVD rigid transform ────────────────────────────────────────────────

def solve_rigid(cam_mm, arm_mm):
    sc = cam_mm.mean(0);  tc = arm_mm.mean(0)
    H  = (cam_mm - sc).T @ (arm_mm - tc)
    U, _, Vt = np.linalg.svd(H)
    d  = np.sign(np.linalg.det(Vt.T @ U.T))
    R  = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t  = tc - R @ sc
    return R, t


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  jog_cal.py — Interactive calibration")
    print("=" * 60)
    print("  W/S=X  A/D=Y  R/F=Z  (Shift=large)  SPACE=capture  Enter=save  Q=quit")
    print()

    cv2.namedWindow("jog_cal", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("jog_cal", 800, 600)

    print("Connecting to camera...")
    device, pipeline, rgb_q, intr = init_camera()
    print(f"  fx={intr['fx']:.1f}")

    print("Connecting to arm...")
    from uarm.wrapper import SwiftAPI
    arm = SwiftAPI(port=const.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    arm.set_position(x=const.HOME_X, y=const.HOME_Y, z=const.HOME_Z,
                     speed=const.ARM_SPEED_FAST, wait=True)
    cx, cy, cz = float(const.HOME_X), float(const.HOME_Y), float(const.HOME_Z)
    print(f"  Arm at HOME ({cx:.0f},{cy:.0f},{cz:.0f})")
    print()
    print("  Jog the arm until the marker turns GREEN, then press SPACE.")
    print(f"  Need {MIN_POSES} captures at different positions.  Press Enter when done.")
    print()

    cam_pts: list[np.ndarray] = []   # camera-frame tvec (metres)
    arm_pts: list[np.ndarray] = []   # arm-frame XYZ (mm)
    last_tvec  = None

    STEP_S = 10.0;  STEP_L = 30.0   # XY step sizes (mm)
    STEP_ZS = 5.0;  STEP_ZL = 15.0  # Z step sizes (mm)

    try:
        while True:
            frame = poll(pipeline, rgb_q)
            if frame is None:
                if not pipeline.isRunning():
                    print("\nCamera pipeline stopped (X_LINK_ERROR?) — reconnect USB and restart.")
                    break
                time.sleep(0.02)
                cv2.waitKey(1)
                continue

            r = detect_marker_3d(frame, intr, draw=True)

            if r is not None:
                last_tvec = r['tvec_m'].copy()
                txt   = (f"DETECTED  z={r['tvec_m'][2]*1000:.0f}mm  "
                         f"reproj={r['reproj_err_px']:.1f}px  "
                         f"captured={len(cam_pts)}  SPACE=capture")
                col   = (0, 255, 60)
            else:
                last_tvec = None
                reason = get_last_reject_reason() or "no_detection"
                txt    = (f"NO MARKER ({reason})  "
                          f"arm=({cx:.0f},{cy:.0f},{cz:.0f})  "
                          f"captured={len(cam_pts)}")
                col    = (0, 60, 255)

            cv2.putText(frame, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
            cv2.putText(frame,
                        "W/S=X  A/D=Y  R/F=Z  Shift=large  "
                        "SPACE=capture  Enter=solve  H=home  Q=quit",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

            cv2.imshow("jog_cal", frame)
            key = cv2.waitKey(30) & 0xFF

            # ── Quit ──────────────────────────────────────────────────────────
            if key in (ord('q'), 27):
                print("Quit — nothing saved.")
                break

            # ── Home ──────────────────────────────────────────────────────────
            elif key == ord('h'):
                cx, cy, cz = float(const.HOME_X), float(const.HOME_Y), float(const.HOME_Z)
                arm.set_position(x=cx, y=cy, z=cz,
                                 speed=const.ARM_SPEED_FAST, wait=True)
                print(f"  Homed ({cx:.0f},{cy:.0f},{cz:.0f})")

            # ── Capture ───────────────────────────────────────────────────────
            elif key == ord(' '):
                if last_tvec is None:
                    print("  No marker — jog until green, then SPACE")
                    continue
                # Collect several frames for a stable median
                samples = [last_tvec.copy()]
                t0 = time.monotonic()
                while len(samples) < CAPTURE_N and time.monotonic() - t0 < CAPTURE_T:
                    fr2 = poll(pipeline, rgb_q)
                    if fr2 is not None:
                        r2 = detect_marker_3d(fr2, intr, draw=False)
                        if r2 is not None:
                            samples.append(r2['tvec_m'].copy())
                    time.sleep(0.05)
                if len(samples) < 3:
                    print("  Too few samples — hold still and try again")
                    continue
                tvec_med = np.median(samples, axis=0)
                std_mm   = float(np.std(np.linalg.norm(
                               np.array(samples) - tvec_med, axis=1)) * 1000)
                arm_xyz  = np.array([
                    cx + const.MARKER_OFFSET_X,
                    cy + const.MARKER_OFFSET_Y,
                    cz + const.MARKER_OFFSET_Z,
                ], dtype=np.float64)
                cam_pts.append(tvec_med)
                arm_pts.append(arm_xyz)
                print(f"  #{len(cam_pts)} captured  arm=({cx:.0f},{cy:.0f},{cz:.0f})"
                      f"  cam_z={tvec_med[2]*1000:.0f}mm  std={std_mm:.1f}mm"
                      f"  (n={len(samples)})")

            # ── Solve & save ──────────────────────────────────────────────────
            elif key == 13:   # Enter
                n = len(cam_pts)
                if n < MIN_POSES:
                    print(f"  Need {MIN_POSES} poses, have {n}.  Keep capturing.")
                    continue
                cam_arr = np.array(cam_pts, dtype=np.float64) * 1000.0   # m→mm
                arm_arr = np.array(arm_pts, dtype=np.float64)
                R, t    = solve_rigid(cam_arr, arm_arr)
                res     = np.linalg.norm((R @ cam_arr.T).T + t - arm_arr, axis=1)
                T4      = np.eye(4)
                T4[:3, :3] = R
                T4[:3,  3] = t
                np.save(const.T_CAM_TO_ARM_FILE, T4)
                print(f"\n  Solved with {n} poses")
                print(f"  Residuals: mean={res.mean():.2f}mm  max={res.max():.2f}mm")
                if res.mean() > 8:
                    print("  WARNING: residual > 8mm — check MARKER_OFFSET_X/Y/Z")
                elif res.mean() > 5:
                    print("  NOTE: > 5mm — usable, try more spread poses for better accuracy")
                else:
                    print("  Accuracy: good")
                print(f"  Saved → {const.T_CAM_TO_ARM_FILE}")
                break

            # ── Jog ──────────────────────────────────────────────────────────
            else:
                dx = dy = dz = 0.0
                if   key == ord('w'): dx = +STEP_S
                elif key == ord('s'): dx = -STEP_S
                elif key == ord('a'): dy = +STEP_S
                elif key == ord('d'): dy = -STEP_S
                elif key == ord('r'): dz = +STEP_ZS
                elif key == ord('f'): dz = -STEP_ZS
                elif key == ord('W'): dx = +STEP_L
                elif key == ord('S'): dx = -STEP_L
                elif key == ord('A'): dy = +STEP_L
                elif key == ord('D'): dy = -STEP_L
                elif key == ord('R'): dz = +STEP_ZL
                elif key == ord('F'): dz = -STEP_ZL

                if dx or dy or dz:
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    r_xy = (nx**2 + ny**2) ** 0.5
                    if r_xy < const.MIN_RADIUS:
                        scale = const.MIN_RADIUS / r_xy
                        nx, ny = nx * scale, ny * scale
                    elif r_xy > const.MAX_RADIUS:
                        scale = const.MAX_RADIUS / r_xy
                        nx, ny = nx * scale, ny * scale
                    cx, cy, cz = nx, ny, nz
                    arm.set_position(x=cx, y=cy, z=cz,
                                     speed=const.ARM_SPEED, wait=True)
                    print(f"\r  arm=({cx:+.0f},{cy:+.0f},{cz:.0f})"
                          f"  r={r_xy:.0f}mm        ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        try:
            arm.set_position(x=const.HOME_X, y=const.HOME_Y, z=const.HOME_Z,
                             speed=const.ARM_SPEED_FAST, wait=True)
        except Exception:
            pass
        try:
            device.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
