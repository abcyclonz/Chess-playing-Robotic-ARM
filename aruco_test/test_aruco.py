#!/usr/bin/env python3
"""
test_aruco.py — ArUco marker detection test  (RGB-only, no stereo depth)

Keys:
  D   scan ALL common ArUco dictionaries and print which one sees your marker
  Q   quit
"""

import time
import numpy as np
import cv2
import cv2.aruco as aruco
import depthai as dai
from collections import deque
from aruco_pose_pnp import detect_marker_3d, get_last_reject_reason, ARUCO_DICT

# All dictionaries to try in the dict-scanner
SCAN_DICTS = {
    "7X7_250":  aruco.DICT_7X7_250,   # confirmed — this is the marker in use
    "7X7_50":   aruco.DICT_7X7_50,
    "7X7_100":  aruco.DICT_7X7_100,
    "4X4_50":   aruco.DICT_4X4_50,
    "4X4_100":  aruco.DICT_4X4_100,
    "5X5_50":   aruco.DICT_5X5_50,
    "5X5_100":  aruco.DICT_5X5_100,
    "6X6_50":   aruco.DICT_6X6_50,
    "6X6_100":  aruco.DICT_6X6_100,
    "ORIG":     aruco.DICT_ARUCO_ORIGINAL,
}


# ── Camera pipeline (RGB only) ────────────────────────────────────────────────

def create_rgb_pipeline(device: dai.Device):
    pipeline  = dai.Pipeline(device)
    cam       = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    try:
        cam.setIspNumFramesPool(1)
        cam.setOutputsNumFramesPool(1)
        cam.setRawNumFramesPool(1)
    except Exception:
        pass
    rgb_out   = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p, fps=15)
    rgb_queue = rgb_out.createOutputQueue(maxSize=1, blocking=False)
    return pipeline, rgb_queue


def get_intrinsics(device: dai.Device) -> dict:
    calib = device.readCalibration()
    M = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480))
    return {'fx': float(M[0][0]), 'fy': float(M[1][1]),
            'cx': float(M[0][2]), 'cy': float(M[1][2])}


# ── Dictionary scanner ────────────────────────────────────────────────────────

def scan_all_dicts(frame: np.ndarray):
    """Try every common ArUco dictionary and report which ones detect a marker."""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    found  = []
    for name, dict_id in SCAN_DICTS.items():
        d   = aruco.getPredefinedDictionary(dict_id)
        det = aruco.ArucoDetector(d, params)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            found.append((name, ids.flatten().tolist()))
    return found


def draw_raw_corners(frame: np.ndarray) -> np.ndarray:
    """Draw corners detected by the configured DICT_4X4_50 detector (no PnP needed)."""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    det    = aruco.ArucoDetector(ARUCO_DICT, params)
    corners, ids, rejected = det.detectMarkers(gray)
    vis = frame.copy()
    # Draw any detected markers
    if ids is not None:
        aruco.drawDetectedMarkers(vis, corners, ids)
    # Draw rejected candidates in purple (quads found but not matching the dict)
    for rq in rejected:
        pts = rq[0].astype(int)
        for i in range(4):
            cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1)%4]), (180, 0, 180), 1)
    return vis, (ids is not None and len(ids) > 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to OAK-D Lite (RGB only — no depth)...")
    device   = dai.Device()
    pipeline, rgb_queue = create_rgb_pipeline(device)

    intrinsics = get_intrinsics(device)
    print(f"  fx={intrinsics['fx']:.1f}  fy={intrinsics['fy']:.1f}  "
          f"cx={intrinsics['cx']:.1f}  cy={intrinsics['cy']:.1f}")
    print("  Marker: DICT_4X4_50 ID 0  |  44.0 mm  |  reproj threshold: 6.0 px")
    print("\n  D = scan ALL dicts (find which one your marker belongs to)")
    print("  Q = quit\n")

    pipeline.start()

    hist = deque(maxlen=30)

    while pipeline.isRunning():
        pkt = rgb_queue.tryGet()
        if pkt is None:
            time.sleep(0.005)
            continue

        frame = pkt.getCvFrame()
        key   = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        if key == ord('d') or key == ord('D'):
            print("\n--- Scanning all dictionaries ---")
            results = scan_all_dicts(frame)
            if results:
                for name, ids in results:
                    print(f"  FOUND  dict={name}  ids={ids}")
            else:
                print("  Nothing detected in any dictionary.")
                print("  Check: lighting, marker not in camera view, or marker too small.")
            print("---------------------------------\n")

        # Try full pose estimation
        r = detect_marker_3d(frame, intrinsics, draw=True)

        # Also draw raw corners + rejected candidates regardless of PnP result
        frame, raw_detected = draw_raw_corners(frame)

        if r is not None:
            t_mm = r['tvec_m'] * 1000.0
            hist.append(t_mm)
            arr  = np.array(hist)
            mean = np.mean(arr, axis=0)
            std  = np.std(arr, axis=0)
            col  = (0, 255, 0) if float(np.max(std)) < 1.0 else (0, 165, 255)

            cv2.putText(frame,
                f"X={mean[0]:+.1f} Y={mean[1]:+.1f} Z={mean[2]:.1f} mm  [{r['solver_used']}]",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(frame,
                f"std=({std[0]:.2f},{std[1]:.2f},{std[2]:.2f}) mm  "
                f"reproj={r['reproj_err_px']:.2f}px  side={r['side_px']:.0f}px",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

            if len(hist) >= 5:
                print(f"\rX={mean[0]:+7.1f}  Y={mean[1]:+7.1f}  Z={mean[2]:7.1f} mm"
                      f"  std=({std[0]:.2f},{std[1]:.2f},{std[2]:.2f})"
                      f"  reproj={r['reproj_err_px']:.2f}px", end="", flush=True)
        else:
            hist.clear()
            reason = get_last_reject_reason() or "unknown"

            if raw_detected:
                # Marker quad found but PnP failed — show in orange
                cv2.putText(frame, "MARKER FOUND — PnP failed",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                cv2.putText(frame, reason,
                            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1)
            else:
                cv2.putText(frame, "NO MARKER DETECTED",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(frame, f"{reason}  (D=scan dicts)",
                            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1)

        cv2.putText(frame, "D=scan dicts  Q=quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.imshow("ArUco Detection Test", frame)

    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
