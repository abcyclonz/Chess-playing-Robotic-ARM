"""
aruco_pose_pnp.py
──────────────────
ArUco marker 6-DOF pose via cv2.solvePnP.

IMPORTANT: distortion coefficients are FORCED TO ZERO.

Why: OAK-D Lite stores factory distortion coefficients that are sized for
its internal stereo rectification pipeline, not for OpenCV's 5-parameter
Brown-Conrady model. Values like [-5.4, 17.2, -22.3] are 100× what any
real lens would produce — they're calibration artefacts from the stereo
module, not a distortion model.

Passing those raw coefficients to cv2.solvePnP produces pathological
pose estimates with ~100px reprojection errors.

In practice, OAK's delivered RGB stream is rectified enough that the
residual distortion is negligible for ArUco-sized features, so dist=zero
is the correct choice here. If you later need sub-mm accuracy at wide
angles, re-calibrate the camera intrinsics with cv2.calibrateCamera.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import constants

_aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
_aruco_params = aruco.DetectorParameters()
# CORNER_REFINE_SUBPIX is reliable across all OpenCV 4.x builds.
# CORNER_REFINE_APRILTAG requires optional compile-time support and its
# aprilTag* parameters silently conflict in OpenCV 4.7+ — avoid it.
_aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
# OpenCV 4.7+ removed the module-level detectMarkers; use ArucoDetector instead.
ARUCO_DICT   = _aruco_dict
ARUCO_PARAMS = _aruco_params
_DETECTOR    = aruco.ArucoDetector(_aruco_dict, _aruco_params)

_SIDE_M = constants.MARKER_BLACK_SIDE_MM / 1000.0
_HALF   = _SIDE_M / 2.0

_OBJECT_POINTS = np.array([
    [-_HALF,  _HALF, 0],
    [ _HALF,  _HALF, 0],
    [ _HALF, -_HALF, 0],
    [-_HALF, -_HALF, 0],
], dtype=np.float64)

# Hard-coded zero distortion. See module docstring.
_ZERO_DIST = np.zeros((5, 1), dtype=np.float64)

MAX_REPROJ_PX = 6.0  # loosened from 3.0 — top-down camera, no distortion correction
MIN_Z_M       = 0.08
MAX_Z_M       = 2.00
MIN_SIDE_PX   = 25.0


def _intrinsics_matrix(intr: dict) -> np.ndarray:
    return np.array([[intr['fx'], 0, intr['cx']],
                     [0, intr['fy'], intr['cy']],
                     [0, 0, 1]], dtype=np.float64)


def _mean_side_length(corners: np.ndarray) -> float:
    edges = [np.linalg.norm(corners[i] - corners[(i+1) % 4]) for i in range(4)]
    return float(np.mean(edges))


def _compute_reproj_error(object_pts, image_pts, rvec, tvec, cam_mtx) -> float:
    try:
        projected, _ = cv2.projectPoints(object_pts, rvec, tvec,
                                          cam_mtx, _ZERO_DIST)
        projected = projected.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(projected - image_pts, axis=1)))
    except Exception:
        return float('inf')


_last_reject_reason: str | None = None


def get_last_reject_reason() -> str | None:
    return _last_reject_reason


def detect_marker_3d(color_image: np.ndarray,
                      intrinsics: dict,
                      dist_coeffs=None,   # accepted for API compat, IGNORED
                      draw: bool = True) -> dict | None:
    """
    Detect marker, solve PnP with zero distortion, validate result.

    Returns dict or None. dist_coeffs argument is accepted for backward
    compatibility but IGNORED (forced to zero). See module docstring.
    """
    global _last_reject_reason
    _last_reject_reason = None

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        _last_reject_reason = "no_detection"
        return None

    image_points = corners[0][0].astype(np.float64)
    side_px      = _mean_side_length(image_points)

    if draw:
        aruco.drawDetectedMarkers(color_image, corners, ids)

    if side_px < MIN_SIDE_PX:
        _last_reject_reason = f"too_small side={side_px:.1f}<{MIN_SIDE_PX}"
        return None

    cam_mtx = _intrinsics_matrix(intrinsics)

    candidates = []

    try:
        n_sols, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            _OBJECT_POINTS, image_points, cam_mtx, _ZERO_DIST,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        for i in range(n_sols):
            tv = tvecs[i].flatten()
            rv = rvecs[i]
            rej = _compute_reproj_error(_OBJECT_POINTS, image_points,
                                         rv, tv, cam_mtx)
            candidates.append((rej, tv, rv, f"ippe#{i}"))
    except cv2.error:
        pass

    try:
        ok, rv, tv = cv2.solvePnP(
            _OBJECT_POINTS, image_points, cam_mtx, _ZERO_DIST,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            tv_flat = tv.flatten()
            rej = _compute_reproj_error(_OBJECT_POINTS, image_points,
                                         rv, tv_flat, cam_mtx)
            candidates.append((rej, tv_flat, rv, "iterative"))
    except cv2.error:
        pass

    best_sane       = None
    best_sane_rej   = float('inf')
    best_insane     = None
    best_insane_rej = float('inf')

    for rej, tv, rv, label in candidates:
        if tv is None or not np.all(np.isfinite(tv)):
            continue
        z = float(tv[2])
        reproj_ok = np.isfinite(rej) and rej < MAX_REPROJ_PX
        z_ok      = MIN_Z_M <= z <= MAX_Z_M
        if reproj_ok and z_ok:
            if rej < best_sane_rej:
                best_sane_rej = rej
                best_sane = (rej, tv, rv, label)
        else:
            if rej < best_insane_rej:
                best_insane_rej = rej
                best_insane = (rej, tv, rv, label, reproj_ok, z_ok)

    if best_sane is not None:
        rej, tv, rv, label = best_sane
        center_px = tuple(np.mean(image_points, axis=0).astype(int))
        if draw:
            try:
                cv2.drawFrameAxes(color_image, cam_mtx, _ZERO_DIST,
                                  rv, tv.reshape(3, 1), 0.02)
            except cv2.error:
                pass
        return {
            'tvec_m'       : tv,
            'rvec'         : rv,
            'center_px'    : center_px,
            'reproj_err_px': rej,
            'side_px'      : side_px,
            'solver_used'  : label,
        }

    if best_insane is None:
        _last_reject_reason = "all_solvers_failed"
    else:
        rej, tv, rv, label, reproj_ok, z_ok = best_insane
        z = float(tv[2])
        reasons = []
        if not reproj_ok: reasons.append(f"reproj={rej:.1f}>{MAX_REPROJ_PX}")
        if not z_ok:      reasons.append(f"z={z:.2f}m∉[{MIN_Z_M},{MAX_Z_M}]")
        _last_reject_reason = f"{label} " + "  ".join(reasons) + f"  side={side_px:.0f}px"
    return None


def detect_aruco_3d(color_image, depth_frame_np, intrinsics):
    r = detect_marker_3d(color_image, intrinsics, draw=True)
    if r is None:
        return None, None
    return tuple(r['tvec_m']), r['center_px']


# ---------------------------------------------------------------------------
# Standalone diagnostic
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from collections import deque
    from depth_utils import initialize_oak

    print("Initialising OAK-D Lite...")
    device, q_rgb, _q_depth, intrinsics = initialize_oak()

    print(f"\nIntrinsics: fx={intrinsics['fx']:.1f}  fy={intrinsics['fy']:.1f}  "
          f"cx={intrinsics['cx']:.1f}  cy={intrinsics['cy']:.1f}")
    print("Distortion: forced to zero (OAK factory values are unusable "
          "for solvePnP — see module docstring)\n")
    print(f"Marker black side: {constants.MARKER_BLACK_SIDE_MM:.1f} mm")
    print(f"Sanity gates: reproj<{MAX_REPROJ_PX}px, z∈[{MIN_Z_M},{MAX_Z_M}]m, "
          f"side≥{MIN_SIDE_PX}px\n")

    hist = deque(maxlen=30)
    try:
        while True:
            color = q_rgb.get().getCvFrame()
            r = detect_marker_3d(color, intrinsics)

            if r is not None:
                t_mm = r['tvec_m'] * 1000
                hist.append(t_mm)
                if len(hist) >= 5:
                    arr = np.array(hist)
                    std  = np.std(arr, axis=0)
                    mean = np.mean(arr, axis=0)
                    col = (0, 255, 0) if max(std) < 1.0 else (0, 165, 255)
                    cv2.putText(color,
                        f"X={mean[0]:+.1f} Y={mean[1]:+.1f} Z={mean[2]:.1f}mm "
                        f"[{r['solver_used']}]",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                    cv2.putText(color,
                        f"std=({std[0]:.2f},{std[1]:.2f},{std[2]:.2f})mm  "
                        f"reproj={r['reproj_err_px']:.2f}px  side={r['side_px']:.0f}px",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            else:
                hist.clear()
                reason = get_last_reject_reason() or "?"
                cv2.putText(color, "REJECTED", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(color, reason, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.imshow("aruco PnP diagnostic", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        device.close()
        cv2.destroyAllWindows()