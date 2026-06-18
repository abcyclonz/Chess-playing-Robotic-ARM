# chessbotV1 — Chess Piece Detection + Arm Pick Integration

This folder is the active development area for the ARIA chess robot.
It combines glass piece detection (RGB camera) with uARM Swift Pro picking.

---

## Hardware

- **Camera**: OAK-D Lite, top-down mount, ~32 cm above board, USB 2.0
- **Arm**: uARM Swift Pro, `/dev/ttyACM0`
- **Pieces**: cobalt-blue glass + clear glass on a printed paper board
- **ArUco marker**: DICT_7X7_250, ID 0, **30 mm** physical print, mounted on gripper facing upward
- **Extension**: 15 cm below the marker to the gripper tip — never command Z where tip would collide

---

## File Structure

```
chessbotV1/
├── CLAUDE.md                     ← this file
├── constants.py                  ← ALL tunable parameters
├── aruco_pose_pnp.py             ← ArUco 6-DOF pose via solvePnP (DICT_7X7_250)
├── jog_cal.py                    ← MAIN CALIBRATION: interactive jog + capture
├── calibrate_topdown.py          ← (legacy) automated 34-pose calibration
├── chess_glass_detector.py       ← live glass piece detection + board corner save
├── calibrate_pieces.py           ← interactive piece calibration tool
├── piece_calibration.json        ← saved piece detection thresholds
├── board_calibration.npy         ← 4 board corner pixels saved by chess_glass_detector.py
├── T_cam_to_arm_topdown.npy      ← camera→arm matrix (output of jog_cal.py)
├── servo_pick.py                 ← MAIN: visual-servo pick/place pipeline
├── chessgame.py                  ← full game loop (Stockfish vs human)
├── debug_board.py                ← real-time square diff visualiser (tuning tool)
├── board_to_arm.py               ← (legacy) static square→arm mapping — not used
│
└── aruco_test/                   ← standalone ArUco test scripts
    ├── test_aruco.py             ← live detection viewer (D=scan dicts, Q=quit)
    ├── aruco_pose_pnp.py         ← copy that imports constants from parent
    └── constants.py              ← minimal (imports from ../constants.py)
```

---

## Key Constants (`constants.py`)

| Constant | Value | Notes |
|---|---|---|
| `PORT` | `/dev/ttyACM0` | Arm serial port |
| `TABLE_Z` | `48.0` | Arm Z when gripper tip touches board surface |
| `HOME_X/Y/Z` | `150, -185, 120` | Park position, off-board, out of camera FOV |
| `TRANSIT_Z` | `120.0` | Arm Z for all lateral moves — ext tip 72mm above board |
| `MARKER_BLACK_SIDE_MM` | `30.0` | Physical print size of ArUco marker |
| `MARKER_OFFSET_X/Y/Z` | `0, 30, 130` | Arm cmd position → marker centre offset (mm) |
| `SERVO_OFFSET_X/Y` | `-10, 0` | Shift servo target to align gripper tip, not marker, to piece |
| `T_CAM_TO_ARM_FILE` | `T_cam_to_arm_topdown.npy` | Transform matrix path |

---

## Step-by-Step Setup (every session)

### 1. Verify ArUco detection
```bash
cd aruco_test && python test_aruco.py
# D = scan all dicts  |  Q = quit
```
Expected: green text, std < 1 mm, reproj < 1 px.

### 2. Calibrate camera→arm transform
```bash
python -u jog_cal.py
```
- Remove all chess pieces first; marker must face upward on gripper
- Jog the arm (W/S/A/D/R/F) until marker turns GREEN in the window
- Press SPACE to capture; collect 6–9 spread positions (vary X, Y, and at least 2 Z heights)
- Press Enter to solve and save `T_cam_to_arm_topdown.npy`
- Target: mean residual < 5 mm (last run: **2.97 mm, 9 poses**)

**Why jog_cal.py instead of calibrate_topdown.py:**
The fixed poses in the automated script may not be in the camera's visible zone
after the hardware is repositioned. Interactive jogging always finds the visible zone.

### 3. Save board corners
```bash
python chess_glass_detector.py
# r = mark 4 board corners  |  s = save board_calibration.npy  |  q = quit
```
Re-run whenever the board is moved.

### 4. Visual-servo pick/place
```bash
python -u servo_pick.py e2 e4        # full move
python -u servo_pick.py --pick e2    # pick only
python -u servo_pick.py --place e4   # place only (piece already held)
python -u servo_pick.py --home       # home arm
```

**How servo_pick.py works:**
1. Loads `board_calibration.npy` (4 corner pixels) and `T_cam_to_arm_topdown.npy`
2. Rough move: projects target square pixel → arm XY via T_cam_to_arm at TRANSIT_Z
3. Servo loop: reads ArUco marker pose each frame, nudges arm by `error × 0.6`
   — `SERVO_OFFSET_X/Y` shifts the target so the **gripper tip** (not marker) lands on the piece
4. When XY error < 3 mm: dive → grip (or release) → lift to TRANSIT_Z
5. If marker leaves FOV for 8+ frames: auto-resets arm to computed rough position

### 5. Full game (optional)
```bash
python -u chessgame.py
```

---

## Tuning pick accuracy (`SERVO_OFFSET_X/Y`)

If the arm consistently picks off-centre in a fixed direction, adjust in `constants.py`:

| Symptom | Fix |
|---|---|
| Picks too far forward (away from arm base) | decrease `SERVO_OFFSET_X` (more negative) |
| Picks too far back (toward arm base) | increase `SERVO_OFFSET_X` (less negative / positive) |
| Picks too far left/right | adjust `SERVO_OFFSET_Y` |

No recalibration needed — just edit the constant and re-test.

---

## ArUco Notes

- **Dictionary**: `cv2.aruco.DICT_7X7_250` (confirmed by scanning)
- **Corner refinement**: `CORNER_REFINE_SUBPIX` — do NOT use `CORNER_REFINE_APRILTAG`
  (aprilTag* parameters conflict in OpenCV 4.7+)
- **Reprojection threshold**: 6.0 px
- **Distortion**: forced to zero — OAK factory coefficients are for stereo rectification
  and cause ~100 px reprojection errors in solvePnP if used

## Camera Pipeline Notes

- **RGB-only** — no stereo depth (adding depth at 30fps overflows USB 2.0)
- All pipelines: pool sizes = 1, FPS = 15, RGB only
- `setIspNumFramesPool(1)`, `setOutputsNumFramesPool(1)`, `setRawNumFramesPool(1)` — all three required

## Z Height Reference

```
arm Z   ext tip above board   note
──────  ────────────────────  ────────────────────────────────
 120       72 mm              TRANSIT_Z — safe lateral moves
  83       35 mm              HOVER_Z — just above piece tops (~30 mm)
  63       15 mm              GRIP_Z — mid-piece grab
  58       10 mm              PLACE_Z — release height
  48        0 mm              TABLE_Z — tip touches board
```
Never command Z < 90 during calibration (extension collision risk).

---

## Do NOT Modify

- `aruco_test/test_aruco.py` key bindings (D/Q)
- ArUco dictionary — confirmed as DICT_7X7_250 ID 0

## Status

- [x] ArUco detection — DICT_7X7_250, 30 mm, sub-mm stable
- [x] Camera→arm calibration — 2.97 mm mean residual (9 poses, jog_cal.py)
- [x] Board corners saved — `board_calibration.npy`
- [x] Visual-servo pick/place — working, gripper offset tuned (`SERVO_OFFSET_X = -10`)
- [ ] Full game test: `python -u chessgame.py`
