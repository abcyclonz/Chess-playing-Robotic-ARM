# ARIA Chess Robot — chessbotV1

Autonomous chess robot using a top-down OAK-D Lite camera and uARM Swift Pro arm.
Human plays cobalt-blue glass pieces. Robot plays clear glass pieces.
Move detection is done with HSV blue-piece tracking (immune to auto-exposure drift).
Pick/place uses visual servoing via an ArUco marker mounted on the gripper.

---

## Hardware

| Part | Detail |
|---|---|
| Camera | OAK-D Lite, top-down mount ~32 cm above board, USB 2.0 |
| Arm | uARM Swift Pro, `/dev/ttyACM0` |
| Pieces | Cobalt-blue glass (human) + clear glass (robot) |
| Board | Printed paper 8×8, standard chess layout |
| ArUco marker | DICT_7X7_250, ID 0, **30 mm** print, mounted on gripper facing upward |

> The extension tip is **15 cm below the marker**. Never command Z where the tip would collide with the board.

---

## File Structure

```
chessbotV1/
├── README.md                     ← this file
├── CLAUDE.md                     ← AI assistant context (technical details)
├── constants.py                  ← ALL tunable parameters
│
├── aruco_pose_pnp.py             ← ArUco 6-DOF pose via solvePnP
├── jog_cal.py                    ← Camera→arm calibration (interactive jog)
├── chess_glass_detector.py       ← Board corner detection + save tool
├── servo_pick.py                 ← Visual-servo pick/place pipeline
├── chessgame.py                  ← Full game loop (Stockfish vs human)
├── debug_board.py                ← Real-time square diff + blue detection viewer
├── measure_table_z.py            ← Jog arm to board surface to measure TABLE_Z
│
├── engine/
│   ├── __init__.py
│   └── stockfish.py              ← Stockfish subprocess wrapper
│
└── aruco_test/                   ← Standalone ArUco detection tests
    ├── test_aruco.py             ← Live detection viewer (D=scan dicts, Q=quit)
    ├── aruco_pose_pnp.py         ← Standalone copy
    └── constants.py              ← Minimal constants (imports from parent)
```

---

## Dependencies

```bash
pip install opencv-python numpy depthai python-chess
pip install uarm-python          # uARM Swift Pro SDK
```

Stockfish binary must be installed and on `$PATH`:
```bash
sudo apt install stockfish
```

---

## Setup (first time or after moving hardware)

### 1. Verify ArUco detection

```bash
cd aruco_test && python test_aruco.py
```

Press `D` to scan all dictionaries. Look for green text, std < 1 mm, reproj < 1 px.
Expected dictionary: **DICT_7X7_250, ID 0**.

### 2. Measure TABLE_Z (if board height changed)

```bash
python measure_table_z.py
```

Jog the arm down (`-` / `F`) until the extension tip just touches the board surface.
Press Enter — `TABLE_Z` in `constants.py` is updated automatically.

### 3. Calibrate camera→arm transform

```bash
python -u jog_cal.py
```

- Remove all chess pieces from the board.
- The ArUco marker must face **upward** on the gripper.
- Jog the arm (W/S/A/D/R/F) until the marker turns **green** in the window.
- Press **Space** to capture that position.
- Capture **6–9 poses** spread across different X, Y, and at least 2 Z heights.
- Press **Enter** to solve and save `T_cam_to_arm_topdown.npy`.
- Target: mean residual **< 5 mm**.

```
Controls
  W / S       +X / -X  (forward / back)    10 mm
  A / D       +Y / -Y  (left / right)      10 mm
  R / F       +Z / -Z  (up / down)          5 mm
  Shift+above           large step         30 / 30 / 15 mm
  Space       capture (only when marker is green)
  Enter       solve + save
  H           home arm
  Q           quit
```

### 4. Save board corners

```bash
python chess_glass_detector.py
```

- Set up the board in its playing position (pieces optional).
- Press `R` to click the 4 board corners on the camera feed (a1 → h1 → h8 → a8).
- Press `S` to save `board_calibration.npy`.
- Press `Q` to quit.

Re-run whenever the board is moved.

### 5. Tune pick offset (if arm picks off-centre)

Edit `constants.py`:

```python
SERVO_OFFSET_X = -10.0   # negative = shift toward arm base
SERVO_OFFSET_Y =   0.0   # positive = shift right
```

| Symptom | Fix |
|---|---|
| Picks too far forward (away from base) | decrease `SERVO_OFFSET_X` |
| Picks too far back (toward base) | increase `SERVO_OFFSET_X` |
| Picks too far left/right | adjust `SERVO_OFFSET_Y` |

No recalibration needed — just edit the constant and retest.

---

## Running the Game

```bash
python -u chessgame.py
```

Follow the prompts:
1. Choose player colour (w = robot is White, b = robot is Black).
2. Choose engine strength (ELO).
3. Robot makes its move first if it is White.
4. Make your move — the camera detects blue piece movement automatically.
5. Type `e7e5` + Enter at any time to override detection with a manual UCI move.

### How move detection works

1. Camera waits for the board to be still (10 stable frame-pairs).
2. **Before image** is captured.
3. Motion is detected (hand reaching in).
4. Board settles again (10 stable frame-pairs).
5. **2-second AE wait** — camera auto-exposure finishes adjusting.
6. **After image** is captured.
7. Blue squares are compared between before/after using HSV hue detection.
8. The move is inferred from which squares lost/gained a blue piece.
9. Validated against legal moves; fallback to pixel-diff if blue detection is ambiguous.

---

## Tuning Tools

### debug_board.py — real-time square viewer

```bash
python debug_board.py
```

| Key | Action |
|---|---|
| `S` | Re-capture diff baseline |
| `B` | Toggle blue detection mode |
| `+` / `-` | Raise / lower diff threshold |
| `Q` | Quit |

**Blue mode** (`B`): each square shows its blue-pixel fraction. Cyan overlay = blue piece detected. Orange inner box = inset sampling region. Use this to tune `BLUE_HUE_LO/HI`, `BLUE_SAT_MIN`, `BLUE_VAL_MIN`, `BLUE_FRAC` in `constants.py`.

### servo_pick.py — standalone pick/place

```bash
python -u servo_pick.py e2 e4        # full move
python -u servo_pick.py --pick e2    # pick only
python -u servo_pick.py --place e4   # place only (piece already held)
python -u servo_pick.py --home       # home arm
```

---

## Key Constants (constants.py)

| Constant | Default | Description |
|---|---|---|
| `TABLE_Z` | `48.0` | Arm Z when extension tip touches board |
| `TRANSIT_Z` | `120.0` | Arm Z for all lateral moves |
| `SERVO_OFFSET_X/Y` | `-10, 0` | Gripper tip alignment correction (mm) |
| `MARKER_OFFSET_Y` | `40.0` | Arm cmd → marker centre offset Y (mm) |
| `MARKER_OFFSET_Z` | `130.0` | Arm cmd → marker centre offset Z (mm) |
| `SQ_INSET` | `8` | Pixels inset from each square edge when sampling |
| `SETTLE_EXTRA_S` | `2.0` | Extra seconds after motion stops before sampling |
| `BLUE_HUE_LO/HI` | `95 / 135` | HSV hue range for blue glass pieces |
| `BLUE_SAT_MIN` | `80` | Minimum saturation for blue detection |
| `BLUE_FRAC` | `0.06` | Fraction of inset square that must be blue |

---

## Z Height Reference

```
Arm Z    Tip above board    Note
──────   ────────────────   ──────────────────────────────
 120         72 mm          TRANSIT_Z — safe lateral travel
  83         35 mm          hover — just above piece tops (~30 mm)
  63         15 mm          grip — mid-piece grab
  58         10 mm          place — release height
  48          0 mm          TABLE_Z — tip touches board
```

Never command Z < 90 during calibration (extension collision risk).

---

## Calibration Files

| File | Created by | Used by |
|---|---|---|
| `T_cam_to_arm_topdown.npy` | `jog_cal.py` | `servo_pick.py`, `chessgame.py` |
| `board_calibration.npy` | `chess_glass_detector.py` | `servo_pick.py`, `chessgame.py`, `debug_board.py` |
| `empty_board.png` | `chessgame.py` (auto) | `chessgame.py` |

These files are machine-specific and excluded from git (see `.gitignore`).
Regenerate them after any hardware repositioning.

---

## Status

- [x] ArUco detection — DICT_7X7_250, 30 mm, sub-mm stable
- [x] Camera→arm calibration — 2.97 mm mean residual (9 poses)
- [x] Board corners saved
- [x] Visual-servo pick/place — working, gripper offset tuned
- [x] Blue piece move detection — HSV hue tracking, AE-immune
- [ ] Full game end-to-end test
