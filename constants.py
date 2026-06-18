"""
constants.py — All tunable parameters for chessbotV1
"""

# ── Arm connection ────────────────────────────────────────────────────────────
PORT            = '/dev/ttyACM0'
ARM_SPEED       = 15000    # mm/min — normal moves
ARM_SPEED_FAST  = 25000    # repositioning / homing
ARM_SPEED_SLOW  = 4000     # dive to pick/place

# ── Arm workspace limits ──────────────────────────────────────────────────────
MIN_RADIUS = 75.0    # mm from arm base — inner limit (h8 at 70mm is physically clamped)
MAX_RADIUS = 350.0   # mm from arm base — outer limit

# ── Z reference ───────────────────────────────────────────────────────────────
TABLE_Z = 48.00      # arm Z when gripper tip touches board surface (mm)

# ── Home position ─────────────────────────────────────────────────────────────
HOME_X = 150.0
HOME_Y = -185.0      # off-board to the side — out of camera FOV
HOME_Z = 120.0

# ── ArUco marker ──────────────────────────────────────────────────────────────
# Marker confirmed: cv2.aruco.DICT_7X7_250, ID 0, 3×3 cm physical print
MARKER_BLACK_SIDE_MM = 30.0    # physical outer side of printed marker (mm)

# Gripper tip → ArUco marker centre offset, in arm frame (mm).
MARKER_OFFSET_X = 0.0
MARKER_OFFSET_Y = 40.0   # updated from 30 — arm moves more forward from base to pick
MARKER_OFFSET_Z = 130.0

# ── Gripper tip offset correction ────────────────────────────────────────────
SERVO_OFFSET_X = -10.0   # mm — arm was picking ~10mm too far forward
SERVO_OFFSET_Y =   0.0   # mm

# ── Pick / place motion ───────────────────────────────────────────────────────
TRANSIT_Z          = 120.0   # arm Z for all lateral moves — ext tip 72mm above board
HOVER_Z            = 35.0    # mm above TABLE_Z for hover before dive
GRIP_Z_OFFSET      = 15.0    # TABLE_Z + this = grip height
PLACE_Z_OFFSET     = 10.0    # TABLE_Z + this = release height
TEST_SPEED_PLUNGE  = 4000    # mm/min — slow dive to pick/place
SETTLE_S           = 0.25    # seconds to wait after arm stops
PICK_DELAY         = 2.0     # seconds after gripper CLOSES before lifting
PLACE_DELAY        = 1.5     # seconds after gripper OPENS before lifting
GRIPPER_OPEN_DELAY = 1.0     # seconds for gripper to fully open before approaching

# Capture bins — arm rests pieces here when taking (off the right side of board)
CAPTURE_BIN_X  = 320.0
CAPTURE_BIN_Y  = 130.0
CAPTURE_BIN_Z  = TABLE_Z + PLACE_Z_OFFSET

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_W = 640
CAM_H = 480
CAM_FPS = 15

DEPTH_MIN = 0.20   # metres
DEPTH_MAX = 1.50   # metres

# ── Calibration output file ───────────────────────────────────────────────────
T_CAM_TO_ARM_FILE = 'T_cam_to_arm_topdown.npy'

# ── Board square sampling ─────────────────────────────────────────────────────
# Pixels to inset from each square edge before sampling.
SQ_INSET = 8    # px — inner 64×64 of each 80×80 square is sampled

# Extra seconds to wait after board pixel-motion stops before sampling after_img.
# Camera AE and piece reflections take ~2s to fully settle after hand leaves.
SETTLE_EXTRA_S = 2.0

# ── Blue piece detection (human plays cobalt-blue glass) ──────────────────────
BLUE_HUE_LO  = 95    # lower hue bound for blue glass
BLUE_HUE_HI  = 135   # upper hue bound (cobalt blue)
BLUE_SAT_MIN = 80    # min saturation — pieces on light squares are less saturated than on dark
BLUE_VAL_MIN = 50    # min brightness — filters very-dark empty board squares
BLUE_FRAC    = 0.06  # fraction of inset square that must be blue to count
