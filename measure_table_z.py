#!/usr/bin/env python3 -u
"""
measure_table_z.py — Jog arm down until extension tip touches board, record Z.

Controls
────────
  +  or  =   raise arm 1 mm
  -          lower arm 1 mm
  F          lower arm 5 mm (fast)
  R          raise arm 5 mm (fast)
  Enter      confirm and save to constants.py
  Q          quit without saving
"""

import re, sys, tty, termios
import constants as const


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def update_constant(name: str, value: float):
    path = "constants.py"
    with open(path) as f:
        src = f.read()
    new_src, n = re.subn(
        rf'^({re.escape(name)}\s*=\s*)[-\d.]+',
        rf'\g<1>{value:.2f}',
        src, flags=re.MULTILINE
    )
    if n == 0:
        print(f"\n  ERROR: '{name}' not found in {path}")
        return False
    with open(path, 'w') as f:
        f.write(new_src)
    return True


def safe_z(arm) -> float:
    """Read Z reliably — strips any trailing non-numeric chars."""
    pos = arm.get_position()
    # pos may be [x, y, z] or [x, y, z, r]; z might be float or str
    raw = pos[2]
    return float(str(raw).strip().rstrip('M').strip())


def main():
    print("=" * 52)
    print("  TABLE_Z Measurement — keyboard jog")
    print("=" * 52)
    print(f"  Current TABLE_Z = {const.TABLE_Z}\n")

    print("Connecting to arm...")
    from uarm.wrapper import SwiftAPI
    arm = SwiftAPI(port=const.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    print("  Connected.")

    # Start from a safe height so we can jog down
    start_z = 60.0
    arm.set_position(x=const.HOME_X, y=0, z=start_z,
                     speed=const.ARM_SPEED, wait=True)
    current_z = start_z
    print(f"  Arm at x={const.HOME_X}  y=0  z={current_z:.1f}")
    print()
    print("  Jog controls:")
    print("    -       lower 1 mm      F  lower 5 mm")
    print("    + / =   raise 1 mm      R  raise 5 mm")
    print("    Enter   confirm Z        Q  quit")
    print()
    print("  Lower until the EXTENSION TIP just touches the board surface.")
    print(f"  Current Z: {current_z:.1f} mm", end="", flush=True)

    while True:
        ch = getch().lower()

        if ch in ('\r', '\n'):          # Enter — confirm
            break
        elif ch == 'q':
            print("\n\n  Aborted.")
            arm.set_position(x=const.HOME_X, y=const.HOME_Y, z=const.HOME_Z,
                             speed=const.ARM_SPEED_FAST, wait=True)
            return
        elif ch == '-':
            current_z -= 1.0
        elif ch in ('+', '='):
            current_z += 1.0
        elif ch == 'f':
            current_z -= 5.0
        elif ch == 'r':
            current_z += 5.0
        else:
            continue

        arm.set_position(x=const.HOME_X, y=0, z=current_z,
                         speed=const.ARM_SPEED_SLOW, wait=True)
        measured = safe_z(arm)
        print(f"\r  Current Z: {measured:.1f} mm   ", end="", flush=True)
        current_z = measured   # use actual reported value

    measured = safe_z(arm)
    print(f"\n\n  Recorded Z = {measured:.2f} mm")

    arm.set_position(x=const.HOME_X, y=const.HOME_Y, z=const.HOME_Z,
                     speed=const.ARM_SPEED_FAST, wait=True)
    print("  Arm returned home.")

    print(f"\n  New TABLE_Z = {measured:.2f}")
    ok = update_constant("TABLE_Z", measured)
    if ok:
        print(f"  ✓ constants.py updated: TABLE_Z = {measured:.2f}")
    else:
        print(f"  Set TABLE_Z = {measured:.2f} in constants.py manually.")


if __name__ == "__main__":
    main()
