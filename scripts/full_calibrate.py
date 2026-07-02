#!/usr/bin/env python
"""Full ORCA hand calibration in one run: motors + joint-sensor map/signs + zero.

Sequences the steps that together produce the final calibration:

  1. Motor calibration (same as scripts/calibrate.py): drives each joint to its
     mechanical limits and records motor limits + joint<->motor ratios to
     calibration.yaml.
  2. Joint-sensor mapping: probes each joint to learn which encoder channel it
     drives  ->  ``joint_to_sensor_id`` in config.yaml.
  3. Joint-sensor slope signs: from the *same* probe, which way each channel
     moves as the joint angle increases  ->  ``joint_sensor_slope_sign`` in
     calibration.yaml.
  4. Joint-sensor zero (same as scripts/zero_joint_sensors.py): pose the hand at
     the all-joints-zero reference (a MuJoCo window shows it) and capture the
     per-channel ``voltage_at_zero``  ->  ``joint_sensor_zero`` in
     calibration.yaml.

Steps 2 and 3 are obtained from one probe (:func:`discover_joint_sensors`) and
run *after* the motors are calibrated, because the slope-sign direction
reference comes from the joint<->motor mapping produced in step 1. Step 4 runs
last, since it needs the mapping from step 2 and is a hands-on (back-drivable)
capture.

Motors must be powered and the encoder board connected.

Usage:
    uv run python scripts/full_calibrate.py [config.yaml] \
        [--fingers thumb index | --joints thumb_cmc index_mcp] \
        [--force-wrist] [--skip-motors] [--skip-zero] [--no-mujoco] \
        [--motor-delta 0.5] [--settle 0.5] [--samples 30] [--threshold 0.05]
"""

import argparse
import sys

from common import add_hand_arguments, connect_hand, create_hand, shutdown_hand
from discover_joint_sensors import discover_joint_sensors
from zero_joint_sensors import launch_zero_pose_viewer, resolve_model_path


FINGER_TO_JOINTS = {
    "thumb": ["thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip"],
    "index": ["index_abd", "index_mcp", "index_pip"],
    "middle": ["middle_abd", "middle_mcp", "middle_pip"],
    "ring": ["ring_abd", "ring_mcp", "ring_pip"],
    "pinky": ["pinky_abd", "pinky_mcp", "pinky_pip"],
    "wrist": ["wrist"],
}

ALL_JOINTS = [j for joints in FINGER_TO_JOINTS.values() for j in joints]


def resolve_joints(args, parser) -> list | None:
    """Turn --fingers/--joints into a joint list (None = whole hand)."""
    if args.fingers and args.joints:
        parser.error("Cannot specify both --fingers and --joints. Use either one.")
    if args.fingers:
        joints = [j for finger in args.fingers for j in FINGER_TO_JOINTS[finger]]
        print(f"Calibrating fingers: {args.fingers}")
        print(f"Resolved joints: {joints}")
        return joints
    if args.joints:
        print(f"Calibrating joints: {args.joints}")
        return args.joints
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full ORCA hand calibration: motors + joint-sensor mapping + signs."
    )
    add_hand_arguments(parser)
    parser.add_argument("--force-wrist", action="store_true",
                        help="Recalibrate the wrist even if already calibrated.")
    parser.add_argument("--fingers", type=str, nargs="+", choices=list(FINGER_TO_JOINTS),
                        help="Fingers to calibrate (e.g. --fingers thumb index).")
    parser.add_argument("--joints", type=str, nargs="+", choices=ALL_JOINTS,
                        help="Individual joints to calibrate (e.g. --joints thumb_cmc index_mcp).")
    parser.add_argument("--skip-motors", action="store_true",
                        help="Skip motor calibration; only discover sensor mapping + signs.")
    parser.add_argument("--skip-zero", action="store_true",
                        help="Skip the hands-on joint-sensor zero capture.")
    # Sensor-probe tuning (see discover_joint_sensors.py).
    parser.add_argument("--motor-delta", type=float, default=0.5, help="Relative motor move per probe (rad)")
    parser.add_argument("--settle", type=float, default=0.5, help="Settle time after a move (s)")
    parser.add_argument("--samples", type=int, default=30, help="Voltage frames to average")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min voltage change to accept (V)")
    parser.add_argument("--probe-current", type=int, default=None, help="Max current during probing (mA)")
    # Joint-sensor zero capture (see zero_joint_sensors.py).
    parser.add_argument("--mujoco-model", default=None, help="Path to a MuJoCo scene XML for the zero pose")
    parser.add_argument("--no-mujoco", action="store_true", help="Skip the MuJoCo reference viewer")
    parser.add_argument("--zero-samples", type=int, default=200,
                        help="Frames to average for the zero capture (default 200)")
    args = parser.parse_args()

    joints = resolve_joints(args, parser)

    hand = create_hand(args.config_path, use_mock=args.mock)
    try:
        connect_hand(hand)

        ok, msg = hand.connect_joint_sensors(start_stream=False)
        print(msg)
        if not ok:
            print("Failed to connect joint sensors; cannot run the full calibration.")
            return 1

        if args.skip_motors:
            print("\n[1/3] Skipping motor calibration (--skip-motors).")
        else:
            print("\n[1/3] Motor calibration...")
            # Motors are calibrated without the joint sensors: the mapping does
            # not exist yet, and this run establishes the direction reference the
            # slope signs need.
            hand.calibrate(force_wrist=args.force_wrist, joints=joints,
                           calibrate_joint_sensors=False)
            print("Motor calibration complete.")

        print("\n[2/3] Joint-sensor mapping + slope signs (single probe)...")
        mapping, signs, conflicts = discover_joint_sensors(
            hand,
            joints=joints,
            motor_delta=args.motor_delta,
            settle_time=args.settle,
            num_samples=args.samples,
            threshold=args.threshold,
            probe_current=args.probe_current,
        )
        if conflicts:
            print("\nFull calibration finished with UNRESOLVED sensor conflicts; "
                  "mapping/signs were not written. Re-probe the affected joints, "
                  "then re-run with --skip-motors.")
            return 1

        zeroed = 0
        if args.skip_zero:
            print("\n[3/3] Skipping joint-sensor zero capture (--skip-zero).")
        else:
            zeroed = zero_joint_sensors(hand, args)

        print("\nFull calibration complete.")
        print(f"  motor calibration -> {hand.config.calibration_path}")
        print(f"  joint_to_sensor_id ({len(mapping)}) -> {hand.config.config_path}")
        print(f"  slope signs ({len(signs)}) -> {hand.config.calibration_path}")
        print(f"  joint_sensor_zero ({zeroed}) -> {hand.config.calibration_path}")
        return 0
    finally:
        shutdown_hand(hand)


def zero_joint_sensors(hand, args) -> int:
    """Capture the all-joints-zero pose as ``voltage_at_zero`` (step 4).

    Disables torque so the hand is back-drivable, optionally shows the MuJoCo
    zero-pose reference, waits for the user to pose the hand, then averages and
    persists the per-channel zero. Returns the number of joints zeroed.
    """
    print("\n[3/3] Joint-sensor zero capture...")
    if not hand.config.joint_to_sensor_id:
        print("No joint_to_sensor_id mapping; skipping zero capture.")
        return 0

    hand.disable_torque()
    print("Motor torque disabled — the hand is now back-drivable.")

    viewer = None
    if not args.no_mujoco:
        model_path = resolve_model_path(args.mujoco_model, hand.config.type)
        if model_path is None:
            print("No MuJoCo model found — proceeding without the viewer.")
        else:
            viewer = launch_zero_pose_viewer(model_path)

    try:
        input("\nMove ALL joints to zero to match the reference pose, then press Enter...")
        print(f"Capturing {args.zero_samples} frames...")
        zero = hand.calibrate_joint_sensor_zero(num_samples=args.zero_samples)
        print("\nCaptured voltage_at_zero (written to calibration.yaml):")
        for joint in sorted(zero):
            channel = hand.config.joint_to_sensor_id[joint]
            print(f"  {joint:<12} ch{channel:<2} {zero[joint]:.4f} V")
        return len(zero)
    except KeyboardInterrupt:
        print("\nZero capture aborted; joint_sensor_zero unchanged.")
        return 0
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
