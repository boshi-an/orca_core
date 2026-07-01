#!/usr/bin/env python
"""Discover per-joint encoder slope signs and write them to calibration.yaml.

For every joint, this nudges its motor in each direction and checks whether the
mapped channel's voltage rises or falls as the *actual* (motor-derived) joint
angle increases. The resulting ``joint_sensor_slope_sign`` (+1/-1) makes the
sensed angle track the joint in the correct direction.

The signs are written to **calibration.yaml**. Run
``discover_joint_sensor_mapping.py`` first: signs are only applied/persisted for
joints already present in ``joint_to_sensor_id`` (config.yaml).

Motors must be powered and the encoder board connected. Movements are small and
compliant (current-based position mode), and each motor is returned to where it
started. Slope signs also require the motors to be calibrated (the direction
reference comes from the joint<->motor mapping).

Usage:
    uv run python scripts/discover_joint_sensor_signs.py [config.yaml] \
        [--motor-delta 0.5] [--settle 0.5] [--samples 30] [--threshold 0.05] \
        [--joints index_mcp,thumb_mcp]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orca_core import OrcaHand
from orca_core.utils.utils import read_yaml


def main():
    parser = argparse.ArgumentParser(description="Discover per-joint encoder slope signs.")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to config.yaml")
    parser.add_argument("--motor-delta", type=float, default=0.5, help="Relative motor move per probe (rad)")
    parser.add_argument("--settle", type=float, default=0.5, help="Settle time after a move (s)")
    parser.add_argument("--samples", type=int, default=30, help="Voltage frames to average")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min voltage change to accept (V)")
    parser.add_argument("--probe-current", type=int, default=None, help="Max current during probing (mA)")
    parser.add_argument("--joints", default=None, help="Comma-separated subset of joints to probe")
    args = parser.parse_args()

    hand = OrcaHand(config_path=args.config_path)

    ok, msg = hand.connect()
    print(msg)
    if not ok:
        print("Failed to connect to the hand.")
        return

    ok, msg = hand.connect_joint_sensors(start_stream=False)
    print(msg)
    if not ok:
        hand.disconnect()
        sys.exit(1)

    # Require an explicit mapping in config.yaml (not the fallback template),
    # otherwise signs for joints missing from the template are silently dropped.
    raw_config = read_yaml(hand.config.config_path) or {}
    if not (raw_config.get("joint_sensors") or {}).get("joint_to_sensor_id"):
        print("No joint_to_sensor_id in config.yaml. Run "
              "discover_joint_sensor_mapping.py --write first.")
        hand.disconnect()
        sys.exit(1)

    joints = args.joints.split(",") if args.joints else None

    try:
        print("\nProbing joints (each motor is moved then returned)...")
        results = hand.discover_joint_sensor_map(
            joints=joints,
            motor_delta=args.motor_delta,
            settle_time=args.settle,
            num_samples=args.samples,
            threshold=args.threshold,
            probe_current=args.probe_current,
        )

        signs = {
            joint: r["slope_sign"]
            for joint, r in results.items()
            if r.get("slope_sign") is not None
        }

        print("\nDiscovered slope signs (joint -> sign):")
        for joint in results:
            sign = results[joint].get("slope_sign")
            sign_txt = f"{sign:+d}" if sign is not None else "? (channel/motor uncalibrated)"
            print(f"  {joint:<12} {sign_txt}")

        print(f"\nslope signs: {signs}")

        applied = hand.set_joint_sensor_slope_signs(signs, persist=True)
        print(f"Wrote {len(applied)} slope signs to calibration.yaml.")
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        hand.disconnect()


if __name__ == "__main__":
    main()
