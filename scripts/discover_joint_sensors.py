#!/usr/bin/env python
"""Discover the joint<->encoder correspondence *and* slope signs in one probe.

For every joint, this nudges its motor a little in each direction and watches
the magnetic-encoder channels. A single actuation per joint yields both:

    - the channel whose voltage changed the most -> ``joint_to_sensor_id``
      (config.yaml), and
    - whether that channel rises or falls as the *actual* (motor-derived) joint
      angle increases -> ``joint_sensor_slope_sign`` (calibration.yaml).

Both come from one call to :meth:`OrcaHand.discover_joint_sensor_map` — there is
no second, separate probing pass. This replaces the old
``discover_joint_sensor_mapping.py`` + ``discover_joint_sensor_signs.py`` pair.

Motors must be powered (the joints are actuated) and the encoder board
connected. Movements are small and compliant (current-based position mode) and
each motor is returned to where it started. Correct slope signs additionally
require the motors to be calibrated (the direction reference comes from the
joint<->motor mapping); run ``calibrate.py`` first, or use ``full_calibrate.py``
which sequences everything.

Usage:
    uv run python scripts/discover_joint_sensors.py [config.yaml] \
        [--motor-delta 0.5] [--settle 
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orca_core import OrcaHand


def find_channel_conflicts(mapping: dict) -> dict:
    """Return ``{channel: [joints...]}`` for channels claimed by >1 joint."""
    by_channel: dict[int, list[str]] = {}
    for joint, channel in mapping.items():
        by_channel.setdefault(channel, []).append(joint)
    return {ch: js for ch, js in by_channel.items() if len(js) > 1}


def discover_joint_sensors(
    hand,
    joints=None,
    motor_delta=0.5,
    settle_time=0.5,
    num_samples=30,
    threshold=0.05,
    probe_current=None,
):
    """Probe every joint once and derive both the mapping and the slope signs.

    When there are no channel conflicts, the mapping is
    persisted to config.yaml and the slope signs to calibration.yaml, both via
    the hand API. Returns ``(mapping, signs, conflicts)``.
    """
    print("\nProbing joints (each motor is moved then returned)...")
    results = hand.discover_joint_sensor_map(
        joints=joints,
        motor_delta=motor_delta,
        settle_time=settle_time,
        num_samples=num_samples,
        threshold=threshold,
        probe_current=probe_current,
    )

    # Both outputs come from the same per-joint result.
    mapping = {j: r["channel"] for j, r in results.items() if r["channel"] is not None}
    signs = {
        j: r["slope_sign"]
        for j, r in results.items()
        if r.get("slope_sign") is not None
    }
    conflicts = find_channel_conflicts(mapping)

    print("\nDiscovered joint -> channel (slope sign):")
    for joint, r in results.items():
        channel = r["channel"]
        if channel is None:
            print(f"  {joint:<12} no clear channel")
            continue
        sign = r.get("slope_sign")
        sign_txt = f"{sign:+d}" if sign is not None else "? (motors uncalibrated)"
        tag = "  <- CONFLICT" if channel in conflicts else ""
        print(f"  {joint:<12} ch={channel} sign={sign_txt}{tag}")

    if conflicts:
        print("\nWARNING: some channels were attributed to multiple joints:")
        for ch, js in conflicts.items():
            print(f"  channel {ch}: {', '.join(js)}")
        print("Re-run with a larger --motor-delta or --threshold, or probe "
              "those joints individually with --joints.")

    if conflicts:
        print("\nNot writing: resolve channel conflicts first.")
    else:
        hand.set_joint_sensor_mapping(mapping, persist=True)
        print(f"\nWrote joint_to_sensor_id ({len(mapping)} joints) to "
                f"{Path(hand.config.config_path).name}.")
        applied_signs = hand.set_joint_sensor_slope_signs(signs, persist=True)
        print(f"Wrote {len(applied_signs)} slope signs to "
                f"{Path(hand.config.calibration_path).name}.")

    return mapping, signs, conflicts


def main():
    parser = argparse.ArgumentParser(
        description="Discover the joint<->sensor channel mapping and slope signs."
    )
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

    joints = args.joints.split(",") if args.joints else None

    try:
        discover_joint_sensors(
            hand,
            joints=joints,
            motor_delta=args.motor_delta,
            settle_time=args.settle,
            num_samples=args.samples,
            threshold=args.threshold,
            probe_current=args.probe_current,
        )
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        hand.disconnect()


if __name__ == "__main__":
    main()
