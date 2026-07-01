#!/usr/bin/env python
"""Discover the joint <-> encoder-channel correspondence and write it to config.yaml.

For every joint, this nudges its motor a little in each direction and watches
which magnetic-encoder channel's voltage changes the most. That channel is taken
to be the one measuring the joint, producing the ``joint_to_sensor_id`` mapping
you would otherwise fill in by hand.

The mapping is written to the ``joint_sensors`` block of **config.yaml**. Run
this first; then run ``discover_joint_sensor_signs.py`` for the slope signs
(which need this mapping to exist).

Motors must be powered (the joints are actuated) and the encoder board connected.
Movements are small and compliant (current-based position mode), and each motor
is returned to where it started.

Usage:
    uv run python scripts/discover_joint_sensor_mapping.py [config.yaml] \
        [--motor-delta 0.5] [--settle 0.5] [--samples 30] [--threshold 0.05] \
        [--joints index_mcp,thumb_mcp] [--write]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orca_core import OrcaHand
from orca_core.utils.utils import read_yaml, update_yaml


def write_mapping(hand, mapping: dict) -> None:
    """Persist ``joint_to_sensor_id`` into the config.yaml joint_sensors block."""
    config = read_yaml(hand.config.config_path) or {}
    joint_sensors = dict(config.get("joint_sensors") or {})
    joint_sensors["joint_to_sensor_id"] = dict(mapping)
    update_yaml(hand.config.config_path, "joint_sensors", joint_sensors)
    print(f"Wrote joint_to_sensor_id ({len(mapping)} joints) to "
          f"{Path(hand.config.config_path).name}.")


def main():
    parser = argparse.ArgumentParser(description="Discover the joint<->sensor channel correspondence.")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to config.yaml")
    parser.add_argument("--motor-delta", type=float, default=0.5, help="Relative motor move per probe (rad)")
    parser.add_argument("--settle", type=float, default=0.5, help="Settle time after a move (s)")
    parser.add_argument("--samples", type=int, default=30, help="Voltage frames to average")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min voltage change to accept (V)")
    parser.add_argument("--probe-current", type=int, default=None, help="Max current during probing (mA)")
    parser.add_argument("--joints", default=None, help="Comma-separated subset of joints to probe")
    parser.add_argument("--write", action="store_true", help="Write the mapping to config.yaml")
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
        print("\nProbing joints (each motor is moved then returned)...")
        results = hand.discover_joint_sensor_map(
            joints=joints,
            motor_delta=args.motor_delta,
            settle_time=args.settle,
            num_samples=args.samples,
            threshold=args.threshold,
            probe_current=args.probe_current,
        )

        # Build the mapping, dropping joints with no clear channel.
        mapping = {
            joint: r["channel"] for joint, r in results.items() if r["channel"] is not None
        }

        # Flag channels claimed by more than one joint.
        by_channel: dict[int, list[str]] = {}
        for joint, channel in mapping.items():
            by_channel.setdefault(channel, []).append(joint)
        conflicts = {ch: js for ch, js in by_channel.items() if len(js) > 1}

        print("\nDiscovered correspondence (joint -> channel):")
        for joint in results:
            channel = results[joint]["channel"]
            tag = "  <- CONFLICT" if channel is not None and channel in conflicts else ""
            shown = channel if channel is not None else "none"
            print(f"  {joint:<12} ch={shown}{tag}")

        if conflicts:
            print("\nWARNING: some channels were attributed to multiple joints:")
            for ch, js in conflicts.items():
                print(f"  channel {ch}: {', '.join(js)}")
            print("Re-run with a larger --motor-delta or --threshold, or probe "
                  "those joints individually with --joints.")

        print(f"\njoint_to_sensor_id: {mapping}")

        if args.write:
            if conflicts:
                print("Not writing: resolve channel conflicts first.")
            else:
                write_mapping(hand, mapping)
                print("Next: run discover_joint_sensor_signs.py to capture slope signs.")
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        hand.disconnect()


if __name__ == "__main__":
    main()
