#!/usr/bin/env python
"""Minimal example: stream sensed joint positions from the encoder board.

The magnetic-encoder board is read via the pure-python ``SensorReader`` in
``third_party/orca_sensor``. To map sensor channels to joint names, add a block
like this to your ``config.yaml``::

    joint_sensors:
      port: /dev/ttyACM0
      baudrate: 2000000
      joint_to_sensor_id:
        wrist: 0
        index_mcp: 1
        ...

Without a ``joint_to_sensor_id`` mapping the script falls back to printing the
raw per-channel angles.

Usage:
    uv run python scripts/example_joint_sensor.py
    uv run python scripts/example_joint_sensor.py path/to/config.yaml
"""

import sys
import time
from pathlib import Path

from orca_core import OrcaHand

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "orca_core" / "models" / "v2" / "orcahand_right" / "config.yaml"
)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CONFIG)
    hand = OrcaHand(config_path=config_path)

    ok, msg = hand.connect_joint_sensors(start_stream=True)
    print(msg)
    if not ok:
        sys.exit(1)

    has_mapping = bool(hand.config.joint_to_sensor_id)
    time.sleep(0.1)  # let the first frame arrive

    try:
        print("Streaming joint angles (deg) — press Ctrl+C to stop.\n")
        while True:
            if has_mapping:
                pos = hand.get_sensed_joint_positions()
                line = "  ".join(f"{j}: {v:7.2f}" for j, v in sorted(pos.as_dict().items()))
            else:
                angles = hand.get_sensed_joint_angles()
                line = " ".join(f"{a:7.2f}" for a in angles)
            print(line, end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        hand.disconnect()


if __name__ == "__main__":
    main()
