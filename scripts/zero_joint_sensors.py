#!/usr/bin/env python
"""Zero-calibrate the magnetic-encoder joint sensors against a MuJoCo reference.

Workflow:
    1. A MuJoCo window shows the ORCA hand at the all-joints-zero pose.
    2. You move the *real* hand by hand to match that pose (motor torque is
       disabled so it is back-drivable).
    3. Press Enter. The per-channel voltage is averaged and written to
       ``calibration.yaml`` as the new ``voltage_at_zero`` for each mapped joint.

The sensor model is ``angle = slope * (voltage - voltage_at_zero)``; capturing
``voltage_at_zero`` at the known-zero pose makes each mapped joint read ~0 here.
``connect_joint_sensors`` re-applies these offsets automatically from then on.

Usage:
    uv run python scripts/calibrate_joint_sensors.py [config.yaml]
        [--mujoco-model PATH] [--no-mujoco] [--no-motors] [--samples 200]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orca_core import OrcaHand

DESCRIPTION_DIR = PROJECT_ROOT / "third_party" / "orcahand_description"


def resolve_model_path(explicit: str | None, hand_type: str | None) -> Path | None:
    """Pick the MuJoCo scene: explicit path wins, else the bundled description."""
    if explicit:
        return Path(explicit)
    scene = DESCRIPTION_DIR / f"scene_{hand_type or 'right'}.xml"
    return scene if scene.exists() else None


def launch_zero_pose_viewer(model_path: Path):
    """Open a passive MuJoCo viewer with every joint forced to zero.

    Returns the viewer handle (kept alive by the caller), or ``None`` if MuJoCo
    is unavailable or the model fails to load.
    """
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("MuJoCo not installed (pip install mujoco) — skipping the viewer.")
        return None

    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except Exception as e:
        print(f"Could not load MuJoCo model {model_path}: {e} — skipping the viewer.")
        return None

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Force every hinge/slide joint to 0 so the pose is the true zero pose,
    # regardless of any reference offsets baked into the model.
    for jnt in range(model.njnt):
        if model.jnt_type[jnt] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            data.qpos[model.jnt_qposadr[jnt]] = 0.0
    mujoco.mj_forward(model, data)

    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.sync()
    except Exception as e:
        print(f"Could not open the MuJoCo window ({e}) — proceeding without it.")
        return None
    print(f"MuJoCo reference pose shown from {model_path.name}.")
    return viewer


def main():
    parser = argparse.ArgumentParser(description="Zero-calibrate ORCA joint sensors with a MuJoCo reference.")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to config.yaml")
    parser.add_argument("--mujoco-model", default=None, help="Path to a MuJoCo scene XML")
    parser.add_argument("--no-mujoco", action="store_true", help="Skip the MuJoCo viewer")
    parser.add_argument("--no-motors", action="store_true", help="Skip the motor bus (sensors only)")
    parser.add_argument("--samples", type=int, default=200, help="Frames to average (default 200)")
    args = parser.parse_args()

    hand = OrcaHand(config_path=args.config_path)

    if not hand.config.joint_to_sensor_id:
        print("No joint_to_sensor_id mapping configured; nothing to calibrate.")
        return

    # Disable torque so the hand can be posed by hand.
    if not args.no_motors:
        ok, msg = hand.connect()
        print(msg)
        if ok:
            hand.disable_torque()
            print("Motor torque disabled — the hand is now back-drivable.")
        else:
            print("Continuing without motors (sensors only).")

    ok, msg = hand.connect_joint_sensors(start_stream=False)
    print(msg)
    if not ok:
        hand.disconnect()
        sys.exit(1)

    viewer = None
    if not args.no_mujoco:
        model_path = resolve_model_path(args.mujoco_model, hand.config.type)
        if model_path is None:
            print("No MuJoCo model found — proceeding without the viewer.")
        else:
            viewer = launch_zero_pose_viewer(model_path)

    try:
        input("\nMove ALL joints to zero to match the reference pose, then press Enter...")
        print(f"Capturing {args.samples} frames...")
        zero = hand.calibrate_joint_sensor_zero(num_samples=args.samples)

        print("\nCaptured voltage_at_zero (written to calibration.yaml):")
        for joint in sorted(zero):
            channel = hand.config.joint_to_sensor_id[joint]
            print(f"  {joint:<12} ch{channel:<2} {zero[joint]:.4f} V")
        print(f"\nDone — {len(zero)} joints zeroed.")
    except KeyboardInterrupt:
        print("\nAborted; calibration.yaml unchanged.")
    finally:
        if viewer is not None:
            viewer.close()
        hand.disconnect()


if __name__ == "__main__":
    main()
