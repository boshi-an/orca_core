    

import argparse

from orca_core.hardware_hand import OrcaHand


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Closed-loop joint control UI for the ORCA hand.")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to config.yaml")
    parser.add_argument("--kp", type=float, default=0.01, help="Proportional gain (default 0.3)")
    parser.add_argument("--rate", type=float, default=20.0, help="Control rate in Hz (default 20)")
    parser.add_argument("--no-sensors", action="store_true", help="Skip the encoder; open loop only")
    args = parser.parse_args()

    hand = OrcaHand(config_path=args.config_path)

    ok, msg = hand.connect()
    print(msg)
    if not ok:
        print("Failed to connect to the hand.")
        exit(0)

    hand.init_joints(force_calibrate=False)
    sensor_ok, sensor_msg = hand.connect_joint_sensors(start_stream=True)

    if not sensor_ok:
        print("Failed to connect to the joint sensors.")
        exit(0)

    hand.set_control_mode("current")

    while True :
        hand.set_joint_current({joint: 50.0 for joint in hand.config.joint_ids})
