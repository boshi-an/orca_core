from orca_core import OrcaHand
import time


if __name__ == "__main__":
    # Example usage:
    try :
        hand = OrcaHand()
        status = hand.connect()
        hand.enable_torque()

        # Set the desired joint positions to 0
        hand.set_joint_pos({joint: 0 for joint in hand.joint_ids})
        time.sleep(1)
        # Set the desired joint positions to 90 degrees
        hand.set_joint_pos({joint: 45 for joint in hand.joint_ids})
        time.sleep(2)
    finally:
        hand.disable_torque()
        hand.disconnect()
        