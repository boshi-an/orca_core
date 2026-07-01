"""Tests for joint-space current control.

Regression coverage for the bug where ``_joint_to_motor_current`` scaled the
commanded current by the position transmission ratio (~0.015), collapsing a
100 mA command to ~1.5 at the motor so the hand never moved. Joint currents are
motor-space (mA) and must pass through with only a calibrated direction sign.
"""

import pytest


def _idx(hand, joint):
    motor_id = hand.config.joint_to_motor_map[joint]
    return hand.config.motor_id_to_idx_dict[motor_id]


def test_joint_to_motor_current_passes_through_with_sign(initialized_mock_hand):
    hand = initialized_mock_hand
    command = 60.0
    motor_current = hand._joint_to_motor_current(
        {j: command for j in hand.config.joint_ids}
    )
    for joint in hand.config.joint_ids:
        inverted = hand.config.joint_inversion_dict.get(joint, False)
        expected = -command if inverted else command
        assert motor_current[_idx(hand, joint)] == pytest.approx(expected)


def test_joint_to_motor_current_ignores_transmission_ratio(initialized_mock_hand):
    """Current must NOT be multiplied by the joint->motor position ratio."""
    hand = initialized_mock_hand
    ratios = hand.calibration.joint_to_motor_ratios_dict
    # A joint whose ratio is clearly not +-1, so ratio-scaling is detectable.
    joint = next(
        j for j in hand.config.joint_ids
        if abs(abs(ratios[hand.config.joint_to_motor_map[j]]) - 1.0) > 0.1
    )
    motor_id = hand.config.joint_to_motor_map[joint]
    out = hand._joint_to_motor_current({joint: 100.0})[_idx(hand, joint)]
    ratio_scaled = 100.0 * ratios[motor_id]
    assert out != pytest.approx(ratio_scaled)
    assert abs(out) == pytest.approx(100.0)


def test_joint_to_motor_current_preserves_none(initialized_mock_hand):
    hand = initialized_mock_hand
    joint = hand.config.joint_ids[0]
    out = hand._joint_to_motor_current({joint: None})
    assert out[_idx(hand, joint)] is None


def test_set_joint_current_writes_signed_current_to_motors(initialized_mock_hand):
    hand = initialized_mock_hand
    command = 40.0
    hand.set_joint_current({j: command for j in hand.config.joint_ids})
    for joint in hand.config.joint_ids:
        motor_id = hand.config.joint_to_motor_map[joint]
        inverted = hand.config.joint_inversion_dict.get(joint, False)
        expected = -command if inverted else command
        assert hand._motor_client._cur[motor_id] == pytest.approx(expected)


def test_set_joint_current_clamps_to_configured_limits(initialized_mock_hand):
    hand = initialized_mock_hand
    joint = hand.config.joint_ids[0]
    motor_id = hand.config.joint_to_motor_map[joint]
    _, max_current = hand.config.joint_current_dict[joint]
    hand.set_joint_current({joint: max_current * 10})  # far beyond the limit
    inverted = hand.config.joint_inversion_dict.get(joint, False)
    expected = -max_current if inverted else max_current
    assert hand._motor_client._cur[motor_id] == pytest.approx(expected)
