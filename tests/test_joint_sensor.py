"""Tests for joint-position (magnetic-encoder) sensing.

Covers the pure mapping helper, config parsing/validation of the
``joint_sensors`` block, and the ``OrcaHand`` integration with an injected fake
sensor client (no serial hardware required).
"""

import shutil
from pathlib import Path

import pytest

from orca_core.hand_config import HandConfigValidationError, OrcaHandConfig
from orca_core.hardware.sensing.constants import DEFAULT_JOINT_TO_SENSOR_ID
from orca_core.hardware.sensing.joint_sensor_client import (
    map_angles_to_joints,
    pick_changed_channel,
    required_slope_sign,
)
from orca_core.hardware.sensing.types import JointSensorReading
from orca_core.hardware_hand import MockOrcaHand
from orca_core.utils.utils import read_yaml, update_yaml

SRC_MODEL = "orca_core/models/v2/orcahand_right"


@pytest.fixture
def model_with_sensors(tmp_path):
    """Copy a packaged model and add a ``joint_sensors`` block to its config."""
    def _make(joint_to_sensor_id, **extra):
        dst = tmp_path / "model"
        shutil.copytree(SRC_MODEL, dst)
        cfg = str(dst / "config.yaml")
        block = {"joint_to_sensor_id": joint_to_sensor_id, **extra}
        update_yaml(cfg, "joint_sensors", block)
        return cfg

    return _make


class FakeJointSensorClient:
    """Stand-in for JointSensorClient returning deterministic angles/voltages."""

    def __init__(self, angles, voltages=None):
        self._angles = list(angles)
        self._voltages = list(voltages) if voltages is not None else [0.0] * 16
        self.is_connected = True
        self.stream_started = False
        self.voltage_at_zero = {}

    def get_latest(self):
        return list(self._angles), 123.0

    def read_angles(self):
        return list(self._angles)

    def capture_zero_voltages(self, num_samples):
        return list(self._voltages)

    def set_voltage_at_zero(self, overrides):
        self.voltage_at_zero.update(overrides)

    def get_voltage_at_zero(self):
        return dict(self.voltage_at_zero)

    def set_slope_sign(self, signs):
        self.slope_signs = getattr(self, "slope_signs", {})
        self.slope_signs.update(signs)

    def start_stream(self):
        self.stream_started = True

    def stop_stream(self):
        self.stream_started = False

    def disconnect(self):
        self.is_connected = False


def test_map_angles_to_joints_basic():
    angles = [float(i) for i in range(16)]
    mapped = map_angles_to_joints(angles, {"wrist": 0, "index_mcp": 6})
    assert mapped == {"wrist": 0.0, "index_mcp": 6.0}


def test_map_angles_to_joints_drops_out_of_range():
    angles = [1.0, 2.0, 3.0]
    mapped = map_angles_to_joints(angles, {"a": 0, "b": 99, "c": -1})
    assert mapped == {"a": 1.0}


def test_config_parses_joint_sensors_block(model_with_sensors):
    cfg = model_with_sensors(
        {"wrist": 0, "index_mcp": 1, "thumb_mcp": 2},
        port="/dev/ttyACM9",
        baudrate=2_000_000,
    )
    config = OrcaHandConfig.from_config_path(config_path=cfg)
    assert config.joint_sensor_port == "/dev/ttyACM9"
    assert config.joint_sensor_baudrate == 2_000_000
    assert config.joint_to_sensor_id == {"wrist": 0, "index_mcp": 1, "thumb_mcp": 2}


def test_config_defaults_without_block():
    # With no joint_sensors block, the mapping falls back to the template
    # (the v2 right hand has all 16 finger joints), and the port stays unset.
    config = OrcaHandConfig.from_config_path(
        config_path=f"{SRC_MODEL}/config.yaml"
    )
    assert config.joint_to_sensor_id == DEFAULT_JOINT_TO_SENSOR_ID
    assert config.joint_sensor_port is None


def test_config_rejects_duplicate_sensor_id(model_with_sensors):
    cfg = model_with_sensors({"wrist": 0, "index_mcp": 0})
    with pytest.raises(HandConfigValidationError, match="more than one joint"):
        OrcaHandConfig.from_config_path(config_path=cfg)


def test_config_rejects_out_of_range_sensor_id(model_with_sensors):
    cfg = model_with_sensors({"wrist": 99})
    with pytest.raises(HandConfigValidationError, match="out of range"):
        OrcaHandConfig.from_config_path(config_path=cfg)


def test_config_rejects_unknown_joint(model_with_sensors):
    cfg = model_with_sensors({"not_a_joint": 0})
    with pytest.raises(HandConfigValidationError, match="not defined"):
        OrcaHandConfig.from_config_path(config_path=cfg)


def test_hand_get_sensed_joint_positions(model_with_sensors):
    cfg = model_with_sensors({"wrist": 0, "index_mcp": 1, "thumb_mcp": 2})
    hand = MockOrcaHand(config_path=cfg)
    hand._joint_sensor_client = FakeJointSensorClient([i * 10.0 for i in range(16)])

    pos = hand.get_sensed_joint_positions()
    assert pos.as_dict() == {"wrist": 0.0, "index_mcp": 10.0, "thumb_mcp": 20.0}


def test_hand_get_sensed_joint_angles_raw(model_with_sensors):
    cfg = model_with_sensors({"wrist": 0})
    hand = MockOrcaHand(config_path=cfg)
    hand._joint_sensor_client = FakeJointSensorClient([float(i) for i in range(16)])
    assert hand.get_sensed_joint_angles() == [float(i) for i in range(16)]


def test_hand_get_sensed_joint_data(model_with_sensors):
    cfg = model_with_sensors({"wrist": 0, "index_mcp": 1})
    hand = MockOrcaHand(config_path=cfg)
    hand._joint_sensor_client = FakeJointSensorClient([i * 2.0 for i in range(16)])

    reading = hand.get_sensed_joint_data()
    assert isinstance(reading, JointSensorReading)
    assert reading["wrist"] == 0.0
    assert reading["index_mcp"] == 2.0
    assert len(reading.raw) == 16
    assert reading.timestamp == 123.0


def test_get_sensed_positions_requires_mapping(model_with_sensors):
    cfg = model_with_sensors({})
    hand = MockOrcaHand(config_path=cfg)
    hand._joint_sensor_client = FakeJointSensorClient([0.0] * 16)
    with pytest.raises(ValueError, match="joint_to_sensor_id"):
        hand.get_sensed_joint_positions()


def test_get_sensed_positions_requires_connection(model_with_sensors):
    cfg = model_with_sensors({"wrist": 0})
    hand = MockOrcaHand(config_path=cfg)
    with pytest.raises(RuntimeError, match="not connected"):
        hand.get_sensed_joint_angles()


def test_calibrate_joint_sensor_zero_captures_and_persists(model_with_sensors):
    cfg = model_with_sensors({"index_mcp": 5, "thumb_mcp": 2})
    voltages = [0.0] * 16
    voltages[5] = 2.10
    voltages[2] = 1.78

    hand = MockOrcaHand(config_path=cfg)
    client = FakeJointSensorClient([0.0] * 16, voltages=voltages)
    hand._joint_sensor_client = client

    zero = hand.calibrate_joint_sensor_zero(num_samples=10)
    assert zero == {"index_mcp": 2.10, "thumb_mcp": 1.78}
    # Applied live, keyed by channel.
    assert client.voltage_at_zero == {5: 2.10, 2: 1.78}
    # Persisted to calibration.yaml.
    cal = read_yaml(str(Path(cfg).parent / "calibration.yaml"))
    assert cal["joint_sensor_zero"] == {"index_mcp": 2.10, "thumb_mcp": 1.78}


def test_apply_joint_sensor_zero_on_connect(model_with_sensors):
    cfg = model_with_sensors({"index_mcp": 5, "thumb_mcp": 2})
    update_yaml(
        str(Path(cfg).parent / "calibration.yaml"),
        "joint_sensor_zero",
        {"index_mcp": 2.10, "thumb_mcp": 1.78},
    )

    hand = MockOrcaHand(config_path=cfg)
    client = FakeJointSensorClient([0.0] * 16)
    hand._joint_sensor_client = client

    applied = hand._apply_joint_sensor_zero()
    assert applied == 2
    assert client.voltage_at_zero == {5: 2.10, 2: 1.78}


def test_calibrate_zero_requires_mapping(model_with_sensors):
    cfg = model_with_sensors({})
    hand = MockOrcaHand(config_path=cfg)
    hand._joint_sensor_client = FakeJointSensorClient([0.0] * 16)
    with pytest.raises(ValueError, match="joint_to_sensor_id"):
        hand.calibrate_joint_sensor_zero()


def test_pick_changed_channel_picks_largest_above_threshold():
    channel, delta, runner_up = pick_changed_channel(
        [1.0, 1.0, 1.0], [1.0, 1.4, 1.05], threshold=0.05
    )
    assert channel == 1
    assert delta == pytest.approx(0.4)
    assert runner_up == pytest.approx(0.05)


def test_pick_changed_channel_below_threshold_returns_none():
    channel, _, _ = pick_changed_channel([1.0, 1.0], [1.0, 1.01], threshold=0.05)
    assert channel is None


class CoupledFakeSensor:
    """Fake sensor whose channel voltages track specific motor positions.

    Simulates the physical coupling discovery relies on: moving the motor that
    drives a joint changes exactly that joint's encoder channel.
    """

    is_connected = True

    def __init__(self, hand, motor_to_channel, base=1.6, scale=0.5):
        self._hand = hand
        self._m2c = motor_to_channel
        self._base = base
        self._scale = scale

    def average_voltages(self, num_samples):
        voltages = [self._base] * 16
        positions = self._hand.get_motor_pos(as_dict=True)
        for motor_id, channel in self._m2c.items():
            voltages[channel] = self._base + self._scale * positions[motor_id]
        return voltages

    def read_voltages(self):
        return self.average_voltages(1)

    def start_stream(self):
        pass

    def stop_stream(self):
        pass

    def disconnect(self):
        pass


def test_discover_joint_sensor_map_recovers_ground_truth(model_with_sensors):
    cfg = model_with_sensors({"index_mcp": 5})
    hand = MockOrcaHand(config_path=cfg)
    hand.connect()

    ground_truth = {"index_mcp": 7, "thumb_mcp": 3, "ring_pip": 11}
    motor_to_channel = {
        hand.config.joint_to_motor_map[j]: ch for j, ch in ground_truth.items()
    }
    hand._joint_sensor_client = CoupledFakeSensor(hand, motor_to_channel)

    results = hand.discover_joint_sensor_map(
        joints=list(ground_truth),
        motor_delta=0.5,
        settle_time=0.0,
        num_samples=3,
        threshold=0.05,
    )
    discovered = {j: results[j]["channel"] for j in ground_truth}
    assert discovered == ground_truth


def test_discover_joint_sensor_map_requires_sensor(model_with_sensors):
    cfg = model_with_sensors({"index_mcp": 5})
    hand = MockOrcaHand(config_path=cfg)
    hand.connect()
    with pytest.raises(RuntimeError, match="Joint sensor is not connected"):
        hand.discover_joint_sensor_map()


def test_required_slope_sign():
    # Joint and voltage move together -> positive slope keeps them aligned.
    assert required_slope_sign(10.0, 0.2) == 1
    # Joint up but voltage down -> slope must be negative.
    assert required_slope_sign(10.0, -0.2) == -1
    # Undeterminable: no joint reference, or movement below the noise floor.
    assert required_slope_sign(None, 0.2) is None
    assert required_slope_sign(1e-9, 0.2) is None
    assert required_slope_sign(10.0, 1e-9) is None


class CoupledJointSensor:
    """Fake sensor whose channels track joint angles with a configurable sign.

    ``sensor_signs[joint] = -1`` simulates an encoder mounted so its voltage runs
    opposite to the actual joint angle; discovery should then report ``-1``.
    """

    is_connected = True

    def __init__(self, hand, joint_to_channel, sensor_signs, base=1.6, scale=0.02):
        self._hand = hand
        self._j2c = joint_to_channel
        self._signs = sensor_signs
        self._base = base
        self._scale = scale
        self.slope_signs = {}

    def average_voltages(self, num_samples):
        voltages = [self._base] * 16
        positions = self._hand.get_joint_position().as_dict()
        for joint, channel in self._j2c.items():
            pos = positions.get(joint) or 0.0
            voltages[channel] = self._base + self._signs[joint] * self._scale * pos
        return voltages

    def read_voltages(self):
        return self.average_voltages(1)

    def set_slope_sign(self, signs):
        self.slope_signs.update(signs)

    def start_stream(self):
        pass

    def stop_stream(self):
        pass

    def disconnect(self):
        pass


def _calibrated_hand(model_with_sensors, joint_to_channel):
    """Build a MockOrcaHand with the probed joints calibrated (real direction)."""
    cfg = model_with_sensors(dict(joint_to_channel))
    pre = MockOrcaHand(config_path=cfg)
    motor_ids = {pre.config.joint_to_motor_map[j] for j in joint_to_channel}
    cal = str(Path(cfg).parent / "calibration.yaml")
    update_yaml(cal, "motor_limits", {mid: [0.0, 6.0] for mid in motor_ids})
    update_yaml(cal, "joint_to_motor_ratios", {mid: 0.05 for mid in motor_ids})
    hand = MockOrcaHand(config_path=cfg)
    hand.connect()
    return hand


def test_discover_joint_sensor_map_recovers_slope_sign(model_with_sensors):
    ground_truth = {"index_mcp": 7, "thumb_mcp": 3, "ring_pip": 11}
    sensor_signs = {"index_mcp": 1, "thumb_mcp": -1, "ring_pip": -1}

    hand = _calibrated_hand(model_with_sensors, ground_truth)
    hand._joint_sensor_client = CoupledJointSensor(hand, ground_truth, sensor_signs)

    results = hand.discover_joint_sensor_map(
        joints=list(ground_truth), motor_delta=0.5, settle_time=0.0,
        num_samples=3, threshold=0.05,
    )
    assert {j: results[j]["channel"] for j in ground_truth} == ground_truth
    assert {j: results[j]["slope_sign"] for j in ground_truth} == sensor_signs


def test_set_joint_sensor_slope_signs_persists_and_applies(model_with_sensors):
    cfg = model_with_sensors({"index_mcp": 5, "thumb_mcp": 2})
    hand = MockOrcaHand(config_path=cfg)
    client = FakeJointSensorClient([0.0] * 16)
    hand._joint_sensor_client = client

    applied = hand.set_joint_sensor_slope_signs(
        {"index_mcp": -1, "thumb_mcp": 1, "unmapped": -1, "index_mcp_bad": 5},
        persist=True,
    )
    assert applied == {"index_mcp": -1, "thumb_mcp": 1}
    # Applied live, keyed by channel.
    assert client.slope_signs == {5: -1, 2: 1}
    # Persisted to calibration.yaml.
    cal = read_yaml(str(Path(cfg).parent / "calibration.yaml"))
    assert cal["joint_sensor_slope_sign"] == {"index_mcp": -1, "thumb_mcp": 1}


def test_apply_joint_sensor_slope_sign_on_connect(model_with_sensors):
    cfg = model_with_sensors({"index_mcp": 5, "thumb_mcp": 2})
    update_yaml(
        str(Path(cfg).parent / "calibration.yaml"),
        "joint_sensor_slope_sign",
        {"index_mcp": -1, "thumb_mcp": 1},
    )
    hand = MockOrcaHand(config_path=cfg)
    client = FakeJointSensorClient([0.0] * 16)
    hand._joint_sensor_client = client

    applied = hand._apply_joint_sensor_slope_sign()
    assert applied == 2
    assert client.slope_signs == {5: -1, 2: 1}
