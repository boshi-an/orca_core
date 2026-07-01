# ==============================================================================
# Copyright (c) 2025 ORCA
#
# This file is part of ORCA and is licensed under the MIT License.
# You may use, copy, modify, and distribute this file under the terms of the MIT License.
# See the LICENSE file at the root of this repository for full license information.
# ==============================================================================
"""Client for the Teensy magnetic-encoder joint-position sensors.

Thin wrapper around the pure-python ``SensorReader`` shipped in
``third_party/orca_sensor/pure_python/sensor_reader.py``. The wrapper keeps the
ORCA-facing API small (connect / read / background-stream) while delegating all
serial framing and calibration to ``SensorReader`` so calibration stays in one
place.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from orca_core.hardware.sensing.constants import (
    DEFAULT_JOINT_SENSOR_BAUDRATE,
    DEFAULT_JOINT_SENSOR_PORT,
    NUM_JOINT_SENSORS,
)

logger = logging.getLogger(__name__)


def _default_sensor_reader_dir() -> str:
    """Return the bundled ``third_party`` directory holding ``sensor_reader.py``."""
    here = os.path.dirname(os.path.abspath(__file__))
    # orca_core/hardware/sensing -> repo root is three levels up.
    repo_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    return os.path.join(repo_root, "third_party", "orca_sensor", "pure_python")


def _import_sensor_reader(reader_dir: Optional[str] = None):
    """Import and return the ``SensorReader`` class from ``third_party``.

    Args:
        reader_dir: Directory containing ``sensor_reader.py``. Defaults to the
            bundled ``third_party/orca_sensor/pure_python`` directory.

    Raises:
        ImportError: If the module or one of its dependencies is unavailable.
    """
    reader_dir = reader_dir or _default_sensor_reader_dir()
    if reader_dir not in sys.path:
        sys.path.insert(0, reader_dir)
    try:
        from sensor_reader import SensorReader  # type: ignore  # noqa: E402
    except ImportError as e:
        raise ImportError(
            "Could not import the joint-position SensorReader from "
            f"{reader_dir!r}. Ensure third_party/orca_sensor is present and that "
            "its dependencies (pyserial, pyyaml, numpy, pandas) are installed."
        ) from e
    return SensorReader


def map_angles_to_joints(
    angles: List[float],
    joint_to_sensor_id: Dict[str, int],
) -> Dict[str, float]:
    """Map a raw per-channel angle list to configured joint names.

    Args:
        angles: Angles ordered by sensor id (index ``i`` is sensor ``i``).
        joint_to_sensor_id: Mapping from joint name to the sensor channel that
            measures it.

    Returns:
        Dictionary of joint name → angle (degrees) for every joint whose sensor
        channel is present in *angles*.
    """
    joint_angles: Dict[str, float] = {}
    for joint, sensor_id in joint_to_sensor_id.items():
        if 0 <= sensor_id < len(angles):
            joint_angles[joint] = float(angles[sensor_id])
    return joint_angles


def pick_changed_channel(
    baseline: List[float],
    moved: List[float],
    threshold: float = 0.05,
) -> tuple[Optional[int], float, float]:
    """Identify which channel changed most between two voltage snapshots.

    Args:
        baseline: Per-channel voltage before the joint moved.
        moved: Per-channel voltage after the joint moved.
        threshold: Minimum absolute voltage change for a confident match.

    Returns:
        ``(channel, best_delta, runner_up_delta)``. ``channel`` is ``None`` when
        the largest change is below *threshold* (no joint clearly responded).
        ``runner_up_delta`` is the second-largest change, useful for judging how
        cleanly the winning channel stands out.
    """
    deltas = [abs(m - b) for b, m in zip(moved, baseline)]
    if not deltas:
        return None, 0.0, 0.0
    order = sorted(range(len(deltas)), key=lambda i: deltas[i], reverse=True)
    best = order[0]
    best_delta = deltas[best]
    runner_up = deltas[order[1]] if len(order) > 1 else 0.0
    if best_delta < threshold:
        return None, best_delta, runner_up
    return best, best_delta, runner_up


def required_slope_sign(
    delta_joint: Optional[float],
    delta_voltage: float,
    eps: float = 1e-6,
) -> Optional[int]:
    """Return the slope sign that makes the sensed angle track the joint angle.

    Given a probe that changed the *actual* joint angle by ``delta_joint`` (from
    the direction-correct motor mapping) and the encoder channel voltage by
    ``delta_voltage``, the slope sign must satisfy ``sign(slope * dV) ==
    sign(d_joint)`` so that ``angle = slope * (V - V0)`` moves with the joint.

    Args:
        delta_joint: Change in motor-derived joint angle over the probe. ``None``
            (uncalibrated joint) makes the sign undeterminable.
        delta_voltage: Change in the channel's voltage over the probe.
        eps: Movements smaller than this are treated as noise.

    Returns:
        ``+1`` or ``-1``, or ``None`` if either movement was too small to judge.
    """
    if delta_joint is None or abs(delta_joint) < eps or abs(delta_voltage) < eps:
        return None
    return 1 if (delta_joint > 0) == (delta_voltage > 0) else -1


class JointSensorClient:
    """Read joint angles from the Teensy magnetic-encoder board.

    Either poll synchronously with :meth:`read_angles`, or start the background
    reader with :meth:`start_stream` and poll the freshest frame with
    :meth:`get_latest`. The underlying ``SensorReader`` is not thread-safe, so do
    not mix direct reads with an active background stream.
    """

    def __init__(
        self,
        port: str = DEFAULT_JOINT_SENSOR_PORT,
        baudrate: int = DEFAULT_JOINT_SENSOR_BAUDRATE,
        params_dir: Optional[str] = None,
        reader_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize the client.

        Args:
            port: Serial device for the Teensy board.
            baudrate: Serial speed; the Teensy firmware streams at 2 Mbps.
            params_dir: Override for the sensor calibration/params directory.
                ``None`` uses the directory bundled with ``SensorReader``.
            reader_dir: Override for the directory containing ``sensor_reader.py``.
            verbose: Forwarded to ``SensorReader`` for setup/frame diagnostics.
        """
        self.port = port
        self.baudrate = baudrate
        self.params_dir = params_dir
        self.verbose = verbose

        self._reader_cls = _import_sensor_reader(reader_dir)
        self._reader = None
        self._connected = False

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[List[float]] = None
        self._latest_ts: Optional[float] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def num_sensors(self) -> int:
        """Number of joint-angle channels reported by the sensor."""
        if self._reader is not None:
            return self._reader.num_sensors
        return NUM_JOINT_SENSORS

    def connect(self) -> None:
        """Open the serial link to the Teensy board."""
        if self._connected:
            return

        kwargs = dict(
            port=self.port,
            baudrate=self.baudrate,
            connect=True,
            verbose=self.verbose,
        )
        if self.params_dir is not None:
            kwargs["params_dir"] = self.params_dir

        self._reader = self._reader_cls(**kwargs)
        self._connected = True
        logger.info("Connected to joint sensor at %s", self.port)

    def disconnect(self) -> None:
        """Stop streaming (if active) and close the serial link."""
        self.stop_stream()
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:  # noqa: BLE001 - closing should never raise upward
                pass
        self._reader = None
        self._connected = False

    def read_angles(self) -> List[float]:
        """Read one validated frame and return the per-channel joint angles (deg)."""
        if not self._connected or self._reader is None:
            raise RuntimeError("Joint sensor is not connected; call connect() first.")
        return list(self._reader.read_angles())

    def read_voltages(self) -> List[float]:
        """Read one validated frame and return the per-channel voltages (V)."""
        if not self._connected or self._reader is None:
            raise RuntimeError("Joint sensor is not connected; call connect() first.")
        return list(self._reader.read_voltages())

    def average_voltages(self, num_samples: int = 30) -> List[float]:
        """Average ``num_samples`` frames of per-channel voltage.

        Any running background stream is paused for the duration (the underlying
        reader is single-threaded) and resumed afterwards.

        Args:
            num_samples: Number of frames to average (must be >= 1).

        Returns:
            Per-channel mean voltage, ordered by channel id.
        """
        if not self._connected or self._reader is None:
            raise RuntimeError("Joint sensor is not connected; call connect() first.")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")

        was_streaming = self._running.is_set()
        if was_streaming:
            self.stop_stream()
        try:
            self._reader.flush_input()
            acc: Optional[List[float]] = None
            for _ in range(num_samples):
                voltages = self._reader.read_voltages()
                if acc is None:
                    acc = list(voltages)
                else:
                    for i, v in enumerate(voltages):
                        acc[i] += v
            return [v / num_samples for v in acc]
        finally:
            if was_streaming:
                self.start_stream()

    def capture_zero_voltages(self, num_samples: int = 200) -> List[float]:
        """Average per-channel voltage for capturing ``voltage_at_zero``.

        Thin alias of :meth:`average_voltages` with a larger default sample
        count, used while holding the hand at its zero pose.
        """
        return self.average_voltages(num_samples)

    def set_voltage_at_zero(self, overrides: Dict[int, float]) -> None:
        """Override ``voltage_at_zero`` on the high-level sensors by channel id.

        Args:
            overrides: Mapping from channel id to the new ``voltage_at_zero`` (V).
                Channels absent from the sensor pipeline are ignored.
        """
        if not self._connected or self._reader is None:
            raise RuntimeError("Joint sensor is not connected; call connect() first.")
        sensors = self._reader.manager.high_level_sensors
        for channel, voltage in overrides.items():
            sensor = sensors.get(channel)
            if sensor is not None:
                sensor.voltage_at_zero = float(voltage)

    def get_voltage_at_zero(self) -> Dict[int, float]:
        """Return the current ``voltage_at_zero`` of each high-level channel."""
        if not self._connected or self._reader is None:
            raise RuntimeError("Joint sensor is not connected; call connect() first.")
        return {
            channel: float(sensor.voltage_at_zero)
            for channel, sensor in self._reader.manager.high_level_sensors.items()
        }

    def set_slope_sign(self, signs: Dict[int, int]) -> None:
        """Force the *sign* of ``slope`` per channel, keeping the CSV magnitude.

        The slope magnitude stays exactly as loaded from
        ``calib_joint_angle_linear.csv`` (``sensor.slope`` is set from the CSV at
        construction); only its sign is overridden, so the encoder direction can
        be flipped to match the actual joint angle.

        Args:
            signs: Mapping from channel id to ``+1`` or ``-1``. Other values and
                channels absent from the pipeline are ignored.
        """
        if not self._connected or self._reader is None:
            raise RuntimeError("Joint sensor is not connected; call connect() first.")
        sensors = self._reader.manager.high_level_sensors
        for channel, sign in signs.items():
            sensor = sensors.get(channel)
            if sensor is not None and sign in (1, -1):
                sensor.slope = abs(sensor.slope) * sign

    def get_slope(self) -> Dict[int, float]:
        """Return the current ``slope`` (deg/V) of each high-level channel."""
        if not self._connected or self._reader is None:
            raise RuntimeError("Joint sensor is not connected; call connect() first.")
        return {
            channel: float(sensor.slope)
            for channel, sensor in self._reader.manager.high_level_sensors.items()
        }

    def start_stream(self) -> None:
        """Start the background reader that keeps the most recent frame."""
        if not self._connected or self._reader is None:
            raise RuntimeError("Joint sensor is not connected; call connect() first.")
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_stream(self) -> None:
        """Stop the background reader thread."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                # Discard buffered frames so each stored reading is the freshest.
                self._reader.flush_input()
                angles = self._reader.read_angles()
                with self._lock:
                    self._latest = list(angles)
                    self._latest_ts = time.time()
            except Exception as e:  # noqa: BLE001 - keep the loop alive on transient errors
                logger.warning("Joint sensor read failed: %s", e)
                time.sleep(0.01)

    def get_latest(self) -> Tuple[Optional[List[float]], Optional[float]]:
        """Return the most recent ``(angles, timestamp)`` from the stream.

        Returns ``(None, None)`` if no frame has been captured yet.
        """
        with self._lock:
            if self._latest is None:
                return None, None
            return list(self._latest), self._latest_ts
