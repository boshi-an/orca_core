# ==============================================================================
# Copyright (c) 2025 ORCA
#
# This file is part of ORCA and is licensed under the MIT License.
# You may use, copy, modify, and distribute this file under the terms of the MIT License.
# See the LICENSE file at the root of this repository for full license information.
# ==============================================================================

from dataclasses import dataclass
from typing import Dict, List

from .utils.utils import read_yaml
from .constants import (
    MOTOR_LIMITS_DICT,
    JOINT_TO_MOTOR_RATIOS,
    WRIST_CALIBRATED,
    CALIBRATED,
    JOINT_SENSOR_ZERO,
    JOINT_SENSOR_RANGE,
    JOINT_SENSOR_RANGE_CALIBRATED,
)


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable snapshot of a hand's calibration state.

    Produced by the calibration routine and stored on the hand instance.
    Replacing the instance attribute is the only way to update calibration
    state — the internals are never mutated in place.

    Attributes:
        motor_limits_dict: Maps motor ID → ``[lower, upper]`` hard limits (rad).
            Values are ``None`` before the corresponding joint is calibrated.
        joint_sensor_limits_dict: Maps motor ID → ``[lower, upper]`` joint sensor limits (rad).
        joint_to_motor_ratios_dict: Maps motor ID → rad/rad gear ratio.
            Zero before calibration.
        calibrated: ``True`` when all joints have been fully calibrated.
        wrist_calibrated: ``True`` when the wrist joint has been calibrated.
        joint_sensor_range_calibrated: ``True`` when the joint sensor range has been calibrated.
    """

    motor_limits_dict: Dict[int, List]
    joint_sensor_limits_dict: Dict[int, List]
    joint_to_motor_ratios_dict: Dict[int, float]
    calibrated: bool
    wrist_calibrated: bool
    joint_sensor_range_calibrated: bool

    @classmethod
    def empty(cls, motor_ids: List[int]) -> "CalibrationResult":
        """Return a blank (uncalibrated) result for the given motor IDs."""
        return cls(
            motor_limits_dict={mid: [None, None] for mid in motor_ids},
            joint_sensor_limits_dict={mid: [None, None] for mid in motor_ids},
            joint_to_motor_ratios_dict={mid: 0.0 for mid in motor_ids},
            calibrated=False,
            wrist_calibrated=False,
            joint_sensor_range_calibrated=False,
        )

    @classmethod
    def from_calibration_path(
        cls,
        calibration_path: str,
        motor_ids: List[int],
    ) -> "CalibrationResult":
        """Load calibration state from a ``calibration.yaml`` file.

        Returns an :meth:`empty` result for any fields absent from the file.

        Args:
            calibration_path: Absolute path to ``calibration.yaml``.
            motor_ids: Ordered list of motor IDs; used to build the dicts.
        """
        calibration = read_yaml(calibration_path) or {}

        motor_limits_raw = calibration.get(MOTOR_LIMITS_DICT, {})
        motor_limits_dict = {
            mid: motor_limits_raw.get(mid, [None, None]) for mid in motor_ids
        }
        joint_sensor_limits_dict = {
            mid: calibration.get(JOINT_SENSOR_RANGE, {}).get(mid, [None, None])
            for mid in motor_ids
        }

        ratios_raw = calibration.get(JOINT_TO_MOTOR_RATIOS, {})
        joint_to_motor_ratios_dict = {
            mid: ratios_raw.get(mid, 0.0) for mid in motor_ids
        }

        return cls(
            motor_limits_dict=motor_limits_dict,
            joint_sensor_limits_dict=joint_sensor_limits_dict,
            joint_to_motor_ratios_dict=joint_to_motor_ratios_dict,
            calibrated=calibration.get(CALIBRATED, False) or False,
            wrist_calibrated=calibration.get(WRIST_CALIBRATED, False) or False,
            joint_sensor_range_calibrated=calibration.get(
                JOINT_SENSOR_RANGE_CALIBRATED, False
            ) or False
        )
