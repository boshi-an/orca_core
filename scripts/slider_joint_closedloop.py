#!/usr/bin/env python
"""Closed-loop, joint-by-joint control of the ORCA hand with a Tkinter UI.

Unlike ``slider_joint.py`` (open loop — it commands a motor target derived from
the joint→motor calibration and trusts it), this script reads the *actual* joint
angle from the magnetic-encoder board and adjusts each joint's command until the
sensed angle matches the slider target. This corrects for tendon stretch,
calibration drift, and other compliance the open-loop path ignores.

Per sensed joint, an independent PID current controller runs each tick::

    error       = target - sensed
    integral   += error * dt                     # anti-windup clamped
    derivative  = -d(sensed)/dt                   # on measurement, low-pass filtered
    command     = kp*error + ki*integral + kd*derivative   # mA, clamped to limits

Derivative is taken on the measurement (not the error) so moving a slider does
not cause a derivative kick, and the integral is clamped to the actuation range
to prevent wind-up. The command is sent through ``set_joint_current`` (motors
held in ``current`` mode). Joints without an encoder channel command zero
current. When no encoder is available at all, the UI falls back to open-loop
``position`` control, commanding the slider targets directly.

Requirements:
    - A ``joint_sensors`` mapping (``joint_to_sensor_id``) so sensed angles can
      be matched to joints. Defaults to the bundled template.
    - ``joint_current_limits`` in the config (used to clamp the command).
    - The encoder board connected. If it can't be reached the UI still runs in
      open-loop position mode.

Usage:
    uv run python scripts/slider_joint_closedloop.py [config.yaml] [--kp 50] [--rate 20]
"""

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

# Low-pass factor for the derivative term (0 = no filtering, ->1 = heavy).
DERIV_FILTER_ALPHA = 0.6

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orca_core import OrcaHand


class ClosedLoopControlUI:
    def __init__(self, root, hand, sensor_ok, kp=0.3, kd=0.0, ki=0.0, rate_hz=20.0):
        self.hand = hand
        # self.joint_roms = hand.config.joint_roms_dict
        self.joint_current_limits = hand.config.joint_current_dict
        self.joint_ids = hand.config.joint_ids
        self.joint_sensor_range = {
            joint: hand.calibration.joint_sensor_limits_dict[hand.config.joint_to_motor_map[joint]]
            for joint in self.joint_ids
        }
        self.sensed_joints = set(hand.config.joint_to_sensor_id) if sensor_ok else set()
        # Closed loop needs both a live encoder and a joint→channel mapping.
        self.sensor_ok = sensor_ok and bool(self.sensed_joints)

        self.period_ms = max(1, int(1000.0 / rate_hz))

        # target_vars: slider value (deg). command: PID current output (mA).
        self.target_vars = {j: tk.DoubleVar() for j in self.joint_ids}
        self.command = {}
        # Per-joint PID state.
        self.integral = {}
        self.prev_measured = {}
        self.deriv = {}
        self._last_tick_t = None
        self.kp_var = tk.DoubleVar(value=kp)
        self.kd_var = tk.DoubleVar(value=kd)
        self.ki_var = tk.DoubleVar(value=ki)
        self.closed_loop_var = tk.BooleanVar(value=sensor_ok)
        self.running_var = tk.BooleanVar(value=False)
        # Operating mode is switched lazily and only when it changes, never per
        # tick (a mode switch cycles torque and writes EEPROM).
        self._active_mode = None

        self.sensed_labels = {}
        self.error_labels = {}

        self._seed()
        self._build_ui(root)
        self._tick_id = root.after(self.period_ms, self._control_tick)
        self.root = root

    # -- setup --------------------------------------------------------------

    def _seed(self):
        """Reset commands and PID state to a clean starting point."""
        for joint in self.joint_ids:
            self.command[joint] = 0.0
            self.integral[joint] = 0.0
            self.prev_measured[joint] = None
            self.deriv[joint] = 0.0
        self._last_tick_t = None

    def _build_ui(self, root):
        root.title("ORCA Hand — Closed-Loop Joint Control")
        root.geometry("640x720")

        top = ttk.Frame(root)
        top.pack(fill=tk.X, pady=8, padx=8)

        ttk.Button(top, text="Enable Torque", command=self._enable_torque).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Disable Torque", command=self._disable_torque).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(top, text="Run control", variable=self.running_var).pack(side=tk.LEFT, padx=12)

        cl = ttk.Checkbutton(top, text="Closed loop", variable=self.closed_loop_var)
        cl.pack(side=tk.LEFT, padx=4)
        if not self.sensor_ok:
            cl.state(["disabled"])

        for label, var in (("kp", self.kp_var), ("ki", self.ki_var), ("kd", self.kd_var)):
            ttk.Label(top, text=label).pack(side=tk.LEFT, padx=(12, 2))
            ttk.Spinbox(
                top, from_=0.0, to=200.0, increment=0.5, width=5, textvariable=var
            ).pack(side=tk.LEFT)

        status = "encoder connected" if self.sensor_ok else "NO encoder — open loop only"
        self.status_label = ttk.Label(root, text=f"Status: {status}")
        self.status_label.pack(anchor=tk.W, padx=10)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(header, text="joint", width=14).pack(side=tk.LEFT)
        ttk.Label(header, text="target", width=22, anchor=tk.CENTER).pack(side=tk.LEFT)
        ttk.Label(header, text="tgt", width=7, anchor=tk.E).pack(side=tk.LEFT)
        ttk.Label(header, text="sensed", width=8, anchor=tk.E).pack(side=tk.LEFT)
        ttk.Label(header, text="err", width=8, anchor=tk.E).pack(side=tk.LEFT)

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        for joint in self.joint_ids:
            rom_min = self.joint_sensor_range[joint][0]
            rom_max = self.joint_sensor_range[joint][1]
            row = ttk.Frame(body)
            row.pack(fill=tk.X, pady=2)

            sensed_tag = " (CL)" if joint in self.sensed_joints else ""
            ttk.Label(row, text=joint + sensed_tag, width=14).pack(side=tk.LEFT)

            ttk.Scale(
                row, from_=rom_min, to=rom_max, orient=tk.HORIZONTAL,
                variable=self.target_vars[joint], length=180,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

            tgt = ttk.Label(row, width=7, anchor=tk.E)
            tgt.pack(side=tk.LEFT)
            self.target_vars[joint].trace_add(
                "write", lambda *a, j=joint, lbl=tgt: lbl.config(text=f"{self.target_vars[j].get():.1f}")
            )
            tgt.config(text=f"{self.target_vars[joint].get():.1f}")

            sensed = ttk.Label(row, width=8, anchor=tk.E, text="—")
            sensed.pack(side=tk.LEFT)
            err = ttk.Label(row, width=8, anchor=tk.E, text="—")
            err.pack(side=tk.LEFT)
            self.sensed_labels[joint] = sensed
            self.error_labels[joint] = err

    # -- actions ------------------------------------------------------------

    def _enable_torque(self):
        self.hand.enable_torque()
        self._seed()
        print("Torque enabled.")

    def _disable_torque(self):
        self.running_var.set(False)
        self.hand.disable_torque()
        print("Torque disabled.")

    # -- control loop -------------------------------------------------------

    def _control_tick(self):
        try:
            self._step()
        except Exception as e:  # keep the UI alive on transient hardware errors
            print(f"Control tick error: {e}")
        finally:
            self._tick_id = self.root.after(self.period_ms, self._control_tick)

    def _set_mode(self, mode):
        """Switch operating mode only on change; reset commands on entry."""
        if mode == self._active_mode:
            return
        self.hand.set_control_mode(mode)
        self._active_mode = mode
        self._seed()

    def _step(self):
        if not self.running_var.get():
            # Paused: drop the timestamp so control resumes with a fresh dt
            # instead of integrating over the whole pause.
            self._last_tick_t = None
            return

        if self.closed_loop_var.get() and self.sensor_ok:
            self._closed_loop_current_step()
        else:
            self._open_loop_position_step()

    def _closed_loop_current_step(self):
        """PID current controller: kp*error + ki*integral + kd*derivative."""
        self._set_mode("current")
        sensed = self.hand.get_sensed_joint_positions().as_dict()
        kp, ki, kd = self.kp_var.get(), self.ki_var.get(), self.kd_var.get()

        now = time.perf_counter()
        dt = 0.0 if self._last_tick_t is None else now - self._last_tick_t
        self._last_tick_t = now

        for joint in self.joint_ids:
            measured = sensed.get(joint)
            if measured is None:
                # No feedback for this joint -> command no torque, hold no state.
                self.command[joint] = 0.0
                self.integral[joint] = 0.0
                self.prev_measured[joint] = None
                self.deriv[joint] = 0.0
                continue

            target = self.target_vars[joint].get()
            error = target - measured
            cur_min, cur_max = self.joint_current_limits[joint]

            # Derivative on the (filtered) measurement to avoid setpoint kick.
            if dt > 0.0 and self.prev_measured[joint] is not None:
                raw_deriv = -(measured - self.prev_measured[joint]) / dt
                self.deriv[joint] = (
                    DERIV_FILTER_ALPHA * self.deriv[joint]
                    + (1.0 - DERIV_FILTER_ALPHA) * raw_deriv
                )
            self.prev_measured[joint] = measured

            # Integrate with anti-windup: keep ki*integral within the limits.
            if dt > 0.0 and ki != 0.0:
                self.integral[joint] += error * dt
                lo, hi = sorted((cur_min / ki, cur_max / ki))
                self.integral[joint] = max(lo, min(hi, self.integral[joint]))

            cmd = kp * error + ki * self.integral[joint] + kd * self.deriv[joint]
            self.command[joint] = max(cur_min, min(cur_max, cmd))

            self.sensed_labels[joint].config(text=f"{measured:.1f}")
            self.error_labels[joint].config(text=f"{error:+.1f}")

        self.hand.set_joint_current(dict(self.command))

    def _open_loop_position_step(self):
        """No encoder: command slider targets straight through in position mode."""
        self._set_mode("position")
        for joint in self.joint_ids:
            self.command[joint] = self.target_vars[joint].get()
        self.hand.set_joint_positions(dict(self.command))

    def shutdown(self):
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        self.hand.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Closed-loop joint control UI for the ORCA hand.")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to config.yaml")
    parser.add_argument("--kp", type=float, default=2, help="Proportional gain, mA per deg error (default 2)")
    parser.add_argument("--kd", type=float, default=0.0, help="Derivative gain, mA per deg/s error (default 0)")
    parser.add_argument("--ki", type=float, default=0.0, help="Integral gain, mA per deg*s error (default 0)")
    parser.add_argument("--rate", type=float, default=20.0, help="Control rate in Hz (default 20)")
    parser.add_argument("--no-sensors", action="store_true", help="Skip the encoder; open loop only")
    args = parser.parse_args()

    hand = OrcaHand(config_path=args.config_path)

    ok, msg = hand.connect()
    print(msg)
    if not ok:
        print("Failed to connect to the hand.")
        return

    hand.init_joints(force_calibrate=False)

    sensor_ok = False
    if not args.no_sensors:
        sensor_ok, sensor_msg = hand.connect_joint_sensors(start_stream=True)
        print(sensor_msg)
    if not sensor_ok:
        print("Running in OPEN-LOOP mode (no encoder feedback).")

    root = tk.Tk()
    ui = ClosedLoopControlUI(
        root, hand, sensor_ok,
        kp=args.kp, kd=args.kd, ki=args.ki, rate_hz=args.rate,
    )
    root.protocol("WM_DELETE_WINDOW", lambda: (ui.shutdown(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
