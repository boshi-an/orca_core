#!/usr/bin/env python
"""Closed-loop, joint-by-joint control of the ORCA hand with a Tkinter UI.

Unlike ``slider_joint.py`` (open loop — it commands a motor target derived from
the joint→motor calibration and trusts it), this script reads the *actual* joint
angle from the magnetic-encoder board and adjusts each joint's command until the
sensed angle matches the slider target. This corrects for tendon stretch,
calibration drift, and other compliance the open-loop path ignores.

Two closed-loop methods are selectable at runtime:

*Current PID* — an independent PID current controller per sensed joint::

    error       = target - sensed
    integral   += error * dt                     # anti-windup clamped
    derivative  = -d(sensed)/dt                   # on measurement, low-pass filtered
    command     = kp*error + ki*integral + kd*derivative   # mA, clamped to limits

Derivative is taken on the measurement (not the error) so moving a slider does
not cause a derivative kick, and the integral is clamped to the actuation range
to prevent wind-up. The command is sent through ``set_joint_current`` (motors
held in ``current`` mode). Joints without an encoder channel command zero
current.

*Position I* — the motors run in ``current_based_position`` mode (position
control with a current limit) and an outer integral loop trims the commanded
position so the *sensed* joint angle tracks the target::

    command = target + ki*integral    # deg, clamped to ROM

When no encoder is available at all, the UI falls back to open-loop
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
    # Closed-loop algorithms selectable at runtime.
    #   current  -> PID whose output is a motor current  ("current" mode)
    #   position -> PI whose output corrects the position command
    #               ("current_based_position" mode)
    METHODS = (("current", "Current PID"), ("position", "Position I"))

    def __init__(self, root, hand, sensor_ok, kp=0.3, kd=0.0, ki=0.0,
                 ki_pos=0.0, rate_hz=20.0):
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

        self.period_s = 1.0 / rate_hz
        self.period_ms = max(1, int(1000.0 / rate_hz))
        # Timing: absolute deadline scheduling + a smoothed measured-rate readout.
        self._next_deadline = 0.0
        self._last_wall_t = None
        self.measured_hz = None

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
        # Integral gain for the position-I method (different units/scale).
        self.ki_pos_var = tk.DoubleVar(value=ki_pos)
        self.method_var = tk.StringVar(value="current")
        self.closed_loop_var = tk.BooleanVar(value=sensor_ok)
        self.running_var = tk.BooleanVar(value=False)
        # Operating mode is switched lazily and only when it changes, never per
        # tick (a mode switch cycles torque and writes EEPROM).
        self._active_mode = None

        self.sensed_labels = {}
        self.error_labels = {}
        self.current_bars = {}

        self._seed()
        self._build_ui(root)
        self.root = root
        self._next_deadline = time.perf_counter()
        self._tick_id = root.after(self.period_ms, self._control_tick)

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
        root.geometry("760x760")

        style = ttk.Style(root)
        # Green bar = live/current sensed position (target is the draggable slider).
        style.configure("Current.Horizontal.TProgressbar", background="#2e9e4f")

        top = ttk.Frame(root)
        top.pack(fill=tk.X, pady=8, padx=8)

        ttk.Button(top, text="Enable Torque", command=self._enable_torque).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Disable Torque", command=self._disable_torque).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(top, text="Run control", variable=self.running_var).pack(side=tk.LEFT, padx=12)

        cl = ttk.Checkbutton(top, text="Closed loop", variable=self.closed_loop_var)
        cl.pack(side=tk.LEFT, padx=4)
        if not self.sensor_ok:
            cl.state(["disabled"])

        ttk.Label(top, text="mode").pack(side=tk.LEFT, padx=(12, 2))
        for key, text in self.METHODS:
            rb = ttk.Radiobutton(
                top, text=text, value=key, variable=self.method_var,
                command=self._on_method_changed,
            )
            rb.pack(side=tk.LEFT)
            if not self.sensor_ok:
                rb.state(["disabled"])

        # Each method shows only its own gains; switching swaps the group in place.
        self.gains_frame = ttk.Frame(top)
        self.gains_frame.pack(side=tk.LEFT)

        self.current_gains = ttk.Frame(self.gains_frame)
        for label, var in (("kp", self.kp_var), ("ki", self.ki_var), ("kd", self.kd_var)):
            ttk.Label(self.current_gains, text=label).pack(side=tk.LEFT, padx=(12, 2))
            ttk.Spinbox(
                self.current_gains, from_=0.0, to=200.0, increment=0.5, width=5, textvariable=var
            ).pack(side=tk.LEFT)

        self.position_gains = ttk.Frame(self.gains_frame)
        ttk.Label(self.position_gains, text="ki").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Spinbox(
            self.position_gains, from_=0.0, to=20.0, increment=0.1, width=5,
            textvariable=self.ki_pos_var,
        ).pack(side=tk.LEFT)

        self._on_method_changed()

        statusbar = ttk.Frame(root)
        statusbar.pack(fill=tk.X, padx=10)
        status = "encoder connected" if self.sensor_ok else "NO encoder — open loop only"
        self.status_label = ttk.Label(statusbar, text=f"Status: {status}")
        self.status_label.pack(side=tk.LEFT)
        self.freq_label = ttk.Label(
            statusbar, text=f"Rate: — Hz (target {1.0 / self.period_s:.0f} Hz)", anchor=tk.E
        )
        self.freq_label.pack(side=tk.RIGHT)

        legend = ttk.Frame(root)
        legend.pack(fill=tk.X, padx=10, pady=(2, 0))
        ttk.Label(legend, text="slider = target    ").pack(side=tk.LEFT)
        ttk.Label(legend, text="■", foreground="#2e9e4f").pack(side=tk.LEFT)
        ttk.Label(legend, text=" = current (sensed)").pack(side=tk.LEFT)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(header, text="joint", width=14).pack(side=tk.LEFT)
        ttk.Label(header, text="target / current", width=30, anchor=tk.CENTER).pack(side=tk.LEFT)
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

            # Paired sliding bars: an interactive Scale for the target and a
            # read-only Progressbar underneath tracking the sensed position.
            bars = ttk.Frame(row, width=200)
            bars.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

            ttk.Scale(
                bars, from_=rom_min, to=rom_max, orient=tk.HORIZONTAL,
                variable=self.target_vars[joint],
            ).pack(fill=tk.X, expand=True)

            current_bar = ttk.Progressbar(
                bars, orient=tk.HORIZONTAL, mode="determinate",
                maximum=1000, style="Current.Horizontal.TProgressbar",
            )
            current_bar.pack(fill=tk.X, expand=True, pady=(1, 0))
            self.current_bars[joint] = current_bar

            tgt = ttk.Label(row, width=7, anchor=tk.E)
            tgt.pack(side=tk.LEFT)
            self.target_vars[joint].trace_add(
                "write",
                lambda *a, j=joint, lbl=tgt: self._on_target_changed(j, lbl),
            )
            self._on_target_changed(joint, tgt)

            sensed = ttk.Label(row, width=8, anchor=tk.E, text="—")
            sensed.pack(side=tk.LEFT)
            err = ttk.Label(row, width=8, anchor=tk.E, text="—")
            err.pack(side=tk.LEFT)
            self.sensed_labels[joint] = sensed
            self.error_labels[joint] = err

    def _bar_fraction(self, joint, value):
        """Map a joint value onto 0..1000 within its ROM (for a Progressbar)."""
        lo, hi = self.joint_sensor_range[joint]
        if hi == lo:
            return 0
        frac = (value - lo) / (hi - lo)
        return int(1000 * max(0.0, min(1.0, frac)))

    def _on_target_changed(self, joint, label):
        label.config(text=f"{self.target_vars[joint].get():.1f}")

    def _on_method_changed(self):
        """Show the selected method's gains and reset PID state on a switch."""
        self.current_gains.pack_forget()
        self.position_gains.pack_forget()
        if self.method_var.get() == "position":
            self.position_gains.pack(side=tk.LEFT)
        else:
            self.current_gains.pack(side=tk.LEFT)
        self._seed()

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
        now = time.perf_counter()
        if self._last_wall_t is not None:
            interval = now - self._last_wall_t
            if interval > 0.0:
                inst_hz = 1.0 / interval
                self.measured_hz = (
                    inst_hz if self.measured_hz is None
                    else 0.8 * self.measured_hz + 0.2 * inst_hz
                )
                self.freq_label.config(
                    text=f"Rate: {self.measured_hz:5.1f} Hz (target {1.0 / self.period_s:.0f} Hz)"
                )
        self._last_wall_t = now

        try:
            self._step()
        except Exception as e:  # keep the UI alive on transient hardware errors
            print(f"Control tick error: {e}")
        finally:
            # Schedule off an absolute deadline so per-tick work doesn't make the
            # loop drift slower than the requested rate. If we've fallen behind,
            # resync instead of firing a burst of catch-up ticks.
            self._next_deadline += self.period_s
            delay = self._next_deadline - time.perf_counter()
            if delay < 0.0:
                self._next_deadline = time.perf_counter()
                delay = 0.0
            self._tick_id = self.root.after(max(1, round(delay * 1000)), self._control_tick)

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
            if self.method_var.get() == "position":
                self._closed_loop_position_step()
            else:
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
                self.current_bars[joint]["value"] = 0
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
            self.current_bars[joint]["value"] = self._bar_fraction(joint, measured)

        self.hand.set_joint_current(dict(self.command))

    def _closed_loop_position_step(self):
        """Integral controller in current_based_position mode.

        The motor runs its own position loop (with a current limit); this outer
        integral term trims the commanded position by ``ki*integral`` so the
        *sensed* joint angle tracks the slider target despite tendon stretch and
        calibration drift.
        """
        self._set_mode("current_based_position")
        sensed = self.hand.get_sensed_joint_positions().as_dict()
        ki = self.ki_pos_var.get()

        now = time.perf_counter()
        dt = 0.0 if self._last_tick_t is None else now - self._last_tick_t
        self._last_tick_t = now

        for joint in self.joint_ids:
            target = self.target_vars[joint].get()
            measured = sensed.get(joint)
            if measured is None:
                # No feedback: command the raw target (position control only).
                self.command[joint] = target
                self.integral[joint] = 0.0
                self.current_bars[joint]["value"] = 0
                continue

            error = target - measured
            rom_lo, rom_hi = self.joint_sensor_range[joint]

            # Anti-windup: keep the integral's position correction within ROM span.
            if dt > 0.0 and ki != 0.0:
                self.integral[joint] += error * dt
                span = abs(rom_hi - rom_lo)
                lo, hi = sorted((-span / ki, span / ki))
                self.integral[joint] = max(lo, min(hi, self.integral[joint]))

            self.command[joint] = target + ki * self.integral[joint]

            self.sensed_labels[joint].config(text=f"{measured:.1f}")
            self.error_labels[joint].config(text=f"{error:+.1f}")
            self.current_bars[joint]["value"] = self._bar_fraction(joint, measured)

        # set_joint_positions clamps commands to configured ROM bounds.
        self.hand.set_joint_positions(dict(self.command))

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
    parser.add_argument("--ki-pos", type=float, default=0.0,
                        help="Position-I integral gain, deg cmd per deg*s error (default 0)")
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
        kp=args.kp, kd=args.kd, ki=args.ki,
        ki_pos=args.ki_pos, rate_hz=args.rate,
    )
    root.protocol("WM_DELETE_WINDOW", lambda: (ui.shutdown(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
