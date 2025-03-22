from . import Controller
from numpy.typing import NDArray
from scipy.signal import savgol_filter
import numpy as np
import time


class PIDController(Controller):
    def __init__(self, dt: float, Kc: float, tau_I: float, tau_D: float):
        """
        Args:
            dt: Discrete time step
            Kc: Proportional gain
            tau_I: Integral time constant
            tau_D: Derivative time constant
        """
        super().__init__(dt)
        self.Kc = Kc
        self.tau_I = tau_I
        self.tau_D = tau_D

        self.I = 0.0
        self.previous_setpoint = 0

    def policy(self, state: NDArray) -> float:
        """
        The PID policy is given by:

        u(t) = Kc * e(t) + Ki * sum(e(t)) + Kd * d/dt (e(t))

        Where:
        * e(t) is the error at time t
        * Kc is the proportional gain
        * Ki is the integral gain
        * Kd is the derivative gain

        Args:
            state: Current state of the system in the form [y(t), ysp(t)]
        """

        start_time = time.perf_counter()

        if (
            self.previous_setpoint != state[1]
        ):  # Reset the integral term when setpoint change
            self.I = 0.0

        et = state[1] - state[0]
        self.error_list.append(et)

        self.I += et * self.dt

        # Derivative approximation using Savitzky-Golay filter
        window_length = 5
        if len(self.error_list) < window_length:
            de_dt = 0.0
        else:
            de_dt = savgol_filter(
                self.error_list, window_length, 2, deriv=1, delta=self.dt
            )[-1]

        u_bar = 15.55  # Stationary value of the action
        u_pid = u_bar + self.Kc * (et + 1 / self.tau_I * self.I + self.tau_D * de_dt)

        end_time = time.perf_counter()

        self.action_list.append(u_pid)
        self.computational_time.append(end_time - start_time)

        return u_pid

    def reset(self):
        self.I = 0.0
        self.error_list = []
        self.action_list = []
        self.computational_time = []
        self.previous_setpoint = 0
