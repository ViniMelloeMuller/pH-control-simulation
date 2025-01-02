################################################################################
# THIS FILE CONTAINS THE IMPLEMENTATION OF THE DYNAMICAL SYSTEM AS A CLASS
################################################################################
import numpy as np 
import pandas as pd 
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.signal import savgol_filter

class DynamicalSystem:
    """ Implements the pH system """
    def __init__(self, x0: ArrayLike | None = None, dt: float = 0.5):
        """
        Args:
            x0: Initial state of the system
            dt: Fixed delta time between states
        """
        self.u3ss = 16.60  # ml/s
        self.u2ss = 0.55  # ml/s
        self.u1ss = 15.55  # ml/s

        self.V    = 2900  # ml

        self.Wa1  = -3.05e-3  # mol
        self.Wa2  = -3e-2  # mol
        self.Wa3  = 3.0e-3  # mol
        self.Wa   = -4.32e-4  # mol
        self.Wb1  = 5e-5  # mol
        self.Wb2  = 3e-2  # mol
        self.Wb3  = 0.0  # mol
        self.Wb   = 5.28e-4  # mol
        self.pk1  = 6.35
        self.pk2  = 10.25
        self.yss  = 7.0

        self.dt = dt

        if x0 is None:
            self.x0 = np.array([self.Wa, self.Wb])
        else:
            self.x0 = x0

        self.x = self.x0

        self.data = {
            "t": [0],
            "x": [self.x],
            "y": [self.y],
            "u1": [self.u1ss],
            "u2": [self.u2ss],
        }

    def dxdt(self, x: NDArray, u: NDArray) -> NDArray:
        """ Computes the derivative of the state """
        f = np.array([[self.u3ss / self.V * (self.Wa3 - x[0])], [self.u3ss / self.V * (self.Wb3 - x[1])]])
        g = np.array([[1 / self.V * (self.Wa1 - x[0])], [1 / self.V * (self.Wb1 - x[1])]])
        p = np.array([[1 / self.V * (self.Wa2 - x[0])], [1 / self.V * (self.Wb2 - x[1])]])
        return f + g * u[0] + p * u[1]

    @property
    def y(self) -> float:
        """ Function that updates the pH based on the current state 
        Args:
            x0: Initial guess for the pH
        """

        def h(pH):
            return self.x[0] + 10 ** (pH - 14) - 10 ** (-pH) + self.x[1] * (1 + 2 * 10 ** (pH - self.pk2)) / (1 + 10 ** (self.pk1 - pH) + 10 ** (pH - self.pk2))

        pH = fsolve(h, x0=7.0)[0]
        return pH

    def step(self, u: NDArray | None = None) -> NDArray:
        """ Updates the internal system using the Runge-Kutta 4th order method 
        Args:
            u: Input to the system (Control action). If none, then defaults to the steady state
        """
        if u is None:
            u = np.array([self.u1ss, self.u2ss])

        k1 = self.dxdt(self.x, u).flatten()
        k2 = self.dxdt(self.x + self.dt / 2 * k1, u).flatten()
        k3 = self.dxdt(self.x + self.dt / 2 * k2, u).flatten()
        k4 = self.dxdt(self.x + self.dt * k3, u).flatten()

        # Update the state
        self.x = self.x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Logs the data
        self.data["t"].append(self.data["t"][-1] + self.dt)
        self.data["x"].append(self.x)
        self.data["y"].append(self.y)
        self.data["u1"].append(u[0])
        self.data["u2"].append(u[1])

        pass

    def reset(self):
        """ Resets the system to the initial state """
        self.x = self.x0
        self.data = {
            "t": [0],
            "x": [self.x],
            "y": [self.y],
            "u1": [self.u1ss],
            "u2": [self.u2ss],
        }
        pass

    def plot(self, ax:plt.Axes|None = None):
        """ Plots the data """
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(18 / 2.4, 9 / 2.4))

        ax2 = ax[1].twinx()

        ax[0].plot(self.data["t"], self.data["y"], "-k");
        ax[1].step(self.data["t"], self.data["u1"], "-r");
        ax2.step(self.data["t"], self.data["u2"], "-b");

        ax_list = [ax[0], ax[1], ax2]
        
        return ax_list

    @property
    def get_df(self) -> pd.DataFrame:
        """ Returns the data as a pandas DataFrame """
        df = pd.DataFrame(self.data)
        df.set_index("t", inplace=True)
        return df

class PIDController:
    def __init__(self, Kc: float, tau_I: float, tau_D: float, dt: float):
        self.Kc = Kc
        self.tau_I = tau_I
        self.tau_D = tau_D
        self.dt = dt
        self.I = 0.0

        self.previous_setpoint = 0
        self.error_list = []
        self.setpoint_list = []

    def update(self, ysp: float, y: float) -> float:
        if self.previous_setpoint != ysp:
            self.I = 0.0

        self.previous_setpoint = ysp
        self.setpoint_list.append(ysp)

        E = ysp - y
        P = self.Kc * E

        self.I += 1 / self.tau_I * E * self.dt

        if len(self.error_list) < 10:
            D = 0
        else:
            filtered = savgol_filter(self.error_list, window_length=10, polyorder=2, deriv=1, delta=self.dt)[-1]
            D = self.tau_D * filtered

        self.error_list.append(E)

        return P + self.I + D

    def reset(self):
        self.I = 0.0
        self.error_list = []
        self.setpoint_list = []
        pass

    @property
    def performance(self) -> tuple[float, float]:
        IAE = np.sum(np.abs(self.error_list) * self.dt)
        EC = np.sum(np.abs(np.diff(self.error_list)))
        return IAE, EC