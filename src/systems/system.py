from . import DynamicalSystem
import numpy as np
from numpy.typing import NDArray, ArrayLike
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.optimize import fsolve


class pHSystem(DynamicalSystem):
    def __init__(self, dt: float, x0: NDArray | None = None):
        """
        Args:
        dt: Fixed delta time between states
        x0: Initial state of the system
        """
        self._u3ss = 16.60  # ml/s
        self._u2ss = 0.55  # ml/s
        self._u1ss = 15.55  # ml/s
        self._V = 2900  # ml
        self._Wa1 = -3.05e-3  # mol
        self._Wa2 = -3e-2  # mol
        self._Wa3 = 3.0e-3  # mol
        self._Wa = -4.32e-4  # mol
        self._Wb1 = 5e-5  # mol
        self._Wb2 = 3e-2  # mol
        self._Wb3 = 0.0  # mol
        self._Wb = 5.28e-4  # mol
        self._pk1 = 6.35
        self._pk2 = 10.25
        self._yss = 7.0

        self.dt = dt

        self.x0 = np.array([self._Wa, self._Wb]) if x0 is None else x0
        self.x = self.x0

        self.data = {
            "t": [0],
            "x": [self.x],
            "y": [self.y],
            "u1": [self._u1ss],
            "u2": [self._u2ss],
        }

    def dxdt(self, x: NDArray, u: NDArray) -> NDArray:
        """Computes the derivative of the state"""
        f = np.array(
            [
                [self._u3ss / self._V * (self._Wa3 - x[0])],
                [self._u3ss / self._V * (self._Wb3 - x[1])],
            ]
        )
        g = np.array(
            [
                [1 / self._V * (self._Wa1 - x[0])],
                [1 / self._V * (self._Wb1 - x[1])],
            ]
        )
        p = np.array(
            [
                [1 / self._V * (self._Wa2 - x[0])],
                [1 / self._V * (self._Wb2 - x[1])],
            ]
        )
        return f + g * u[0] + p * u[1]

    @property
    def y(self) -> float:
        """
        Function that updates the pH value based on the current state
        Args:
            x0: Initial guess for the pH
        """

        def h(pH):
            return (
                self.x[0]
                + 10 ** (pH - 14)
                - 10 ** (-pH)
                + self.x[1]
                * (1 + 2 * 10 ** (pH - self._pk2))
                / (1 + 10 ** (self._pk1 - pH) + 10 ** (pH - self._pk2))
            )

        pH = fsolve(h, x0=7.0)[0]
        return pH

    def step(self, u: NDArray | None = None):
        """
        Updates the internal system using the Runge-Kutta 4th order method
        Args:
            u: Input to the system (Control action). If none, then defaults to
            the steady state values
        """
        if u is None:
            u = np.array([self._u1ss, self._u2ss])

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
        """Resets the system to the initial state"""
        self.x = self.x0
        self.data = {
            "t": [0],
            "x": [self.x],
            "y": [self.y],
            "u1": [self._u1ss],
            "u2": [self._u2ss],
        }
        pass

    def plot(self, ax: Axes | None = None, setpoint_list: list[float] | None = None):
        """Plots the data"""
        if ax is None:
            fig, ax_ = plt.subplots(2, 1, figsize=(18 / 2.4, 9 / 2.4))
        else:
            ax_ = ax

        ax2 = ax_[1].twinx()

        if setpoint_list is not None:
            ax_[0].step(self.data["t"], setpoint_list, "--")

        ax_[0].plot(self.data["t"], self.data["y"], "-k")
        ax_[1].step(self.data["t"], self.data["u1"], "-r")
        ax2.step(self.data["t"], self.data["u2"], "-b")
        ax_list = [ax_[0], ax_[1], ax2]

        return ax_list

    @property
    def get_df(self) -> pd.DataFrame:
        """Returns the data as a pandas DataFrame"""
        df = pd.DataFrame(self.data)
        df.set_index("t", inplace=True)
        return df
