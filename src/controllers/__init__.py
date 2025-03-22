################################################################################
# Contém as implementações dos controladores utilizados
################################################################################
from numpy.typing import ArrayLike, NDArray
import numpy as np
from abc import ABC, abstractmethod


class Controller(ABC):
    def __init__(self, dt: float):
        """
        Constructor of the Controller class
        Args:
            dt: Discrete time step
        """
        self.dt = dt
        self.error_list = []
        self.action_list = []
        self.computational_time = []

    @abstractmethod
    def policy(self, state: ArrayLike | NDArray) -> float | ArrayLike:
        """
        This should be used to update the control action based on the
        current state. That is, the policy of the controller.
        u = f(x_t)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the controller to its initial state"""
        pass

    @property
    def performance(self) -> dict[str, float]:
        """
        Returns the performance of the controller. The performance of the
        controller is defined by the error list, action list and computational
        time
        """
        IAE = float(
            np.sum([abs(e) for e in self.error_list])
        )  # Integral of the Absolute Error

        ITAE = float(
            np.sum([abs(e) * self.dt for e in self.error_list])
        )  # Integral of the Time-weighted Absolute Error

        EC = float(np.sum(np.abs(np.diff(self.action_list))))  # Control effort

        MCT = float(np.mean(self.computational_time))  # Mean computational time

        performance_metrics = {
            "IAE": IAE,
            "ITAE": ITAE,
            "EC": EC,
            "MCT": MCT,
        }

        return performance_metrics
