from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray


class DynamicalSystem(ABC):
    def __init__(self, dt: float, x0: ArrayLike | None = None):
        """
        Constructor of the DynamicalSystem class
        Args:
            dt: Discrete time step
        """

    @abstractmethod
    def step(self, u: float | NDArray | None):
        """
        This should be used to update the state of the system based on the
        current action. That is, the dynamics of the system.
        x_{t+1} = f(x_t, u_t)
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the system to its initial state
        """
        pass
