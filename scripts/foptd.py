import numpy as np 
from numpy.typing import NDArray
from . import DynamicalSystem
from scipy.optimize import curve_fit

class FOPTDModel:
    def __init__(self, K: float, tau: float, theta: float, dt: float=0.5):
        self.K = K
        self.tau = tau
        self.theta = theta

        self.dt = dt
        self.y = 0

        self.u_history = []

        self.data = {
            "t": [0],
            "y": [self.y],
            "u": [15.55],
        }
        pass

    def step(self, u: NDArray | None = None) -> NDArray:
        self.u_history.append(u)
        delay_steps = int(self.theta / self.dt)

        if (len(self.u_history) > delay_steps):
            u_delayed = self.u_history[-(-delay_steps + 1)]
        else:
            u_delayed = 0

        dy = (self.K * u_delayed - self.y) * (self.dt / self.tau)
        self.y += dy

        # Logs the data
        self.data["t"].append(self.data["t"][-1] + self.dt)
        self.data["y"].append(self.y)
        self.data["u"].append(u)

        pass
    
    def reset(self):
        self.u_history = []
        self.data = {
            "t": [0],
            "y": [0],
            "u": [0],
        }
        pass




def foptd(t: list[float], K: float, tau: float, theta: float):
    """ Modelo FOPTD do sistema
    Args:
        t: tempo
        Kp: Ganho do processo
        tau: constante de tempo
        td: tempo morto
    """
    response = np.zeros_like(t)
    idx = t >= theta
    response[idx] = K * (1 - np.exp(-(t[idx] - theta) / tau))
    return response


def fit_foptd(y_real: NDArray, time: NDArray) -> NDArray:
    y_real = y_real - y_real[0] # Variável desvio

    K_initial = y_real.max() / (15.55 + 1 - 15.55)
    tau_initial = (time[np.argmax(y_real > 0.632 * y_real.max())] - time[0])
    theta_initial = time[np.argmax(y_real > 0.05 * y_real.max())]

    initial_guess = [K_initial, tau_initial, theta_initial]
    
    popt, _ = curve_fit(foptd, time, y_real, p0=initial_guess)
    return popt

