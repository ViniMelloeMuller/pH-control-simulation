import argparse
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score as r2
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.systems.system import pHSystem


def foptd_model(t: np.ndarray, Kp: float, tau: float, td: float) -> np.ndarray:
    """
    First Order Plus Dead Time model:
        y(t) = K*(1 - exp(-(t - td) / tau)) for t >= td
        y(t) = 0 for t < td
    Args:
        t: Time vector
        Kp: Process gain
        tau: Time constant
        td: Dead time
    """
    y = np.zeros_like(t)
    idx = t >= td
    y[idx] = Kp * (1 - np.exp(-(t[idx] - td) / tau))
    return y


def fit_foptd(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fits a First Order Plus Dead Time model to the data

    Returns:
        (Kp, tau, td) parameters
    """

    Kp0 = np.max(y)
    td0 = t[np.where(y >= 0.1 * Kp0)[0][0]] if np.any(y >= 0.1 * Kp0) else 0
    tau0 = 0.5 * (t[-1] - td0) if t[-1] > td0 else 1

    p0 = [Kp0, tau0, td0]
    bounds = ([0, 1e-6, 0], [np.inf, np.inf, np.inf])

    popt, _ = curve_fit(foptd_model, t, y, p0=p0, bounds=bounds)
    return tuple(popt)


def main():
    parser = argparse.ArgumentParser(description="Fit a FOPTD model to a step response")
    parser.add_argument(
        "--plot", action="store_true", help="Plot the step response and the fit"
    )
    args = parser.parse_args()

    # Defines the simulation parameters
    system = pHSystem(dt=0.5)
    u_step = system._u1ss + 1.0  # -> unit step

    for _ in range(1000):
        system.step(u=np.array([u_step, system._u2ss]))

    tfit = np.array(system.data["t"])
    yfit = np.array(system.data["y"])
    yfit -= yfit[0]
    Kp, tau, td = fit_foptd(tfit, yfit)

    yhat = foptd_model(tfit, Kp, tau, td)

    print(f"Kp: {Kp}, tau: {tau}, td: {td}")

    # Cohen and Coon (1953) tuning
    Kc = 1.35 / Kp * (tau / td + 0.185)
    tau_I = 2.5 * td * (tau + 0.185 * td) / (tau + 0.611 * td)
    tau_D = 0.37 * td * tau / (tau + 0.185 * td)
    PID_params = {"Kc": Kc, "tau_I": tau_I, "tau_D": tau_D}

    print(f"Kc: {Kc}, tau_I: {tau_I}, tau_D: {tau_D}")
    print(f"R2: {r2(yfit, yhat)}")

    os.makedirs("models", exist_ok=True)
    params_filepath = os.path.join("models", "pid_params.json")
    with open(params_filepath, "w") as f:
        json.dump(PID_params, f, indent=4)
    print("parametros salvos em ", params_filepath)

    if args.plot:
        plt.figure(figsize=(8, 5))
        plt.plot(tfit, yfit, "b--", label="Simulated data")
        plt.plot(tfit, yhat, "k-", label="FOPTD")
        plt.legend()
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("(y - y_0)")

        plt.savefig("results/FOPTD.png", format="png", dpi=800, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
