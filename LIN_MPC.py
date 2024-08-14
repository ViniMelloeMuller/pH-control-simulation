from PID import *
import numpy as np
from LIN_model import series_to_supervised
import pickle
import time

noise_scale = 0.05
with open("models/LIN_model.pkl", "rb") as f:
    linear_model = pickle.load(f)


def MPC_optimization_Linear(y0, u0, Nc, Np, ysp, model=linear_model):
    k = y0.shape[0]
    y = np.copy(y0)
    u = np.copy(u0)

    bounds = [(0.5 * u1ss, 1.5 * u1ss)] * (Nc)

    def f_obj(u_futuros, u=u, y=y):
        u1_futuros = np.empty(Np)
        u1_futuros[:Nc] = u_futuros  # Apply optimized control moves
        u1_futuros[Nc:] = u_futuros[-1]  # Keep control constant after Nc

        u2_futuros = np.ones(Np) * u[-1, 1]
        u_total = np.vstack((u, np.column_stack((u1_futuros, u2_futuros))))
        y_pred = np.zeros(Np)
        y_total = np.append(y, y_pred)

        for l in range(k, k + Np):
            data_input = np.column_stack((u_total[l - k : l], y_total[l - k : l]))
            model_input = series_to_supervised(data_input, n_in=k - 1).values
            y_pred[l - k] = model.predict(model_input)[0]
            y_total[l] = y_pred[l - k]

        Q, R = 1000, 8e-0
        ISE = Q * np.sum((y_pred - ysp) ** 2)
        EC = R * np.sum((np.diff(u_total[:, 0])) ** 2)
        return ISE + EC

    U0 = np.ones(Nc) * u0[-1, 0]
    u_futuros = minimize(f_obj, x0=U0, bounds=bounds, method="SLSQP").x

    return u_futuros[0]


def main():
    # Linear MPC
    # Control loop - SERVO
    k = 2
    t_sim = np.loadtxt("results/PID/PID_servo.csv", delimiter=",")[:, 0]
    dt = t_sim[1] - t_sim[0]

    X_lin = np.zeros((t_sim.shape[0], 2))
    U_lin = np.zeros((t_sim.shape[0], 2))
    Y_lin = np.zeros(t_sim.shape[0])
    E_lin = np.zeros(t_sim.shape[0])

    ysp = np.loadtxt("results/ysp.csv", delimiter=",")

    Y_lin[0] = y_f(X_lin[0], x0=7.0)
    U_lin[:, :] = [u1ss, u2ss]
    X_lin[0, :] = [Wa, Wb]

    CPU_times = []
    print("STARTING SERVO CONTROL")
    for n in tqdm(range(0, 10)):
        X_lin[n + 1, :] = x_next(X_lin[n], U_lin[n], dt)
        Y_lin[n + 1] = y_f(X_lin[n + 1], x0=Y_lin[n]) + np.random.normal(
            scale=noise_scale
        )

    for n in tqdm(range(10, t_sim.shape[0] - 1)):
        start = time.process_time()
        U = MPC_optimization_Linear(
            y0=Y_lin[n - k : n], u0=U_lin[n - k : n], Nc=3, Np=10, ysp=ysp[n]
        )
        U_lin[n, 0] = U
        X_lin[n + 1, :] = x_next(X_lin[n], U_lin[n], dt)
        Y_lin[n + 1] = y_f(X_lin[n + 1], x0=Y_lin[n]) + np.random.normal(
            scale=noise_scale
        )
        CPU_times.append(time.process_time() - start)

    dataset_LIN = np.column_stack((t_sim, U_lin, Y_lin))
    np.savetxt("results/LIN/LIN_servo.csv", dataset_LIN, delimiter=",")
    print("DONE")
    print()
    print("RESULTS:")
    IAE = np.sum(np.abs(ysp - Y_lin) * dt)
    EC = np.sum(np.abs(np.diff(U_lin[:, 0])))
    print(f"IAE={IAE:.2f}\tEC={EC:.2f}")

    ############################################################################

    k = 2
    t_sim = np.loadtxt("results/PID/PID_reg.csv", delimiter=",")[:, 0]
    dt = t_sim[1] - t_sim[0]

    X_lin = np.zeros((t_sim.shape[0], 2))
    U_lin = np.zeros((t_sim.shape[0], 2))
    Y_lin = np.zeros(t_sim.shape[0])

    ysp = np.ones(t_sim.shape[0]) * 7.0

    X_lin[0, :] = [Wa, Wb]
    Y_lin[0] = y_f(X_lin[0], x0=7.0)
    U_lin[:, :] = [u1ss, u2ss]
    U_lin[:, 1] = np.loadtxt("results/PID/PID_reg.csv", delimiter=",")[:, 2]

    CPU_times = []
    print("STARTING REGULATORY CONTROL")
    for n in tqdm(range(0, 10)):
        X_lin[n + 1, :] = x_next(X_lin[n], U_lin[n], dt)
        Y_lin[n + 1] = y_f(X_lin[n + 1], x0=Y_lin[n]) + np.random.normal(
            scale=noise_scale
        )

    for n in tqdm(range(10, t_sim.shape[0] - 1)):
        start = time.process_time()
        U = MPC_optimization_Linear(
            y0=Y_lin[n - k : n], u0=U_lin[n - k : n], Nc=3, Np=10, ysp=ysp[n]
        )
        U_lin[n, 0] = U
        X_lin[n + 1, :] = x_next(X_lin[n], U_lin[n], dt)
        Y_lin[n + 1] = y_f(X_lin[n + 1], x0=Y_lin[n]) + np.random.normal(
            scale=noise_scale
        )
        CPU_times.append(time.process_time() - start)

    dataset_LIN2 = np.column_stack((t_sim, U_lin, Y_lin))
    np.savetxt("results/LIN/LIN_reg.csv", dataset_LIN2, delimiter=",")
    print("DONE")
    print()
    print("RESULTS:")
    IAE = np.sum(np.abs(ysp - Y_lin) * dt)
    EC = np.sum(np.abs(np.diff(U_lin[:, 0])))


if __name__ == "__main__":
    main()
