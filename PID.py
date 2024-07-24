import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import tqdm
from tqdm import tqdm
from scipy.optimize import fsolve, minimize
from scipy.signal import savgol_filter

# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [9/2.4, 9/2.4]
# Condições nominais
u3ss = 16.60  # ml/s
u2ss = 0.55  # ml/s
u1ss = 15.55  # ml/s
V = 2900  # ml
Wa1 = -3.05e-3  # mol
Wa2 = -3e-2  # mol
Wa3 = 3.0e-3  # mol
Wa = -4.32e-4  # mol
Wb1 = 5e-5  # mol
Wb2 = 3e-2  # mol
Wb3 = 0.0  # mol
Wb = 5.28e-4  # mol
pk1 = 6.35
pk2 = 10.25
yss = 7.0


def dxdt(x, u):
    f = np.array([[u3ss / V * (Wa3 - x[0])], [u3ss / V * (Wb3 - x[1])]])
    g = np.array([[1 / V * (Wa1 - x[0])], [1 / V * (Wb1 - x[1])]])
    p = np.array([[1 / V * (Wa2 - x[0])], [1 / V * (Wb2 - x[1])]])
    return f + g * u[0] + p * u[1]


def y_f(x, x0):
    h = (
        lambda y: x[0]
        + 10 ** (y - 14)
        - 10 ** (-y)
        + x[1] * (1 + 2 * 10 ** (y - pk2)) / (1 + 10 ** (pk1 - y) + 10 ** (y - pk2))
    )
    y = fsolve(h, x0=x0)[0]
    return y


def x_next(x, u, dt):
    k1 = dxdt(x, u).flatten()
    k2 = dxdt(x + dt / 2 * k1, u).flatten()
    k3 = dxdt(x + dt / 2 * k2, u).flatten()
    k4 = dxdt(x + dt * k3, u).flatten()
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Metodo de Z-N em Malha Aberta


def main():
    print("Iniciando teste em malha aberta")

    t = np.arange(0, 1000 + 0.5, 0.5)
    dt = t[1] - t[0]

    x1 = np.zeros((t.shape[0], 2))
    x2 = np.zeros((t.shape[0], 2))

    u1 = np.zeros((t.shape[0], 2))
    u2 = np.zeros((t.shape[0], 2))

    y1 = np.zeros(t.shape[0])
    y2 = np.zeros(t.shape[0])

    y1[0] = y_f(x1[0], x0=7.0)
    y2[0] = y_f(x2[0], x0=7.0)

    u1[:, :] = [u1ss, u2ss]
    u2[:, :] = [u1ss, u2ss]

    u1[100:, 0] *= 1.10
    u2[100:, 0] *= 0.90

    x1[0, :] = [Wa, Wb]
    x2[0, :] = [Wa, Wb]

    for k in range(0, t.shape[0] - 1):
        x1[k + 1, :] = x_next(x1[k], u1[k], dt)
        x2[k + 1, :] = x_next(x2[k], u2[k], dt)
        y1[k + 1] = y_f(x1[k + 1], x0=y1[k])
        y2[k + 1] = y_f(x2[k + 1], x0=y2[k])

    e = np.random.normal(scale=0.1, size=len(y1))
    y = y1 + e
    tfit = t[100:] - t[100]
    yfit = y[100:] - 7.0

    A = u1[100, 0] - u1ss

    def f_obj(params):
        K, tau, td = params
        y_foptd = K * A * (1 - np.exp(-(tfit - td) / tau)) * (tfit >= td)
        cost = np.sum(np.square(y_foptd - yfit))
        return cost

    res = minimize(f_obj, x0=[2, 3, 0.1], method="SLSQP", bounds=[(0, np.inf)])
    print("Ajustando Modelo FOPTD...")
    K, tau, td = res.x

    y_model = K * A * (1 - np.exp(-(tfit - td) / tau)) * (tfit >= td)
    print(f"y(s)/u(s) = {K:.2f}/({tau:.2f}s + 1) e^-{td:.2f}s")

    Kc = 5 * (tau / K / td)
    tau_D = 1.0 * td
    tau_I = 2 * td
    print(f"Kc={Kc:2f}, tau_D={tau_D:.2f}, tau_I={tau_I:.2f}")

    # PID
    # Control loop
    import time

    dt = 0.5
    k = 8
    noise_scale = 0.05
    t_sim = np.arange(0, 200 + dt, dt)

    X_PID = np.zeros((t_sim.shape[0], 2))
    U_PID = np.zeros((t_sim.shape[0], 2))
    Y_PID = np.zeros(t_sim.shape[0])
    E_PID = np.zeros(t_sim.shape[0])

    ysp = np.ones(t_sim.shape[0])
    ysp[:] = 7.0
    ysp[10:] = 8.0
    ysp[144:] = 7.0
    ysp[288:] = 6.5
    np.savetxt("results/ysp.csv", ysp, delimiter=",")
    Y_PID[0] = y_f(X_PID[0], x0=7.0)
    U_PID[:, :] = [u1ss, u2ss]
    X_PID[0, :] = [Wa, Wb]

    CPU_times = []
    print("SIMULATING SERVO...")
    for n in tqdm(range(0, k)):
        X_PID[n + 1, :] = x_next(X_PID[n], U_PID[n], dt)
        Y_PID[n + 1] = y_f(X_PID[n + 1], x0=Y_PID[n]) + np.random.normal(
            scale=noise_scale
        )

    I = 0.0
    for n in tqdm(range(k, t_sim.shape[0] - 1)):
        start = time.process_time()

        E_PID[n] = ysp[n] - Y_PID[n]

        P = Kc * E_PID[n]
        I += 1 / tau_I * E_PID[n] * dt

        filtered = savgol_filter(E_PID[:n], min(n, 15), 2, 1)[-1]

        D = tau_D * filtered

        U = u1ss + P + I + D
        U_PID[n, 0] = min(max(U, 0.5 * u1ss), 1.5 * u1ss)
        X_PID[n + 1, :] = x_next(
            X_PID[n],
            U_PID[n],
            dt,
        )
        Y_PID[n + 1] = y_f(X_PID[n + 1], x0=Y_PID[n]) + np.random.normal(
            scale=noise_scale
        )
        CPU_times.append(time.process_time() - start)

    IAE = np.sum(np.abs(ysp - Y_PID) * dt)
    EC = np.sum(np.abs(np.diff(U_PID[:, 0])))

    plt.figure(figsize=(18 / 2.4, 9 / 2.4))
    plt.axis([0, 200, 6.0, 8.5])
    string = (
        rf"$\Delta t = ${dt} s - $\delta$ = {noise_scale}"
        + f"\nIAE={IAE:.2f}"
        + f"\nEC={EC:.2f}"
    )
    plt.text(100, 8.0, string, fontsize=10)
    plt.plot(t_sim, Y_PID, c="k")
    plt.step(t_sim, ysp, ls="--", c="k")
    plt.savefig("results/PID/PID_sim.pdf", bbox_inches="tight")

    dataset_PID = np.column_stack((t_sim, U_PID, Y_PID))
    np.savetxt("results/PID/PID_servo.csv", dataset_PID, delimiter=",")
    print("RESULTS:")
    print(f"IAE={IAE:.2f}\tEC={EC:.2f}")

    ##################################################################

    # GENERATING REGULATORY LOOP

    ##################################################################
    dt = 0.5
    k = 8
    noise_scale = 0.05
    t_sim2 = np.arange(0, 200 + dt, dt)

    X_PID2 = np.zeros((t_sim2.shape[0], 2))
    U_PID2 = np.zeros((t_sim2.shape[0], 2))
    Y_PID2 = np.zeros(t_sim2.shape[0])
    E_PID2 = np.zeros(t_sim2.shape[0])

    ysp2 = np.ones(t_sim2.shape[0])
    ysp2[:] = 7.0
    np.savetxt("results/ysp_reg.csv", ysp2, delimiter=",")
    Y_PID2[0] = y_f(X_PID2[0], x0=7.0)
    U_PID2[:, :] = [u1ss, u2ss]
    U_PID2[:, 1] = np.loadtxt("data/openLoop.csv", delimiter=",")[:, 2]

    X_PID2[0, :] = [Wa, Wb]

    CPU_times2 = []
    print("SIMULATING REGULATORY...")
    for n in tqdm(range(0, k)):
        X_PID2[n + 1, :] = x_next(X_PID2[n], U_PID2[n], dt)
        Y_PID2[n + 1] = y_f(X_PID2[n + 1], x0=Y_PID2[n]) + np.random.normal(
            scale=noise_scale
        )

    I = 0.0
    for n in tqdm(range(k, t_sim2.shape[0] - 1)):
        start = time.process_time()

        E_PID2[n] = ysp2[n] - Y_PID2[n]

        P = Kc * E_PID2[n]
        I += 1 / tau_I * E_PID2[n] * dt

        filtered = savgol_filter(E_PID2[:n], min(n, 15), 2, 1)[-1]

        D = tau_D * filtered

        U = u1ss + P + I + D
        U_PID2[n, 0] = min(max(U, 0.5 * u1ss), 1.5 * u1ss)
        X_PID2[n + 1, :] = x_next(
            X_PID2[n],
            U_PID2[n],
            dt,
        )
        Y_PID2[n + 1] = y_f(X_PID2[n + 1], x0=Y_PID2[n]) + np.random.normal(
            scale=noise_scale
        )
        CPU_times2.append(time.process_time() - start)

    IAE2 = np.sum(np.abs(ysp2 - Y_PID2) * dt)
    EC2 = np.sum(np.abs(np.diff(U_PID2[:, 0])))

    plt.figure(figsize=(18 / 2.4, 9 / 2.4))
    plt.axis([0, 200, 6.0, 8.5])
    string = (
        rf"$\Delta t = ${dt} s - $\delta$ = {noise_scale}"
        + f"\nIAE={IAE2:.2f}"
        + f"\nEC={EC2:.2f}"
    )
    plt.text(100, 8.0, string, fontsize=10)
    plt.plot(t_sim2, Y_PID2, c="k")
    plt.step(t_sim2, ysp2, ls="--", c="k")
    plt.savefig("results/PID/PID_reg.pdf", bbox_inches="tight")

    dataset_PID2 = np.column_stack((t_sim2, U_PID2, Y_PID2))
    np.savetxt("results/PID/PID_reg.csv", dataset_PID2, delimiter=",")
    print("RESULTS:")
    print(f"IAE={IAE2:.2f}\tEC={EC2:.2f}")
    ##################################################################

    # GENERATING EXPERIMENTAL DATA

    ##################################################################
    print("GENERATING TRAINING DATASET")
    # PID
    # Control loop
    import time

    dt = 0.5
    k = 8
    noise_scale = 0.05
    t_sim = np.arange(0, 50000 + dt, dt)

    X_PID = np.zeros((t_sim.shape[0], 2))
    U_PID = np.zeros((t_sim.shape[0], 2))
    Y_PID = np.zeros(t_sim.shape[0])
    E_PID = np.zeros(t_sim.shape[0])

    ysp = np.ones(t_sim.shape[0])

    for ki in range(t_sim.shape[0]):
        if ki % int(150 / dt) == 0:
            ysp[ki:] = np.random.uniform(low=6.0, high=9.5)

    # np.savetxt("ysp2.csv", ysp, delimiter=",");

    Y_PID[0] = y_f(X_PID[0], x0=7.0)
    U_PID[:, :] = [u1ss, u2ss]

    for ki in range(t_sim.shape[0]):
        if ki % int(150 / dt) == 0:
            U_PID[ki:, 1] = np.random.uniform(low=0.0, high=5.0)

    X_PID[0, :] = [Wa, Wb]

    CPU_times = []

    for n in tqdm(range(0, k)):
        X_PID[n + 1, :] = x_next(X_PID[n], U_PID[n], dt)
        Y_PID[n + 1] = y_f(X_PID[n + 1], x0=Y_PID[n]) + np.random.normal(
            scale=noise_scale
        )

    I = 0.0
    for n in tqdm(range(k, t_sim.shape[0] - 1)):
        start = time.process_time()

        E_PID[n] = ysp[n] - Y_PID[n]

        P = Kc * E_PID[n]
        I += 1 / tau_I * E_PID[n] * dt

        filtered_D = savgol_filter(E_PID[:n], min(n, 15), 2, 1)[-1]
        # numerical_dif = np.diff(E_PID)[-1]

        D = tau_D * filtered_D

        U = u1ss + P + I + D
        U_PID[n, 0] = min(max(U, 0.5 * u1ss), 1.5 * u1ss)
        X_PID[n + 1, :] = x_next(
            X_PID[n],
            U_PID[n],
            dt,
        )
        Y_PID[n + 1] = y_f(X_PID[n + 1], x0=Y_PID[n]) + np.random.normal(
            scale=noise_scale
        )
        CPU_times.append(time.process_time() - start)

    dataset_PID = np.column_stack((t_sim, U_PID, Y_PID))
    np.savetxt("data/PID_data.csv", dataset_PID, delimiter=",")
    print("DONE")
    print(f"GENERATED {t_sim.shape[0]/dt/60/60:.2f} HOURS OF TRAINING DATA")

    print("GENERATING VALIDATION/TEST DATASET")

    t_sim = np.arange(0, 15000 + dt, dt)

    X_PID = np.zeros((t_sim.shape[0], 2))
    U_PID = np.zeros((t_sim.shape[0], 2))
    Y_PID = np.zeros(t_sim.shape[0])
    E_PID = np.zeros(t_sim.shape[0])

    ysp = np.ones(t_sim.shape[0])

    for ki in range(t_sim.shape[0]):
        if ki % int(100 / dt) == 0:
            ysp[ki:] = np.random.uniform(low=6.5, high=8.5)

    Y_PID[0] = y_f(X_PID[0], x0=7.0)
    U_PID[:, :] = [u1ss, u2ss]
    for ki in range(t_sim.shape[0]):
        if ki % int(50 / dt) == 0:
            U_PID[ki:, 1] = np.random.uniform(low=0.0, high=5.0)

    X_PID[0, :] = [Wa, Wb]

    CPU_times = []

    for n in tqdm(range(0, k)):
        X_PID[n + 1, :] = x_next(X_PID[n], U_PID[n], dt)
        Y_PID[n + 1] = y_f(X_PID[n + 1], x0=Y_PID[n]) + np.random.normal(
            scale=noise_scale
        )

    I = 0.0
    for n in tqdm(range(k, t_sim.shape[0] - 1)):
        start = time.process_time()

        E_PID[n] = ysp[n] - Y_PID[n]

        P = Kc * E_PID[n]
        I += 1 / tau_I * E_PID[n] * dt

        filtered_D = savgol_filter(E_PID[:n], min(n, 15), 2, 1)[-1]
        # numerical_dif = np.diff(E_PID)[-1]

        D = tau_D * filtered_D

        U = u1ss + P + I + D
        U_PID[n, 0] = min(max(U, 0.5 * u1ss), 1.5 * u1ss)
        X_PID[n + 1, :] = x_next(
            X_PID[n],
            U_PID[n],
            dt,
        )
        Y_PID[n + 1] = y_f(X_PID[n + 1], x0=Y_PID[n]) + np.random.normal(
            scale=noise_scale
        )
        CPU_times.append(time.process_time() - start)

    dataset_PID = np.column_stack((t_sim, U_PID, Y_PID))
    np.savetxt("data/PID_data_test.csv", dataset_PID, delimiter=",")

    print("DONE")
    print(f"GENERATED {t_sim.shape[0]/dt/60/60:.2f} HOURS OF TEST DATA")


if __name__ == "__main__":
    main()
