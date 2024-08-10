import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PID import *


def get_metrics(data):
    """
    Get the metrics beased on a dataset resulting from the
    simulations.

    Returns -> list(metrics)
    """

    Y_data = data[:, -1]
    U_data = data[:, :-1]

    IAE = np.sum(np.abs(ysp - Y_data) * dt)
    ITAE = np.sum(t_sim * np.abs(ysp - Y_data) * dt)
    EC = np.sum(np.abs(np.diff(U_data[:, 0])) * dt)

    return [IAE, ITAE, EC]


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = [9 / 2.4, 9 / 2.4]

c1, c2, c3 = "#7AF500", "#F53800", "#0132F5"
# common
ysp = np.loadtxt("results/ysp.csv", delimiter=",")
t_sim = np.loadtxt("results/PID/PID_servo.csv", delimiter=",")[:, 0]
dt = t_sim[1] - t_sim[0]
noise_scale = 0.05

# PID - SERVO
PID_data = np.loadtxt("results/PID/PID_servo.csv", delimiter=",")[:, 1:]
U_PID = PID_data[:, :-1]
Y_PID = PID_data[:, -1]

IAE = np.sum(np.abs(ysp - Y_PID) * dt)
EC = np.sum(np.abs(np.diff(U_PID[:, 0])))

fig, ax = plt.subplots(2, 1, figsize=(18 / 2.4, 9 / 2.4), sharex=True)

ax[0].plot(t_sim, Y_PID, c=c1, label="PID")
ax[0].step(t_sim, ysp, c="k", ls="--", label="$y_{sp}$")
ax[0].set_ylabel("pH")
ax[0].legend(ncol=2)

ax[1].step(t_sim, U_PID[:, 0], c=c1)
ax[1].set_xlim([0, 201])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$u_1$ (mL/s)")

string = rf"$\Delta t = ${dt} s - $\delta$ = {noise_scale}" + f"\nIAE={IAE:.2f}"
ax[0].text(100, 7.8, string, fontsize=5)
ax[1].text(100, 20, f"\nEC={EC:.2f}", fontsize=5)

plt.savefig("results/PID/PID_servo_full.pdf", bbox_inches="tight")
# plt.show()


# LINEAR
LIN_data = np.loadtxt("results/LIN/LIN_servo.csv", delimiter=",")[:, 1:]
U_LIN = LIN_data[:, :-1]
Y_LIN = LIN_data[:, -1]

IAE = np.sum(np.abs(ysp - Y_LIN) * dt)
EC = np.sum(np.abs(np.diff(U_LIN[:, 0])))

fig, ax = plt.subplots(2, 1, figsize=(18 / 2.4, 9 / 2.4), sharex=True)

ax[0].plot(t_sim, Y_LIN, c=c2, label="Linear")
ax[0].step(t_sim, ysp, c="k", ls="--", label="$y_{sp}$")
ax[0].set_ylabel("pH")
ax[0].legend(ncol=2)


ax[1].step(t_sim, U_LIN[:, 0], c=c2)
ax[1].set_xlim([0, 201])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$u_1$ (mL/s)")

string = rf"$\Delta t = ${dt} s - $\delta$ = {noise_scale}" + f"\nIAE={IAE:.2f}"
ax[0].text(100, 7.8, string, fontsize=5)
ax[1].text(100, 20, f"\nEC={EC:.2f}", fontsize=5)

plt.savefig("results/LIN/LIN_servo_full.pdf", bbox_inches="tight")
# plt.show()


# GRU
GRU_data = np.loadtxt("results/GRU/GRU_servo.csv", delimiter=",")[:, 1:]
U_GRU = GRU_data[:, :-1]
Y_GRU = GRU_data[:, -1]

IAE = np.sum(np.abs(ysp - Y_GRU) * dt)
EC = np.sum(np.abs(np.diff(U_GRU[:, 0])))

fig, ax = plt.subplots(2, 1, figsize=(18 / 2.4, 9 / 2.4), sharex=True)

ax[0].plot(t_sim, Y_GRU, c=c3, label="GRU")
ax[0].step(t_sim, ysp, c="k", ls="--", label="$y_{sp}$")
ax[0].set_ylabel("pH")
ax[0].legend(ncol=2)


ax[1].step(t_sim, U_GRU[:, 0], c=c3)
ax[1].set_xlim([0, 201])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$u_1$ (mL/s)")

string = rf"$\Delta t = ${dt} s - $\delta$ = {noise_scale}" + f"\nIAE={IAE:.2f}"
ax[0].text(100, 7.8, string, fontsize=5)
ax[1].text(100, 20, f"\nEC={EC:.2f}", fontsize=5)

plt.savefig("results/GRU/GRU_servo_full.pdf", bbox_inches="tight")

# Three happy friends :)
fig, ax = plt.subplots(2, 1, figsize=(18 / 2.4, 9 / 2.4), sharex=True)

ax[0].plot(t_sim, Y_LIN, c=c2, label="Linear")
ax[0].plot(t_sim, Y_PID, c=c1, label="PID")
ax[0].plot(t_sim, Y_GRU, c=c3, label="GRU")
ax[0].step(t_sim, ysp, c="k", ls="--", label="$y_{sp}$")
ax[0].set_ylabel("pH")
ax[0].legend(ncol=4, fontsize=6)


ax[1].step(t_sim, U_LIN[:, 0], c=c2, alpha=0.5)
ax[1].step(t_sim, U_PID[:, 0], c=c1, alpha=0.5)
ax[1].step(t_sim, U_GRU[:, 0], c=c3, alpha=0.5)
ax[1].set_xlim([0, 201])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$u_1$ (mL/s)")

plt.savefig("results/three.pdf", bbox_inches="tight")
# plt.show()


################################################################
# REGULATORY
################################################################

PID_data = np.loadtxt("results/PID/PID_reg.csv", delimiter=",")[:, 1:]
LIN_data = np.loadtxt("results/LIN/LIN_reg.csv", delimiter=",")[:, 1:]
GRU_data = np.loadtxt("results/GRU/GRU_reg.csv", delimiter=",")[:, 1:]
OPL_data = np.loadtxt("data/openLoop.csv", delimiter=",")[:, 1:]

U_OPL = OPL_data[:, :-1]
Y_OPL = OPL_data[:, -1]

U_PID = PID_data[:, :-1]
Y_PID = PID_data[:, -1]

U_LIN = LIN_data[:, :-1]
Y_LIN = LIN_data[:, -1]

U_GRU = GRU_data[:, :-1]
Y_GRU = GRU_data[:, -1]

fig, ax = plt.subplots(2, 1, figsize=(18 / 2.4, 9 / 2.4), sharex=True)
ax2 = ax[1].twinx()

ax[0].plot(t_sim, Y_PID, c=c1, label="PID")
ax[0].plot(t_sim, Y_LIN, c=c2, label="LIN")
ax[0].plot(t_sim, Y_GRU, c=c3, label="GRU")
ax[0].plot(t_sim, Y_OPL, c="grey", label="OPL")
ax[0].step(t_sim, np.ones(t_sim.shape[0]) * 7.0, c="k", ls="--", label="$y_{sp}$")
ax[0].set_ylim([5.5, 7.5])
ax[0].set_ylabel("pH")
ax[0].legend(ncol=5)

ax[1].step(t_sim, U_PID[:, 0], c=c1)
ax[1].step(t_sim, U_LIN[:, 0], c=c2)
ax[1].step(t_sim, U_GRU[:, 0], c=c3)

ax[1].set_xlim([0, 201])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$u_1$ (mL/s)")

ax2.step(t_sim, U_OPL[:, 1], c="k", ls="--", alpha=0.5, label="$u_2$")
ax2.set_ylabel("$u_2$ (mL/s)")
ax2.legend()

plt.savefig("results/three_reg.pdf", bbox_inches="tight")
# plt.show()
################################################################
# PLOTING NUMERICAL RESULTS
################################################################

# SERVO
## loading data

ysp = np.loadtxt("results/ysp.csv", delimiter=",")

t_sim = np.loadtxt("results/PID/PID_servo.csv", delimiter=",")[:, 0]
dt = t_sim[1] - t_sim[0]

PID_servo = np.loadtxt("results/PID/PID_servo.csv", delimiter=",")[:, 1:]
LIN_servo = np.loadtxt("results/LIN/LIN_servo.csv", delimiter=",")[:, 1:]
GRU_servo = np.loadtxt("results/GRU/GRU_servo.csv", delimiter=",")[:, 1:]

PID_reg = np.loadtxt("results/PID/PID_reg.csv", delimiter=",")[:, 1:]
LIN_reg = np.loadtxt("results/LIN/LIN_reg.csv", delimiter=",")[:, 1:]
GRU_reg = np.loadtxt("results/GRU/GRU_reg.csv", delimiter=",")[:, 1:]

servo_df = {
    "PID": get_metrics(PID_servo),
    "LIN": get_metrics(LIN_servo),
    "GRU": get_metrics(GRU_servo),
}

reg_df = {
    "PID": get_metrics(PID_reg),
    "LIN": get_metrics(LIN_reg),
    "GRU": get_metrics(GRU_reg),
}
index = ["IAE", "ITAE", "EC"]
servo_df = pd.DataFrame(servo_df, index=index).T
reg_df = pd.DataFrame(reg_df, index=index).T

servo_df_rel = servo_df.apply(
    lambda row: (row - servo_df.loc["PID"]) / servo_df.loc["PID"] * 100, axis=1
)
reg_df_rel = reg_df.apply(
    lambda row: (row - reg_df.loc["PID"]) / reg_df.loc["PID"] * 100, axis=1
)

servo_df_rel.columns = [string + " (%)" for string in index]
reg_df_rel.columns = [string + " (%)" for string in index]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
colors = ["white", "grey", "black"]

ax[0].set_title("Servo")
ax[0].axhline(0, c="k", lw=2)
servo_df_rel.loc[["LIN", "GRU"]].plot(
    kind="bar", ax=ax[0], edgecolor="black", color=colors
)
ax[0].grid(axis="y", ls="--")
ax[0].legend_.remove()
ax[0].set_xticklabels(["LIN", "GRU"], rotation=0)

ax[1].set_title("Regulatory")
ax[1].axhline(0, c="k", lw=2)

reg_df_rel.loc[["LIN", "GRU"]].plot(
    kind="bar", ax=ax[1], edgecolor="black", color=colors
)
ax[1].grid(axis="y", ls="--")
ax[1].legend_.remove()
ax[1].set_xticklabels(["LIN", "GRU"], rotation=0)
ax[1].set_yticks(np.arange(-100, 100 + 10, 10))

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("results/metrics.pdf", bbox_inches="tight")

# OFFLINE PLOTING
offline_GRU = np.loadtxt("results/GRU/offline.csv", delimiter=",")
offline_LIN = np.loadtxt("results/LIN/offline.csv", delimiter=",")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(offline_LIN[:, 0], c="k", label="Data", ls="--")
ax.plot(offline_GRU[:, 1], c=c3, label="GRU")
ax.plot(offline_LIN[:, 1], c=c2, label="LIN")
ax.legend()

plt.savefig("results/offline.pdf", bbox_inches="tight")
plt.show()
