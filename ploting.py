import matplotlib.pyplot as plt 
import numpy as np
from PID import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = [9/2.4, 9/2.4]

c1, c2, c3 = "#0152F0", "#F51800", "#25FA00"
#common
ysp = np.loadtxt("results/ysp.csv", delimiter=",")
t_sim = np.loadtxt("results/PID/PID_servo.csv", delimiter=",")[:,0]
dt = t_sim[1] - t_sim[0]
noise_scale = 0.05

#PID - SERVO
PID_data = np.loadtxt("results/PID/PID_servo.csv",delimiter=",")[:, 1:]
U_PID = PID_data[:, :-1]
Y_PID = PID_data[:,  -1]

IAE = np.sum(np.abs(ysp-Y_PID)*dt)
EC = np.sum(np.abs(np.diff(U_PID[:,0])))

fig, ax = plt.subplots(2,1,figsize=(18/2.4, 9/2.4),sharex=True)

ax[0].plot(t_sim, Y_PID, c=c1, label="PID")
ax[0].step(t_sim, ysp, c="k",ls="--", label="$y_{sp}$")
ax[0].set_ylabel("pH")
ax[0].legend(ncol=2)

ax[1].step(t_sim, U_PID[:,0], c=c1)
ax[1].set_xlim([0, 201])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$u_1$ (mL/s)")

string = fr"$\Delta t = ${dt} s - $\delta$ = {noise_scale}"\
						+f"\nIAE={IAE:.2f}"
ax[0].text(100, 7.8, string, fontsize=5)
ax[1].text(100, 20, f"\nEC={EC:.2f}", fontsize=5)

plt.savefig("results/PID/PID_servo_full.pdf", bbox_inches="tight")
#plt.show()


#LINEAR
LIN_data = np.loadtxt("results/LIN/LIN.csv",delimiter=",")[:, 1:]
U_LIN = LIN_data[:, :-1]
Y_LIN = LIN_data[:,  -1]

IAE = np.sum(np.abs(ysp-Y_LIN)*dt)
EC = np.sum(np.abs(np.diff(U_LIN[:,0])))

fig, ax = plt.subplots(2,1,figsize=(18/2.4, 9/2.4),sharex=True)

ax[0].plot(t_sim, Y_LIN, c=c2, label="Linear")
ax[0].step(t_sim, ysp, c="k",ls="--", label="$y_{sp}$")
ax[0].set_ylabel("pH")
ax[0].legend(ncol=2)


ax[1].step(t_sim, U_LIN[:,0], c=c2)
ax[1].set_xlim([0, 201])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$u_1$ (mL/s)")

string = fr"$\Delta t = ${dt} s - $\delta$ = {noise_scale}"\
						+f"\nIAE={IAE:.2f}"
ax[0].text(100, 7.8, string, fontsize=5)
ax[1].text(100, 20, f"\nEC={EC:.2f}", fontsize=5)

plt.savefig("results/LIN/LIN_servo_full.pdf", bbox_inches="tight")
#plt.show()


#GRU



#Three happy friends :)
fig, ax = plt.subplots(2,1,figsize=(18/2.4, 9/2.4),sharex=True)

ax[0].plot(t_sim, Y_LIN, c=c2, label="Linear")
ax[0].plot(t_sim, Y_PID, c=c1, label="PID")
ax[0].step(t_sim, ysp, c="k",ls="--", label="$y_{sp}$")
ax[0].set_ylabel("pH")
ax[0].legend(ncol=4, fontsize=6)


ax[1].step(t_sim, U_LIN[:,0], c=c2, alpha=0.5)
ax[1].step(t_sim, U_PID[:,0], c=c1, alpha=0.5)
ax[1].set_xlim([0, 201])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$u_1$ (mL/s)")

plt.savefig("results/three.pdf", bbox_inches="tight")
#plt.show()


################################################################
# REGULATORY
################################################################

PID_data = np.loadtxt("results/PID/PID_reg.csv",delimiter=",")[:, 1:]
LIN_data = np.loadtxt("results/LIN/LIN_reg.csv",delimiter=",")[:, 1:]
OPL_data = np.loadtxt("data/openLoop.csv",delimiter=",")[:, 1:]

U_OPL = OPL_data[:, :-1]
Y_OPL = OPL_data[:, -1]

U_PID = PID_data[:, :-1]
Y_PID = PID_data[:,  -1]

U_LIN = LIN_data[:, :-1]
Y_LIN = LIN_data[:,  -1]

fig, ax = plt.subplots(2,1,figsize=(18/2.4, 9/2.4),sharex=True)
ax2 = ax[1].twinx()

ax[0].plot(t_sim, Y_PID, c=c1, label="PID")
ax[0].plot(t_sim, Y_LIN, c=c2, label="LIN")
ax[0].plot(t_sim, Y_OPL, c="cyan", label="OPL")
ax[0].step(t_sim, np.ones(t_sim.shape[0])*7.0, c="k",ls="--", label="$y_{sp}$")
ax[0].set_ylim([5.5, 7.5])
ax[0].set_ylabel("pH")
ax[0].legend(ncol=4)

ax[1].step(t_sim, U_PID[:,0], c=c1)
ax[1].step(t_sim, U_LIN[:,0], c=c2)

ax[1].set_xlim([0, 201])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("$u_1$ (mL/s)")

ax2.step(t_sim, U_OPL[:,1], c="k", ls="--", alpha=0.5, label="$u_2$")
ax2.set_ylabel("$u_2$ (mL/s)")
ax2.legend()

plt.savefig("results/three_reg.pdf", bbox_inches="tight")
plt.show()

