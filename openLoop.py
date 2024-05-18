from PID import *

c1, c2, c3 = "#0152F0", "#F51800", "#25FA00"

##################################################################

# GENERATING REGULATORY LOOP

##################################################################
dt = 0.5
k=8
noise_scale = 0.05
t_sim = np.arange(0, 200+dt, dt)

X_OL = np.zeros((t_sim.shape[0], 2))
U_OL = np.zeros((t_sim.shape[0], 2))
Y_OL = np.zeros(t_sim.shape[0])

ysp = np.ones(t_sim.shape[0])
ysp[:] = 7.0

X_OL[0,:] = [Wa, Wb]
Y_OL[0] = y_f(X_OL[0], x0=7.0)
U_OL[:,:] = [u1ss, u2ss]

#Disturbances
U_OL[10:, 1] = 0.0
U_OL[144:, 1] = 0.1
U_OL[288:, 1] = 0.2


print("SIMULATING ...")
for n in tqdm(range(t_sim.shape[0]-1)):
    X_OL[n+1,:] = x_next(X_OL[n], U_OL[n], dt)
    Y_OL[n+1] = y_f(X_OL[n+1], x0=Y_OL[n]) + np.random.normal(scale=noise_scale)

plt.step(t_sim, Y_OL)
plt.show()
dataset_OL = np.column_stack((t_sim, U_OL, Y_OL))
np.savetxt("data/openLoop.csv", dataset_OL, delimiter=",")