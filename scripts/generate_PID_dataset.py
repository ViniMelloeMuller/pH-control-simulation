import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.controllers.PID import PIDController
from src.systems.system import pHSystem

DATASET_SIZE = int(1e4)  # Number of samples in the dataset

np.random.seed(42)

default_params = {
    "Kc": 1.0,
    "tau_I": 1.0,
    "tau_D": 1.0,
}


def main():
    system = pHSystem(dt=0.5)

    try:
        with open("models/pid_params.json", "r") as f:
            pid_params = json.load(f)
    except FileNotFoundError:
        pid_params = default_params

    controller = PIDController(dt=system.dt, **pid_params)

    setpoint_arr = np.ones(DATASET_SIZE) * system._yss
    for i in range(0, DATASET_SIZE, 200):
        setpoint_arr[i : i + 200] = np.random.uniform(5.0, 9.0)

    for step in tqdm(range(DATASET_SIZE - 1)):
        pid_state = np.array([system.y, setpoint_arr[step]])
        u_pid = controller.policy(pid_state)

        action = np.array([u_pid, system._u2ss])
        system.step(action)

    ax = system.plot(setpoint_list=setpoint_arr)
    plt.savefig("results/PID_dataset.png", dpi=800, bbox_inches="tight")
    plt.show()

    system.get_df.to_csv(f"data/PID_dataset_{DATASET_SIZE}.csv")


if __name__ == "__main__":
    main()
