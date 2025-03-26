################################################################################
# IMPLEMENTS THE DIRECT INVERSE NEURAL CONTROLLER                              #
################################################################################

from . import Controller
import numpy as np
import time
import pandas as pd
import os

os.environ["KERAS_BACKEND"] = "torch"
from keras import Sequential


def create_x_y_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Recives historical data and returns the X and y data for training the DIC controller.

    X: [u1(t-1), u2(t-1), y(t-1), y(t), u2(t), y(t+1)]
    Y: [u1(t)]
    """
    df["u1_t-1"] = df["u1"].shift(1)
    df["u2_t-1"] = df["u2"].shift(1)
    df["y_t-1"] = df["y"].shift(1)
    df["y_t+1"] = df["y"].shift(-1)

    df_valid = df.dropna().reset_index(drop=True)

    # Features
    X = np.column_stack(
        (
            df_valid["u1_t-1"].to_numpy(),
            df_valid["u2_t-1"].to_numpy(),
            df_valid["y_t-1"].to_numpy(),
            df_valid["y"].to_numpy(),
            df_valid["u2"].to_numpy(),
            df_valid["y_t+1"].to_numpy(),
        )
    )

    Y = df_valid["u1"].to_numpy().reshape(-1, 1)
    return X, Y


class DICController(Controller):
    def __init__(self, dt: float, model: Sequential):
        """
        Args:
            dt: Discrete time step
            model: Neural network model
        """
        super().__init__(dt)
        self.model = model

    def policy(self, state: np.ndarray) -> np.ndarray:
        """
        Args:
            state: [y(t), u2(t)]
        Returns:
            u1(t)
        """
        X = np.array(state).reshape(1, -1)

        start_time = time.perf_counter()
        u1 = self.model.predict(X, verbose=0)
        u1 = np.clip(u1, 0, 20)
        end_time = time.perf_counter()

        self.action_list.append(u1)
        self.computational_time.append(end_time - start_time)
        return u1

    def reset(self):
        self.error_list = []
        self.action_list = []
        self.computational_time = []
