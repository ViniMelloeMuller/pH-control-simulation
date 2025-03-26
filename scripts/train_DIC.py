import argparse
import os
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.controllers.DIC import create_x_y_data

DATA_PATH = "data/PID_dataset_100000.csv"
STUDY_PATH = "models/DIC/DIC_study.pkl"

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.8, patience=5, min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    ),
]


def create_model(params: dict) -> Sequential:
    """Create a simple feedforward neural network model."""
    model = Sequential()
    model.add(keras.layers.Input((6, 1)))

    model.add(
        keras.layers.Dense(
            params["hidden_units"],
            activation="relu",
            kernel_initializer=keras.initializers.glorot_normal(),
            kernel_regularizer=keras.regularizers.l1_l2(
                l1=params["l1"], l2=params["l2"]
            ),
        )
    )

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(params["dropout"]))

    model.add(
        keras.layers.Dense(
            params["hidden_units"] // 2,
            activation="relu",
            kernel_initializer=keras.initializers.glorot_normal(),
            kernel_regularizer=keras.regularizers.l1_l2(
                l1=params["l1"], l2=params["l2"]
            ),
        )
    )

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(params["dropout"]))

    model.add(keras.layers.Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def get_avg_inference_time(model: Sequential, X: np.ndarray) -> float:
    """Gets the average inference time of a single sample prediction"""
    times = []
    for _ in range(100):
        X_sample = X[np.random.randint(0, len(X))].reshape(1, -1)
        start = time.perf_counter()
        model.predict(X_sample, verbose=0)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times)  # seconds


def save_study(study: optuna.study.Study):
    joblib.dump(study, STUDY_PATH)


def load_study() -> optuna.study.Study:
    if os.path.exists(STUDY_PATH):
        return joblib.load(STUDY_PATH)
    else:
        return optuna.create_study(direction="minimize")


def main(N_TRIALS: int | None, TIMEOUT: int | None):
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH} - Run generate_PID_dataset.py first"
        )

    X, y = create_x_y_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = MinMaxScaler()

    scaler.fit(X_train)

    os.makedirs("models/DIC", exist_ok=True)
    joblib.dump(scaler, "models/DIC/DIC_scaler.pkl")

    XN_train = scaler.transform(X_train)
    XN_test = scaler.transform(X_test)

    def optuna_objective(trial):
        params = {
            "hidden_units": trial.suggest_int("hidden_units", 10, 300, step=20),
            "l1": trial.suggest_float("l1", 1e-6, 1e-3, log=True),
            "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        }

        model = create_model(params)
        model.fit(
            XN_train,
            y_train,
            epochs=200,
            batch_size=128,
            verbose=1,
            callbacks=callbacks,
            validation_data=(XN_test, y_test),
        )

        avg_inference_time = get_avg_inference_time(model, X_train)

        test_score = model.evaluate(XN_test, y_test)
        trial.set_user_attr("test_score", test_score)
        trial.set_user_attr("avg_inference_time", avg_inference_time)

        objective_value = test_score
        return objective_value

    study = load_study()
    study.optimize(optuna_objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    best_params = study.best_params
    best_scores = study.best_trial.user_attrs

    print(f"Best parameters: {best_params}")
    print(f"Best test score: {best_scores}")

    best_model = create_model(best_params)
    best_model.fit(
        XN_train,
        y,
        epochs=900,
        batch_size=128,
        verbose=1,
        validation_split=0.2,
        callbacks=callbacks,
    )
    best_model.save("models/DIC/DIC_model.keras")
    print("Best Model saved")

    save_study(study)

    print("STUDY RESULTS:")
    print(study.trials_dataframe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DIC controller")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_trials", type=int, help="Number of trials to run")
    group.add_argument(
        "--timeout", type=int, help="Stop study after the given number of second(s)."
    )

    args = parser.parse_args()
    N_TRIALS = args.n_trials
    TIMEOUT = args.timeout

    main(N_TRIALS, TIMEOUT)
