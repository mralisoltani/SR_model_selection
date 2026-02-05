import numpy as np
import pandas as pd  # for CSV
from pathlib import Path


def generate_datasets(n, noiselvl=0.1):
    np.random.seed(42)  # for reproducibility
    generate_friedman(n, noiselvl)
    generate_kotanchek(n, noiselvl)
    generate_salustowicz(n, noiselvl)
    generate_salustowicz2d(n, noiselvl)
    generate_ratpol3d(n, noiselvl)
    generate_ratpol2d(n, noiselvl)
    generate_ripple(n, noiselvl)


# f1
def generate_friedman(n, noiselvl):
    X = np.random.rand(n, 10)
    y_clean = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * np.square(X[:, 2] - 0.5)
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    y = y_clean + np.random.randn(n) * np.std(y_clean) * noiselvl
    sigma = np.std(y_clean) * noiselvl
    sigma_col = np.full((n, 1), sigma)

    df = pd.DataFrame(
        np.concatenate((X, y.reshape(n, -1), y_clean.reshape(n, -1), sigma_col), axis=1)
    )
    df.rename(
        columns={
            0: "x1",
            1: "x2",
            2: "x3",
            3: "x4",
            4: "x5",
            5: "x6",
            6: "x7",
            7: "x8",
            8: "x9",
            9: "x10",
            10: "y",
            11: "y_clean",
            12: "sigma",
        },
        inplace=True,
    )
    df.to_csv(f"data/friedman_{n}_noise-{noiselvl}.csv", header=True, index=False)


# f2
def generate_kotanchek(n, noiselvl):
    X = np.random.rand(n, 2) * 4
    y_clean = np.exp(-np.square(X[:, 0] - 1)) / (1.2 + np.square(X[:, 1] - 2.5))
    y = y_clean + np.random.randn(n) * np.std(y_clean) * noiselvl
    sigma = np.std(y_clean) * noiselvl
    sigma_col = np.full((n, 1), sigma)

    df = pd.DataFrame(
        np.concatenate((X, y.reshape(n, -1), y_clean.reshape(n, -1), sigma_col), axis=1)
    )
    df.rename(
        columns={0: "x1", 1: "x2", 2: "y", 3: "y_clean", 4: "sigma"}, inplace=True
    )
    df.to_csv(f"data/kotanchek_{n}_noise-{noiselvl}.csv", header=True, index=False)


# f3
def generate_salustowicz(n, noiselvl):
    x = np.random.rand(n) * 10
    y_clean = (
        np.exp(-x)
        * np.power(x, 3)
        * np.cos(x)
        * np.sin(x)
        * (np.cos(x) * np.square(np.sin(x)) - 1)
    )
    y = y_clean + np.random.randn(n) * np.std(y_clean) * noiselvl
    sigma = np.std(y_clean) * noiselvl
    sigma_col = np.full((n, 1), sigma)

    df = pd.DataFrame(
        np.concatenate(
            (x.reshape(n, -1), y.reshape(n, -1), y_clean.reshape(n, -1), sigma_col),
            axis=1,
        )
    )
    df.rename(columns={0: "x1", 1: "y", 2: "y_clean", 3: "sigma"}, inplace=True)
    df.to_csv(f"data/salustowicz_{n}_noise-{noiselvl}.csv", header=True, index=False)


# f4
def generate_salustowicz2d(n, noiselvl):
    x1 = np.random.rand(n) * 10
    x2 = np.random.rand(n) * 10
    y_clean = (
        np.exp(-x1)
        * np.power(x1, 3)
        * np.cos(x1)
        * np.sin(x1)
        * (np.cos(x1) * np.square(np.sin(x1)) - 1)
        * (x2 - 5)
    )
    y = y_clean + np.random.randn(n) * np.std(y_clean) * noiselvl
    sigma = np.std(y_clean) * noiselvl
    sigma_col = np.full((n, 1), sigma)

    df = pd.DataFrame(
        np.concatenate(
            (
                x1.reshape(n, -1),
                x2.reshape(n, -1),
                y.reshape(n, -1),
                y_clean.reshape(n, -1),
                sigma_col,
            ),
            axis=1,
        )
    )
    df.rename(
        columns={0: "x1", 1: "x2", 2: "y", 3: "y_clean", 4: "sigma"}, inplace=True
    )
    df.to_csv(f"data/salustowicz2d_{n}_noise-{noiselvl}.csv", header=True, index=False)


# f5
def generate_ratpol3d(n, noiselvl):
    x1 = np.random.rand(n) * (2 - 0.05) + 0.05
    x2 = np.random.rand(n) + 1
    x3 = np.random.rand(n) * (2 - 0.05) + 0.05

    y_clean = 30 * (x1 - 1) * (x3 - 1) / (x1 * np.square(x2) - 10 * np.square(x2))
    y = y_clean + np.random.randn(n) * np.std(y_clean) * noiselvl
    sigma = np.std(y_clean) * noiselvl
    sigma_col = np.full((n, 1), sigma)

    df = pd.DataFrame(
        np.concatenate(
            (
                x1.reshape(n, -1),
                x2.reshape(n, -1),
                x3.reshape(n, -1),
                y.reshape(n, -1),
                y_clean.reshape(n, -1),
                sigma_col,
            ),
            axis=1,
        )
    )
    df.rename(
        columns={0: "x1", 1: "x2", 2: "x3", 3: "y", 4: "y_clean", 5: "sigma"},
        inplace=True,
    )
    df.to_csv(f"data/ratpol3d_{n}_noise-{noiselvl}.csv", header=True, index=False)


# f6
def generate_ratpol2d(n, noiselvl):
    x1 = np.random.rand(n) * 6
    x2 = np.random.rand(n) * 6

    y_clean = (np.power(x1 - 3, 4) + np.power(x2 - 3, 3) - (x2 - 3)) / (
        np.power(x2 - 2, 4) + 10
    )
    y = y_clean + np.random.randn(n) * np.std(y_clean) * noiselvl
    sigma = np.std(y_clean) * noiselvl
    sigma_col = np.full((n, 1), sigma)

    df = pd.DataFrame(
        np.concatenate(
            (
                x1.reshape(n, -1),
                x2.reshape(n, -1),
                y.reshape(n, -1),
                y_clean.reshape(n, -1),
                sigma_col,
            ),
            axis=1,
        )
    )
    df.rename(
        columns={0: "x1", 1: "x2", 2: "y", 3: "y_clean", 4: "sigma"}, inplace=True
    )
    df.to_csv(f"data/ratpol2d_{n}_noise-{noiselvl}.csv", header=True, index=False)


# f7
def generate_ripple(n, noiselvl):
    x1 = np.random.rand(n) * 6
    x2 = np.random.rand(n) * 6

    y_clean = (x1 - 3) * (x2 - 3) + 2 * np.sin((x1 - 4) * (x2 - 4))
    y = y_clean + np.random.randn(n) * np.std(y_clean) * noiselvl
    sigma = np.std(y_clean) * noiselvl
    sigma_col = np.full((n, 1), sigma)

    df = pd.DataFrame(
        np.concatenate(
            (
                x1.reshape(n, -1),
                x2.reshape(n, -1),
                y.reshape(n, -1),
                y_clean.reshape(n, -1),
                sigma_col,
            ),
            axis=1,
        )
    )
    df.rename(
        columns={0: "x1", 1: "x2", 2: "y", 3: "y_clean", 4: "sigma"}, inplace=True
    )
    df.to_csv(f"data/ripple_{n}_noise-{noiselvl}.csv", header=True, index=False)


n = 10100  # number of points
n_level = 0.1  # noise level
generate_datasets(n, n_level)


# Seperate datapoints into train/val and test sets

datapoints = {
    "f1": f"data/friedman_{n}_noise-{n_level}.csv",
    "f2": f"data/kotanchek_{n}_noise-{n_level}.csv",
    "f3": f"data/salustowicz_{n}_noise-{n_level}.csv",
    "f4": f"data/salustowicz2d_{n}_noise-{n_level}.csv",
    "f5": f"data/ratpol3d_{n}_noise-{n_level}.csv",
    "f6": f"data/ratpol2d_{n}_noise-{n_level}.csv",
    "f7": f"data/ripple_{n}_noise-{n_level}.csv",
}

for k, _ in datapoints.items():
    print(f"Generating datapoints for {k}: {datapoints[k]}")
    csv_path = Path(datapoints[k])  # change to your actual file
    df = pd.read_csv(csv_path)
    x_cols = [c for c in df.columns if c.startswith("x")]

    N_total = len(df)
    n_train = 80  # int(0.6 * N_total)  # 60% train
    n_val = 20  # int(0.2 * N_total)    # 20% validation
    N_train_val = n_train + n_val  # train and val together

    # Split into train/val and test
    train_val_df = df.iloc[:N_train_val].reset_index(drop=True)
    test_df = df.iloc[N_train_val:].reset_index(drop=True)

    # Save as separate files
    train_val_df.to_csv(
        csv_path.with_name(csv_path.stem + "_train_val.csv"), index=False
    )
    test_df.to_csv(csv_path.with_name(csv_path.stem + "_test.csv"), index=False)

    print(f"N (train + val): {N_train_val}")
    print(f"N (test): {N_total - N_train_val}")
    print("Train/Val file:", csv_path.with_name(csv_path.stem + "_train_val.csv"))
    print("Test file:", csv_path.with_name(csv_path.stem + "_test.csv"))
    print("Done")
