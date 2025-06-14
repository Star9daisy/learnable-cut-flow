from pathlib import Path
from typing import cast

import h5py
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(
    n_samples: int = 10000, seed: int | None = None
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    rs = np.random.RandomState(seed)

    # Load data
    path = Path(__file__).parent / "data/1603.09349/records"
    with h5py.File(path / "test_no_pile_5000000.h5", "r") as f:
        x = np.array(f["features"])
        y = np.array(f["targets"])

    # Take random samples
    indices = rs.randint(0, len(x), size=n_samples)
    x, y = x[indices], y[indices]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=seed
    )

    # Type casting
    x_train = cast(np.ndarray, x_train)
    x_test = cast(np.ndarray, x_test)
    y_train = cast(np.ndarray, y_train)
    y_test = cast(np.ndarray, y_test)

    return (x_train, y_train), (x_test, y_test)
