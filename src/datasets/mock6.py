from typing import cast

import numpy as np
from sklearn.model_selection import train_test_split

from .mock5 import load_data as load_mock5


def load_data(
    n_samples: int = 10000, seed: int | None = None
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray]:
    rs = np.random.RandomState(seed)

    # Randomly permute all features of mock5
    (x_train, y_train), (x_test, y_test) = load_mock5(n_samples, seed)
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    order = rs.permutation(x.shape[1])
    x = x[:, order]

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed
    )

    # Type casting
    x_train = cast(np.ndarray, x_train)
    x_test = cast(np.ndarray, x_test)
    y_train = cast(np.ndarray, y_train)
    y_test = cast(np.ndarray, y_test)
    order = cast(np.ndarray, order)

    return (x_train, y_train), (x_test, y_test), order
