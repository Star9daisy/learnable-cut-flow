from typing import cast

import numpy as np
from sklearn.model_selection import train_test_split


def load_data(
    n_samples: int = 10000, seed: int | None = None
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    n_samples_per_class = n_samples // 2
    rs = np.random.RandomState(seed)

    # [1/4] Signal is in the left side of an observable distribution
    # signal | background
    sig_x1 = rs.normal(-2, 2, (n_samples_per_class, 1))
    bkg_x1 = rs.normal(2, 2, (n_samples_per_class, 1))

    # [2/4] Signal is in the right side of an observable distribution
    # background | signal
    sig_x2 = rs.normal(2, 2, (n_samples_per_class, 1))
    bkg_x2 = rs.normal(-2, 2, (n_samples_per_class, 1))

    # [3/4] Signal is in the middle of an observable distribution
    # background | signal | background
    sig_x3 = rs.normal(0, 2, (n_samples_per_class, 1))
    bkg_x3_left = rs.normal(-5, 2, (n_samples_per_class // 2, 1))
    bkg_x3_right = rs.normal(5, 2, (n_samples_per_class // 2, 1))
    bkg_x3 = np.concatenate([bkg_x3_left, bkg_x3_right])

    # [4/4] Signal is at both edges of an observable distribution
    # signal | background | signal
    sig_x4_left = rs.normal(-5, 2, (n_samples_per_class // 2, 1))
    sig_x4_right = rs.normal(5, 2, (n_samples_per_class // 2, 1))
    sig_x4 = np.concatenate([sig_x4_left, sig_x4_right])
    bkg_x4 = rs.normal(0, 2, (n_samples_per_class, 1))

    # [5/5] Weaker feature (less separable)
    sig_x5 = rs.normal(-1, 2, (n_samples_per_class, 1))
    bkg_x5 = rs.normal(1, 2, (n_samples_per_class, 1))

    # [6/6] Redundant feature 1
    sig_x7 = rs.normal(-5, 2, (n_samples_per_class, 1))
    bkg_x7 = rs.normal(-5, 2, (n_samples_per_class, 1))

    # [7/7] Highly correlated feature 1
    sig_x9 = sig_x1 * 0.9 + rs.normal(0, 1, (n_samples_per_class, 1))
    bkg_x9 = bkg_x1 * 0.9 + rs.normal(0, 1, (n_samples_per_class, 1))

    # Combine all the data
    sig_x = np.concatenate(
        [sig_x1, sig_x2, sig_x3, sig_x4, sig_x5, sig_x7, sig_x9], axis=1
    )
    bkg_x = np.concatenate(
        [bkg_x1, bkg_x2, bkg_x3, bkg_x4, bkg_x5, bkg_x7, bkg_x9], axis=1
    )
    x = np.concatenate([sig_x, bkg_x])

    sig_y = np.ones((n_samples_per_class, 1))
    bkg_y = np.zeros((n_samples_per_class, 1))
    y = np.concatenate([sig_y, bkg_y])

    x = x.astype(np.float32)
    y = y.astype(np.int32)

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=seed
    )

    # Type casting
    x_train = cast(np.ndarray, x_train)
    x_test = cast(np.ndarray, x_test)
    y_train = cast(np.ndarray, y_train)
    y_test = cast(np.ndarray, y_test)

    return (x_train, y_train), (x_test, y_test)
