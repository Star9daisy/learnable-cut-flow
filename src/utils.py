import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def show_record_dataset(
    x: np.ndarray,
    y: np.ndarray,
    bins: int | np.ndarray | list[int | np.ndarray] = 100,
    feature_names: list[str] | None = None,
    n_columns: int = 3,
    to_file: str | None = None,
) -> None:
    y = y.squeeze()
    n_features = x.shape[1]
    bins = [bins] * n_features if not isinstance(bins, list) else bins
    feature_names = feature_names or [f"x{i + 1}" for i in range(n_features)]
    n_rows = (n_features + n_columns - 1) // n_columns

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        x_i = x[:, i]
        bins_i = bins[i]
        sig, bkg = x_i[y == 1], x_i[y == 0]
        sig_weights = 100 * np.ones_like(sig) / len(x_i)
        bkg_weights = 100 * np.ones_like(bkg) / len(x_i)

        config = {"bins": bins_i, "histtype": "step", "linewidth": 1.5}
        ax.hist(sig, label="SIG.", weights=sig_weights, color="r", **config)
        ax.hist(bkg, label="BKG.", weights=bkg_weights, color="b", **config)

        ax.legend(loc="upper right", fontsize=14)
        ax.set_xlabel(feature_names[i], fontsize=18)
        ax.set_ylabel("Percent", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_xlim(bins_i[0], bins_i[-1]) if not isinstance(bins_i, int) else None

    for i in range(n_features, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    if to_file:
        fig.savefig(to_file, dpi=300)
        plt.close(fig)
    else:
        fig.show()


def show_record_feature(
    x: np.ndarray,
    y: np.ndarray,
    bins: int | np.ndarray = 100,
    feature_index: int = 0,
    feature_name: str | None = None,
    to_file: str | None = None,
) -> None:
    y = y.squeeze()
    feature_name = feature_name or f"x{feature_index + 1}"

    fig, ax = plt.subplots()

    x_i = x[:, feature_index]
    sig, bkg = x_i[y == 1], x_i[y == 0]
    sig_weights = 100 * np.ones_like(sig) / len(x_i)
    bkg_weights = 100 * np.ones_like(bkg) / len(x_i)

    config = {"bins": bins, "histtype": "step", "linewidth": 1.5}
    ax.hist(sig, label="SIG.", weights=sig_weights, color="r", **config)
    ax.hist(bkg, label="BKG.", weights=bkg_weights, color="b", **config)

    ax.legend(loc="upper right", fontsize=14)
    ax.set_xlabel(feature_name, fontsize=18)
    ax.set_ylabel("Percent", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlim(bins[0], bins[-1]) if not isinstance(bins, int) else None

    fig.tight_layout()
    if to_file:
        fig.savefig(to_file, dpi=300)
        plt.close(fig)
    else:
        fig.show()


def show_record_dataset_correlation(
    x: np.ndarray,
    feature_names: list[str] | None = None,
    to_file: str | None = None,
) -> None:
    feature_names = feature_names or [f"x{i + 1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=pd.Series(feature_names))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool), k=1)

    fig, ax = plt.subplots()
    sns.heatmap(
        df.corr(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        mask=mask,
        ax=ax,
        annot_kws={"fontsize": 12},
    )

    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)

    fig.tight_layout()
    if to_file:
        fig.savefig(to_file, dpi=300)
        plt.close(fig)
    else:
        fig.show()
