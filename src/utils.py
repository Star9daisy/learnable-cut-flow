# pyright: reportArgumentType=false
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .models import LearnableCutFlowParallelModel, LearnableCutFlowSequentialModel


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

        config = {"bins": bins_i, "histtype": "step"}
        ax.hist(sig, label="SIG.", weights=sig_weights, color="r", **config)
        ax.hist(bkg, label="BKG.", weights=bkg_weights, color="b", **config)

        ax.legend(loc="upper right")
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel("Percent")
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


def show_learned_cuts(
    model: LearnableCutFlowParallelModel | LearnableCutFlowSequentialModel,
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
    feature_names = feature_names or [cut.feature_name for cut in model.learnable_cuts]
    n_rows = (n_features + n_columns - 1) // n_columns
    df = pd.DataFrame(
        x, columns=pd.Series([cut.feature_name for cut in model.learnable_cuts])
    )
    df["y"] = y.squeeze()

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        x_i = df.iloc[:, i]
        y_i = df["y"]
        bins_i = bins[i]
        sig, bkg = x_i[y_i == 1], x_i[y_i == 0]
        sig_weights = 100 * np.ones_like(sig) / len(x_i)
        bkg_weights = 100 * np.ones_like(bkg) / len(x_i)

        HIST_CONFIG = {"bins": bins_i, "histtype": "step"}
        ax.hist(sig, label="SIG.", weights=sig_weights, color="red", **HIST_CONFIG)
        ax.hist(bkg, label="BKG.", weights=bkg_weights, color="blue", **HIST_CONFIG)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel("Percent")
        ax.set_xlim(bins_i[0], bins_i[-1]) if not isinstance(bins_i, int) else None

        x_min, x_max = ax.get_xlim()
        cut_report = model.learned_cuts_report[i]
        cut = cut_report["cut"].replace(
            model.learnable_cuts[i].feature_name, feature_names[i]
        )
        CUT_LINE_CONFIG = {"color": "red", "linestyle": "--"}
        CUT_AREA_CONFIG = {"color": "red", "alpha": 0.1}
        if cut_report["case"] == "left":
            boundary = cut_report["boundaries"][cut_report["index"]]
            ax.axvline(boundary, label=cut, **CUT_LINE_CONFIG)
            ax.axvspan(x_min, boundary, **CUT_AREA_CONFIG)
        elif cut_report["case"] == "right":
            boundary = cut_report["boundaries"][cut_report["index"]]
            ax.axvline(boundary, label=cut, **CUT_LINE_CONFIG)
            ax.axvspan(boundary, x_max, **CUT_AREA_CONFIG)
        elif cut_report["case"] == "middle":
            lower, upper = cut_report["boundaries"]
            ax.axvline(lower, **CUT_LINE_CONFIG)
            ax.axvline(upper, label=cut, **CUT_LINE_CONFIG)
            ax.axvspan(lower, upper, **CUT_AREA_CONFIG)
        else:
            lower, upper = cut_report["boundaries"]
            ax.axvline(lower, **CUT_LINE_CONFIG)
            ax.axvline(upper, label=cut, **CUT_LINE_CONFIG)
            ax.axvspan(x_min, lower, **CUT_AREA_CONFIG)
            ax.axvspan(upper, x_max, **CUT_AREA_CONFIG)

        ax.legend(loc="upper right")
        ax.set_xlim(x_min, x_max)

        if isinstance(model, LearnableCutFlowSequentialModel):
            df = df.query(cut_report["cut"])

    for i in range(n_features, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    if to_file:
        fig.savefig(to_file, dpi=300)
        plt.close(fig)
    else:
        fig.show()


def show_learned_cut(
    cut_index: int,
    model: LearnableCutFlowParallelModel | LearnableCutFlowSequentialModel,
    x: np.ndarray,
    y: np.ndarray,
    bins: int | np.ndarray = 100,
    feature_name: str | None = None,
    to_file: str | None = None,
) -> None:
    y = y.squeeze()
    feature_name = feature_name or model.learnable_cuts[cut_index].feature_name

    fig, ax = plt.subplots()

    x_i = x[:, cut_index]
    sig, bkg = x_i[y == 1], x_i[y == 0]
    sig_weights = 100 * np.ones_like(sig) / len(x_i)
    bkg_weights = 100 * np.ones_like(bkg) / len(x_i)

    HIST_CONFIG = {"bins": bins, "histtype": "step", "linewidth": 1.5}
    ax.hist(sig, label="SIG.", weights=sig_weights, color="r", **HIST_CONFIG)
    ax.hist(bkg, label="BKG.", weights=bkg_weights, color="b", **HIST_CONFIG)
    ax.set_xlabel(feature_name, fontsize=18)
    ax.set_ylabel("Percent", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlim(bins[0], bins[-1]) if not isinstance(bins, int) else None

    x_min, x_max = ax.get_xlim()
    cut_report = model.learned_cuts_report[cut_index]
    cut = cut_report["cut"].replace(
        model.learnable_cuts[cut_index].feature_name, feature_name
    )
    CUT_LINE_CONFIG = {"color": "red", "linestyle": "--"}
    CUT_AREA_CONFIG = {"color": "red", "alpha": 0.1}
    if cut_report["case"] == "left":
        boundary = cut_report["boundaries"][cut_report["index"]]
        ax.axvline(boundary, label=cut, **CUT_LINE_CONFIG)
        ax.axvspan(x_min, boundary, **CUT_AREA_CONFIG)
    elif cut_report["case"] == "right":
        boundary = cut_report["boundaries"][cut_report["index"]]
        ax.axvline(boundary, label=cut, **CUT_LINE_CONFIG)
        ax.axvspan(boundary, x_max, **CUT_AREA_CONFIG)
    elif cut_report["case"] == "middle":
        lower, upper = cut_report["boundaries"]
        ax.axvline(lower, **CUT_LINE_CONFIG)
        ax.axvline(upper, label=cut, **CUT_LINE_CONFIG)
        ax.axvspan(lower, upper, **CUT_AREA_CONFIG)
    else:
        lower, upper = cut_report["boundaries"]
        ax.axvline(lower, **CUT_LINE_CONFIG)
        ax.axvline(upper, label=cut, **CUT_LINE_CONFIG)
        ax.axvspan(x_min, lower, **CUT_AREA_CONFIG)
        ax.axvspan(upper, x_max, **CUT_AREA_CONFIG)

    ax.legend(loc="upper right", fontsize=14)
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    if to_file:
        fig.savefig(to_file, dpi=300)
        plt.close(fig)
    else:
        fig.show()


def show_learned_importance(
    model: LearnableCutFlowParallelModel | LearnableCutFlowSequentialModel,
    feature_names: list[str] | None = None,
    to_file: str | None = None,
) -> None:
    learned_importance = model.learned_importance
    feature_names = feature_names or [cut.feature_name for cut in model.learnable_cuts]

    fig, ax = plt.subplots()
    bars = ax.bar(feature_names, learned_importance)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.4f}",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=14,
        )

    ax.axhline(
        model.importance_baseline,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline: {model.importance_baseline:.4f}",
    )
    ax.legend(fontsize=14)
    ax.set_ylabel("Importance", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)

    fig.tight_layout()
    if to_file:
        fig.savefig(to_file, dpi=300)
        plt.close(fig)
    else:
        fig.show()
