from pathlib import Path
from pickle import dump as save_sklearn_model
from pickle import load as load_sklearn_model
from time import time
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import ops
from keras.models import Model
from keras.models import load_model as load_keras_model
from keras.models import save_model as save_keras_model
from rich.console import Console
from sklearn.base import BaseEstimator

from .layers import LearnableCut


class Timer:
    def __init__(self):
        self.record = 0

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time()
        self.record = self.end_time - self.start_time


def to_numpy(x) -> np.ndarray:
    return ops.convert_to_numpy(x)  # type: ignore


def save_model(model, to_file: str | Path) -> None:
    if isinstance(model, Model):
        save_keras_model(model, to_file)
    elif isinstance(model, BaseEstimator):
        with open(to_file, "wb") as f:
            save_sklearn_model(model, f, protocol=5)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def load_model(path: str | Path) -> Model | BaseEstimator:
    path = Path(path)
    if path.suffix == ".keras":
        return cast(Model, load_keras_model(path))
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            return load_sklearn_model(f)
    else:
        raise ValueError(f"Unsupported model type: {path.suffix}")


def print(*obj: object, **kwargs) -> None:
    console = Console(force_jupyter=False)
    console.print(*obj, **kwargs)


def tex_to_str(tex: str) -> str:
    return (
        tex.replace("$", "")
        .replace(" ", "_")
        .replace("^", "-")
        .replace("\\", "")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
        .replace("=", "_")
    )


def plot_distribution(
    data: list[np.ndarray],
    bins: np.ndarray | int,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    weights: list[np.ndarray] | None = None,
    xlabel: str = "x",
    ylabel: str = "Counts",
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "step",
    linewidth: float = 1.5,
    label_fontsize: int = 18,
    tick_fontsize: int = 14,
    legend_fontsize: int = 12,
    to_file: str | Path | None = None,
) -> None:
    plt.figure(dpi=300)
    for i, x in enumerate(data):
        plt.hist(
            x,
            bins=bins,
            color=colors[i] if colors is not None else None,
            label=labels[i] if labels is not None else None,
            weights=weights[i] if weights is not None else None,
            histtype=histtype,
            linewidth=linewidth,
        )
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    if isinstance(bins, np.ndarray):
        plt.xlim(bins[0], bins[-1])
    plt.tick_params(labelsize=tick_fontsize)
    plt.legend(loc="upper right", fontsize=legend_fontsize)
    plt.tight_layout()
    if to_file is not None:
        plt.savefig(to_file)
        plt.close()
    else:
        plt.show()


def plot_correlation(
    data: np.ndarray,
    features: list[str],
    vmin: float = -1,
    vmax: float = 1,
    cmap: str = "coolwarm",
    annot: bool = True,
    fmt: str = ".2f",
    square: bool = True,
    tick_fontsize: int = 14,
    annotation_fontsize: int = 12,
    to_file: str | Path | None = None,
) -> None:
    df = pd.DataFrame(data, columns=features)
    mask = np.triu(np.ones_like(df.corr(), dtype=bool), k=1)

    plt.figure(dpi=300)
    sns.heatmap(
        df.corr(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        square=square,
        mask=mask,
        annot_kws={"fontsize": annotation_fontsize},
    )
    plt.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    if to_file is not None:
        plt.savefig(to_file)
        plt.close()
    else:
        plt.show()


def plot_learned_cut(
    case: int,
    index: int,
    boundaries: list[float],
    expression: str,
    data: list[np.ndarray],
    bins: np.ndarray | int,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    weights: list[np.ndarray] | None = None,
    xlabel: str = "x",
    ylabel: str = "Counts",
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "step",
    linewidth: float = 1.5,
    label_fontsize: int = 18,
    tick_fontsize: int = 14,
    legend_fontsize: int = 12,
    to_file: str | Path | None = None,
) -> None:
    plt.figure(dpi=300)
    for i, x in enumerate(data):
        plt.hist(
            x,
            bins=bins,
            color=colors[i] if colors is not None else None,
            label=labels[i] if labels is not None else None,
            weights=weights[i] if weights is not None else None,
            histtype=histtype,
            linewidth=linewidth,
        )
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    if isinstance(bins, np.ndarray):
        plt.xlim(bins[0], bins[-1])
    plt.tick_params(labelsize=tick_fontsize)

    x_min, x_max = plt.xlim()
    if case == LearnableCut.LEFT:
        boundary = boundaries[index]
        plt.axvline(boundary, label=expression, color="red", linestyle="--")
        plt.axvspan(x_min, boundary, color="red", alpha=0.1)
    elif case == LearnableCut.RIGHT:
        boundary = boundaries[index]
        plt.axvline(boundary, label=expression, color="red", linestyle="--")
        plt.axvspan(boundary, x_max, color="red", alpha=0.1)
    elif case == LearnableCut.MIDDLE:
        lower, upper = boundaries
        plt.axvline(lower, color="red", linestyle="--")
        plt.axvline(upper, label=expression, color="red", linestyle="--")
        plt.axvspan(lower, upper, color="red", alpha=0.1)
    else:
        lower, upper = boundaries
        plt.axvline(lower, color="red", linestyle="--")
        plt.axvline(upper, label=expression, color="red", linestyle="--")
        plt.axvspan(x_min, lower, color="red", alpha=0.1)
        plt.axvspan(upper, x_max, color="red", alpha=0.1)

    plt.legend(loc="upper right", fontsize=legend_fontsize)
    plt.tight_layout()
    if to_file is not None:
        plt.savefig(to_file)
        plt.close()
    else:
        plt.show()


def plot_learned_importance(
    scores: np.ndarray,
    baseline: float,
    features: list[str],
    errors: np.ndarray | None = None,
    to_file: str | None = None,
) -> None:
    plt.figure(dpi=300)
    bars = plt.bar(features, scores)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.4f}",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=14,
        )

    plt.axhline(
        baseline,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline: {baseline:.4f}",
    )

    if errors is not None:
        for i, (score, error) in enumerate(zip(scores, errors)):
            plt.fill_between(
                [i - 0.4, i + 0.4],
                [score - error, score - error],
                [score + error, score + error],
                color="gray",
                alpha=0.5,
                label="Error" if i == 0 else "",
            )

    plt.ylabel("Importance", fontsize=18)
    plt.tick_params(axis="both", labelsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    if to_file:
        plt.savefig(to_file, dpi=300)
        plt.close()
    else:
        plt.show()
