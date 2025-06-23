from pickle import dump
from typing import cast

from keras import ops
from keras.layers import Concatenate, Dense, Identity, Input, Normalization
from keras.models import Model
from keras.saving import register_keras_serializable
from sklearn.ensemble import GradientBoostingClassifier

from src.layers import LearnableCut, LearnableImportance, Split


class GradientBoostedDecisionTree(GradientBoostingClassifier):
    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_depth=3,
        min_impurity_decrease=0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=0.0001,
        ccp_alpha=0,
        name="gradient_boosted_decision_tree",
        input_shape=None,
    ) -> None:
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )
        self.name = name
        self.input_shape = input_shape

    def compile(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        kwargs.pop("batch_size", None)
        kwargs.pop("verbose", "auto")
        kwargs.pop("steps", None)
        kwargs.pop("callbacks", None)
        return super().predict_proba(*args, **kwargs)[:, [1]]

    def fit(self, *args, **kwargs):
        kwargs.pop("x", None)
        kwargs.pop("y", None)
        kwargs.pop("batch_size", None)
        kwargs.pop("epochs", 1)
        kwargs.pop("verbose", "auto")
        kwargs.pop("callbacks", None)
        kwargs.pop("validation_split", 0.0)
        kwargs.pop("validation_data", None)
        kwargs.pop("shuffle", True)
        kwargs.pop("class_weight", None)
        kwargs.pop("sample_weight", None)
        kwargs.pop("initial_epoch", 0)
        kwargs.pop("steps_per_epoch", None)
        kwargs.pop("validation_steps", None)
        kwargs.pop("validation_batch_size", None)
        kwargs.pop("validation_freq", 1)
        return super().fit(*args, **kwargs)

    def summary(self):
        return super().get_params()

    def save(self, path: str):
        with open(path, "wb") as f:
            dump(self, f, protocol=5)


def MultiLayerPerceptron(input_shape, name="multi_layer_perceptron"):
    n_features = input_shape[1]

    inputs = Input((n_features,), name="inputs")
    x = Normalization(name="normalization")(inputs)
    x = Dense(16, activation="relu", name="dense_1")(x)
    x = Dense(32, activation="relu", name="dense_2")(x)
    x = Dense(64, activation="relu", name="dense_3")(x)
    x = Dense(32, activation="relu", name="dense_4")(x)
    x = Dense(16, activation="relu", name="dense_5")(x)
    outputs = Dense(1, activation="sigmoid", name="outputs")(x)

    return MultiLayerPerceptronModel(inputs=inputs, outputs=outputs, name=name)


@register_keras_serializable()
class MultiLayerPerceptronModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.normalizer = self.layers[1]

    def adapt(self, x):
        self.normalizer.adapt(x)


def LearnableCutFlowParallel(
    input_shape: tuple[int, int],
    centers: float | list[float] = 0.0,
    threshold: float = 0.5,
    features: list[str] | None = None,
    importance_min_percent: float = 0.05,
    name: str = "learnable_cut_flow_parallel",
):
    n_features = input_shape[1]
    centers_c = centers if isinstance(centers, list) else [centers] * n_features
    features = features or [f"x{i + 1}" for i in range(n_features)]

    # (batch_size, n_features)
    inputs = Input((n_features,), name="inputs")
    x = Normalization(name="normalization")(inputs)
    x = LearnableImportance(importance_min_percent, name="learnable_importance")(x)

    # n_features x (batch_size, 1)
    x = Split(name="split")(x)

    # At training: n_features x (batch_size, 2)
    # At inference: n_features x (batch_size, 1)
    x = [
        LearnableCut(
            center=centers_c[i],
            threshold=threshold,
            feature=features[i],
            name=f"learnable_cut_{i + 1}",
        )(x_i)
        for i, x_i in enumerate(x)
    ]
    # At training: (batch_size, n_features x 2)
    # At inference: (batch_size, n_features)
    x = Concatenate(name="concatenate")(x)
    outputs = Identity(name="outputs")(x)

    return LearnableCutFlowParallelModel(
        inputs=inputs,
        outputs=outputs,
        centers=centers_c,
        threshold=threshold,
        features=features,
        importance_min_percent=importance_min_percent,
        name=name,
    )


@register_keras_serializable()
class LearnableCutFlowParallelModel(Model):
    def __init__(
        self,
        centers: list[float],
        threshold: float,
        features: list[str],
        importance_min_percent: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.centers = centers
        self.threshold = threshold
        self.features = features
        self.importance_min_percent = importance_min_percent
        self.kwargs = kwargs

        self.normalizer = cast(Normalization, self.layers[1])
        self.importance = cast(LearnableImportance, self.layers[2])
        self.cuts = cast(list[LearnableCut], self.layers[4:-2])

    def adapt(self, x):
        self.normalizer.adapt(x)

        data_mean = ops.convert_to_numpy(self.normalizer.mean).tolist()[0]
        data_mean = cast(list[float], data_mean)
        data_variance = ops.convert_to_numpy(self.normalizer.variance).tolist()[0]
        data_variance = cast(list[float], data_variance)

        for i, cut in enumerate(self.cuts):
            cut.data_mean = data_mean[i]
            cut.data_variance = data_variance[i]
            cut.importance_score.assign(self.importance.scores[i])

    def call(self, inputs, training=None):
        y = super().call(inputs, training=training)

        if training:
            return y
        else:
            y = ops.where(
                ops.greater(self.importance.scores, self.importance.baseline), y, 1.0
            )
            y = ops.all(ops.greater(y, self.threshold), axis=1, keepdims=True)
            y = ops.cast(y, dtype="float32")
            return y

    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        del sample_weight, training

        if self._compile_loss is not None:
            loss = 0.0

            for i in range(len(self.cuts)):
                y_pred_lower = ops.take(y_pred, [2 * i], axis=1)
                y_pred_upper = ops.take(y_pred, [2 * i + 1], axis=1)

                x_i = ops.take(x, i, axis=1)
                mask_lower = ops.where(ops.less(x_i, self.centers[i]), 1.0, 0.0)
                mask_upper = ops.where(ops.greater(x_i, self.centers[i]), 1.0, 0.0)

                loss_lower = self._compile_loss(y, y_pred_lower, mask_lower)
                loss_upper = self._compile_loss(y, y_pred_upper, mask_upper)
                loss += loss_lower + loss_upper

            loss /= 2 * len(self.cuts)
            return loss

    def get_config(self):
        config = super().get_config()

        normalizer_mean = ops.convert_to_numpy(self.normalizer.mean).tolist()[0]
        normalizer_variance = ops.convert_to_numpy(self.normalizer.variance).tolist()[0]

        config.update(
            {
                "centers": self.centers,
                "threshold": self.threshold,
                "features": self.features,
                "importance_min_percent": self.importance_min_percent,
                "normalizer_mean": normalizer_mean,
                "normalizer_variance": normalizer_variance,
                **self.kwargs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = LearnableCutFlowParallel(
            input_shape=config["inputs"]["config"]["shape"],
            centers=config["centers"],
            threshold=config["threshold"],
            features=config["features"],
            importance_min_percent=config["importance_min_percent"],
            name=config["name"],
        )

        for i, cut in enumerate(model.cuts):
            cut.data_mean = config["normalizer_mean"][i]
            cut.data_variance = config["normalizer_variance"][i]

        return model


def LearnableCutFlowSequential(
    input_shape: tuple[int, int],
    centers: float | list[float] = 0.0,
    threshold: float = 0.5,
    features: list[str] | None = None,
    importance_min_percent: float = 0.05,
    name: str = "learnable_cut_flow_sequential",
):
    n_features = input_shape[1]
    centers_c = centers if isinstance(centers, list) else [centers] * n_features
    features = features or [f"x{i + 1}" for i in range(n_features)]

    # (batch_size, n_features)
    inputs = Input((n_features,), name="inputs")
    x = Normalization(name="normalization")(inputs)
    x = LearnableImportance(importance_min_percent, name="learnable_importance")(x)

    # n_features x (batch_size, 1)
    x = Split(name="split")(x)

    # At training: n_features x (batch_size, 2)
    # At inference: n_features x (batch_size, 1)
    x = [
        LearnableCut(
            center=centers_c[i],
            threshold=threshold,
            feature=features[i],
            name=f"learnable_cut_{i + 1}",
        )(x_i)
        for i, x_i in enumerate(x)
    ]
    # At training: (batch_size, n_features x 2)
    # At inference: (batch_size, n_features)
    x = Concatenate(name="concatenate")(x)
    outputs = Identity(name="outputs")(x)

    return LearnableCutFlowSequentialModel(
        inputs=inputs,
        outputs=outputs,
        centers=centers_c,
        threshold=threshold,
        features=features,
        importance_min_percent=importance_min_percent,
        name=name,
    )


@register_keras_serializable()
class LearnableCutFlowSequentialModel(Model):
    def __init__(
        self,
        centers: list[float],
        threshold: float,
        features: list[str],
        importance_min_percent: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.centers = centers
        self.threshold = threshold
        self.features = features
        self.importance_min_percent = importance_min_percent
        self.kwargs = kwargs

        self.normalizer = cast(Normalization, self.layers[1])
        self.importance = cast(LearnableImportance, self.layers[2])
        self.cuts = cast(list[LearnableCut], self.layers[4:-2])

    def adapt(self, x):
        self.normalizer.adapt(x)

        data_mean = ops.convert_to_numpy(self.normalizer.mean).tolist()[0]
        data_mean = cast(list[float], data_mean)
        data_variance = ops.convert_to_numpy(self.normalizer.variance).tolist()[0]
        data_variance = cast(list[float], data_variance)

        for i, cut in enumerate(self.cuts):
            cut.data_mean = data_mean[i]
            cut.data_variance = data_variance[i]
            cut.importance_score.assign(self.importance.scores[i])

    def call(self, inputs, training=None):
        y = super().call(inputs, training=training)

        if training:
            return y
        else:
            y = ops.where(
                ops.greater(self.importance.scores, self.importance.baseline), y, 1.0
            )
            y = ops.all(ops.greater(y, self.threshold), axis=1, keepdims=True)
            y = ops.cast(y, dtype="float32")
            return y

    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        del sample_weight, training

        if self._compile_loss is not None:
            loss = 0.0
            mask = ops.ones_like(y)

            for i in range(len(self.cuts)):
                y_pred_lower = ops.take(y_pred, [2 * i], axis=1)
                y_pred_upper = ops.take(y_pred, [2 * i + 1], axis=1)

                x_i = ops.take(x, i, axis=1)
                mask_lower = ops.where(ops.less(x_i, self.centers[i]), 1.0, 0.0)
                mask_upper = ops.where(ops.greater(x_i, self.centers[i]), 1.0, 0.0)

                mask_lower = ops.multiply(mask_lower, mask)
                mask_upper = ops.multiply(mask_upper, mask)

                loss_lower = self._compile_loss(y, y_pred_lower, mask_lower)
                loss_upper = self._compile_loss(y, y_pred_upper, mask_upper)
                loss += loss_lower + loss_upper

                mask_pass_lower = ops.where(
                    ops.greater(y_pred_lower, self.threshold), 1.0, 0.0
                )
                mask_pass_upper = ops.where(
                    ops.greater(y_pred_upper, self.threshold), 1.0, 0.0
                )
                mask_pass = ops.multiply(mask_pass_lower, mask_pass_upper)
                mask = ops.multiply(mask, ops.squeeze(mask_pass, axis=1))

            loss /= 2 * len(self.cuts)
            return loss

    def get_config(self):
        config = super().get_config()

        normalizer_mean = ops.convert_to_numpy(self.normalizer.mean).tolist()[0]
        normalizer_variance = ops.convert_to_numpy(self.normalizer.variance).tolist()[0]

        config.update(
            {
                "centers": self.centers,
                "threshold": self.threshold,
                "features": self.features,
                "importance_min_percent": self.importance_min_percent,
                "normalizer_mean": normalizer_mean,
                "normalizer_variance": normalizer_variance,
                **self.kwargs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = LearnableCutFlowSequential(
            input_shape=config["inputs"]["config"]["shape"],
            centers=config["centers"],
            threshold=config["threshold"],
            features=config["features"],
            importance_min_percent=config["importance_min_percent"],
            name=config["name"],
        )

        for i, cut in enumerate(model.cuts):
            cut.data_mean = config["normalizer_mean"][i]
            cut.data_variance = config["normalizer_variance"][i]

        return model


LearnableCutFlowModel = LearnableCutFlowParallelModel | LearnableCutFlowSequentialModel
