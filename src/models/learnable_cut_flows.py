from keras import ops
from keras.layers import Concatenate, Identity, Input, Normalization
from keras.models import Model
from keras.saving import register_keras_serializable

from .layers import LearnableCut, LearnableImportance, Split


def LearnableCutFlowParallel(
    input_shape,
    centers=0.0,
    threshold=0.5,
    feature_names=None,
    name="learnable_cut_flow_parallel",
    importance_min_percent=0.05,
):
    n_features = input_shape[1]
    centers = centers if isinstance(centers, list) else [centers] * n_features
    feature_names = feature_names or [f"x{i + 1}" for i in range(n_features)]

    # Inputs shape: (batch_size, n_features)
    inputs = Input((n_features,), name="inputs")
    x = Normalization(name="normalization")(inputs)
    x = LearnableImportance(name="learnable_importance")(x)

    # Feature list: n_features x (batch_size, 1)
    x = Split(name="split")(x)

    # Learnable cut list:
    # n_features x (batch_size, 2) at training
    # n_features x (batch_size, 1) at inference
    x = [
        LearnableCut(
            center=centers[i],
            threshold=threshold,
            feature_name=feature_names[i],
            name=f"learnable_cut_{i + 1}",
        )(x_i)
        for i, x_i in enumerate(x)
    ]

    # Results shape:
    # (batch_size, n_features x 2) at training
    # (batch_size, n_features) at inference
    x = Concatenate(name="concatenate")(x)
    outputs = Identity(name="outputs")(x)

    return LearnableCutFlowParallelModel(
        inputs=inputs,
        outputs=outputs,
        centers=centers,
        threshold=threshold,
        feature_names=feature_names,
        name=name,
        importance_min_percent=importance_min_percent,
    )


@register_keras_serializable()
class LearnableCutFlowParallelModel(Model):
    def __init__(self, importance_min_percent=0.05, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.importance_min_percent = importance_min_percent

        self.normalization = self.layers[1]
        self.learnable_importance = self.layers[2]
        self.learnable_cuts = self.layers[4:-2]
        self.threshold = self.learnable_cuts[0].threshold

    @property
    def importance_baseline(self):
        return 1 / len(self.learnable_cuts) * self.importance_min_percent

    @property
    def learned_importance(self):
        return self.learnable_importance.scores

    @property
    def learned_cuts(self):
        return [i["cut"] for i in self.learned_cuts_report]

    @property
    def learned_cuts_report(self):
        report = []
        normalization = self.normalization
        normalization.invert = True

        for i, learnable_cut in enumerate(self.learnable_cuts):
            boundaries = learnable_cut.boundaries
            directions = learnable_cut.directions
            case = learnable_cut.case
            index = learnable_cut.index
            feature_name = learnable_cut.feature_name

            boundaries = boundaries / self.learned_importance[i]
            boundaries = normalization(ops.reshape(boundaries, (-1, 1)))[:, i]

            if case == LearnableCut.LEFT:
                case = "left"
                cut = f"{feature_name} < {boundaries[index]:.4f}"
            elif case == LearnableCut.RIGHT:
                case = "right"
                cut = f"{feature_name} > {boundaries[index]:.4f}"
            elif case == LearnableCut.MIDDLE:
                case = "middle"
                lower, upper = boundaries
                cut = f"{lower:.4f} < {feature_name} < {upper:.4f}"
            else:
                case = "edge"
                lower, upper = boundaries
                cut = f"{feature_name} < {lower:.4f} or {feature_name} > {upper:.4f}"

            report.append(
                {
                    "boundaries": boundaries,
                    "directions": directions,
                    "case": case,
                    "index": index,
                    "cut": cut,
                }
            )

        normalization.invert = False
        return report

    def call(self, inputs, training=None):
        outputs = super().call(inputs, training=training)
        if training:
            return outputs
        else:
            # Inference:
            # 1. Check if the feature's importance is greater than the baseline
            #      No: set the output to 1.0 so that the feature is not used.
            #      Yes: move to the next step.
            # 2. Check if the feature's output is greater than the threshold
            #      No: set the output to 0.0 meaning the event is tagged as background.
            #      Yes: set the output to 1.0 meaning the event is tagged as signal.
            # 3. Convert the results from step 2 from boolean to float32 to
            #    represent the probability of the event being signal
            outputs = ops.where(
                self.learned_importance > self.importance_baseline, outputs, 1.0
            )
            outputs = ops.all(outputs > self.threshold, axis=1, keepdims=True)
            outputs = ops.cast(outputs, dtype="float32")
            return outputs

    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        if self._compile_loss is not None:
            loss = 0.0
            for i, learnable_cut in enumerate(self.learnable_cuts):
                # Get the left and right output around the center of the cut
                y_pred_left = ops.take(y_pred, [2 * i], axis=1)
                y_pred_right = ops.take(y_pred, [2 * i + 1], axis=1)

                # Get the mask for the left and right output
                x_i = ops.take(x, i, axis=1)
                mask_left = ops.where(x_i < learnable_cut.center, 1.0, 0.0)
                mask_right = ops.where(x_i > learnable_cut.center, 1.0, 0.0)

                # Calculate the loss for the left and right output
                # The other side of loss is masked out
                loss_left = self._compile_loss(y, y_pred_left, mask_left)
                loss_right = self._compile_loss(y, y_pred_right, mask_right)
                loss += loss_left + loss_right

            # Average the loss over all the learnable cuts and all sides
            loss /= 2 * len(self.learnable_cuts)
            return loss

    # For saving and loading the model
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "importance_min_percent": self.importance_min_percent,
                **self.kwargs,
            }
        )
        return config

    # For saving and loading the model
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return LearnableCutFlowParallel(
            input_shape=config["inputs"]["config"]["shape"],
            centers=config["centers"],
            threshold=config["threshold"],
            feature_names=config["feature_names"],
            name=config["name"],
            importance_min_percent=config["importance_min_percent"],
        )


def LearnableCutFlowSequential(
    input_shape,
    centers=0.0,
    threshold=0.5,
    feature_names=None,
    name="learnable_cut_flow_sequential",
    importance_min_percent=0.05,
):
    n_features = input_shape[1]
    centers = centers if isinstance(centers, list) else [centers] * n_features
    feature_names = feature_names or [f"x{i + 1}" for i in range(n_features)]

    # Inputs shape: (batch_size, n_features)
    inputs = Input((n_features,), name="inputs")
    x = Normalization(name="normalization")(inputs)
    x = LearnableImportance(name="learnable_importance")(x)

    # Feature list: n_features x (batch_size, 1)
    x = Split(name="split")(x)

    # Learnable cut list:
    # n_features x (batch_size, 2) at training
    # n_features x (batch_size, 1) at inference
    x = [
        LearnableCut(
            center=centers[i],
            threshold=threshold,
            feature_name=feature_names[i],
            name=f"learnable_cut_{i + 1}",
        )(x_i)
        for i, x_i in enumerate(x)
    ]

    # Results shape:
    # (batch_size, n_features x 2) at training
    # (batch_size, n_features) at inference
    x = Concatenate(name="concatenate")(x)
    outputs = Identity(name="outputs")(x)

    return LearnableCutFlowSequentialModel(
        inputs=inputs,
        outputs=outputs,
        centers=centers,
        threshold=threshold,
        feature_names=feature_names,
        name=name,
        importance_min_percent=importance_min_percent,
    )


@register_keras_serializable()
class LearnableCutFlowSequentialModel(Model):
    def __init__(self, importance_min_percent=0.05, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.importance_min_percent = importance_min_percent

        self.normalization = self.layers[1]
        self.learnable_importance = self.layers[2]
        self.learnable_cuts = self.layers[4:-2]
        self.threshold = self.learnable_cuts[0].threshold

    @property
    def importance_baseline(self):
        return 1 / len(self.learnable_cuts) * self.importance_min_percent

    @property
    def learned_importance(self):
        return self.learnable_importance.scores

    @property
    def learned_cuts(self):
        return [i["cut"] for i in self.learned_cuts_report]

    @property
    def learned_cuts_report(self):
        report = []
        normalization = self.normalization
        normalization.invert = True

        for i, learnable_cut in enumerate(self.learnable_cuts):
            boundaries = learnable_cut.boundaries
            directions = learnable_cut.directions
            case = learnable_cut.case
            index = learnable_cut.index
            feature_name = learnable_cut.feature_name

            boundaries = boundaries / self.learned_importance[i]
            boundaries = normalization(ops.reshape(boundaries, (-1, 1)))[:, i]

            if case == LearnableCut.LEFT:
                case = "left"
                cut = f"{feature_name} < {boundaries[index]:.4f}"
            elif case == LearnableCut.RIGHT:
                case = "right"
                cut = f"{feature_name} > {boundaries[index]:.4f}"
            elif case == LearnableCut.MIDDLE:
                case = "middle"
                lower, upper = boundaries
                cut = f"{lower:.4f} < {feature_name} < {upper:.4f}"
            else:
                case = "edge"
                lower, upper = boundaries
                cut = f"{feature_name} < {lower:.4f} or {feature_name} > {upper:.4f}"

            report.append(
                {
                    "boundaries": boundaries,
                    "directions": directions,
                    "case": case,
                    "index": index,
                    "cut": cut,
                }
            )

        normalization.invert = False
        return report

    def call(self, inputs, training=None):
        outputs = super().call(inputs, training=training)
        if training:
            return outputs
        else:
            # Inference:
            # 1. Check if the feature's importance is greater than the baseline
            #      No: set the output to 1.0 so that the feature is not used.
            #      Yes: move to the next step.
            # 2. Check if the feature's output is greater than the threshold
            #      No: set the output to 0.0 meaning the event is tagged as background.
            #      Yes: set the output to 1.0 meaning the event is tagged as signal.
            # 3. Convert the results from step 2 from boolean to float32 to
            #    represent the probability of the event being signal
            outputs = ops.where(
                self.learned_importance > self.importance_baseline, outputs, 1.0
            )
            outputs = ops.all(outputs > self.threshold, axis=1, keepdims=True)
            outputs = ops.cast(outputs, dtype="float32")
            return outputs

    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        if self._compile_loss is not None:
            loss = 0.0
            mask = ops.ones_like(y)
            for i, learnable_cut in enumerate(self.learnable_cuts):
                # Get the left and right output around the center of the cut
                y_pred_left = ops.take(y_pred, [2 * i], axis=1)
                y_pred_right = ops.take(y_pred, [2 * i + 1], axis=1)

                # Get the mask for the left and right output
                x_i = ops.take(x, i, axis=1)
                mask_left = ops.where(x_i < learnable_cut.center, 1.0, 0.0)
                mask_right = ops.where(x_i > learnable_cut.center, 1.0, 0.0)

                # Take the previous mask into account
                mask_left = ops.multiply(mask_left, mask)
                mask_right = ops.multiply(mask_right, mask)

                # Calculate the loss for the left and right output
                # The other side of loss is masked out
                loss_left = self._compile_loss(y, y_pred_left, mask_left)
                loss_right = self._compile_loss(y, y_pred_right, mask_right)
                loss += loss_left + loss_right

                # Update the mask
                mask_pass_left = ops.where(y_pred_left > self.threshold, 1.0, 0.0)
                mask_pass_right = ops.where(y_pred_right > self.threshold, 1.0, 0.0)
                mask_pass = ops.multiply(mask_pass_left, mask_pass_right)
                mask = ops.multiply(mask, ops.squeeze(mask_pass, axis=1))

            # Average the loss over all the learnable cuts and all sides
            loss /= 2 * len(self.learnable_cuts)
            return loss

    # For saving and loading the model
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "importance_min_percent": self.importance_min_percent,
                **self.kwargs,
            }
        )
        return config

    # For saving and loading the model
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return LearnableCutFlowSequential(
            input_shape=config["inputs"]["config"]["shape"],
            centers=config["centers"],
            threshold=config["threshold"],
            feature_names=config["feature_names"],
            name=config["name"],
            importance_min_percent=config["importance_min_percent"],
        )
