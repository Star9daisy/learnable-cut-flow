from keras import ops
from keras.initializers import Constant
from keras.layers import Layer
from keras.saving import register_keras_serializable


@register_keras_serializable()
class Split(Layer):
    """Split inputs into a list of tensors according to the number of features.

    Input shape
    -----------
    2D tensor with shape (batch_size, n_features)

    Output shape
    ------------
    A list of tensors, each with shape (batch_size, 1)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.axis = 1

    def build(self, input_shape):
        self.n_features = input_shape[self.axis]

    def call(self, inputs):
        return ops.split(inputs, self.n_features, axis=self.axis)


@register_keras_serializable()
class LearnableImportance(Layer):
    """Scale inputs by softmax-normalized trainable weights.

    Parameters
    ----------
    min_percent: float
        The minimum percentage of the importance score to be considered.

    Properties
    ----------
    baseline: Tensor
        The minimum importance score to be considered.
    scores: Tensor
        The importance scores of features.

    Input shape
    -----------
    2D tensor with shape (batch_size, n_features)

    Output shape
    ------------
    2D tensor with shape (batch_size, n_features)
    """

    def __init__(self, min_percent: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.min_percent = min_percent

    def build(self, input_shape):
        self.n_features = input_shape[1]
        self.kernel = self.add_weight((self.n_features,), initializer="ones")

    @property
    def baseline(self):
        return ops.multiply(ops.divide(1, self.n_features), self.min_percent)

    @property
    def scores(self):
        return ops.softmax(self.kernel)

    def call(self, inputs):
        return self.scores * inputs


@register_keras_serializable()
class LearnableCut(Layer):
    """Learn the cut experssion during training two neurons.

    Parameters
    ----------
    center: float
        The center to split the feature distribution into two parts to learn.
    threshold: float
        The probability threshold to determine the pass/fail of the cut.
    feature: str
        The name of the feature to show in the experssion.
    importance_score: float
        The importance score of the feature to scale the boundaries.
    data_mean: float
        The mean of the feature to shift the boundaries.
    data_variance: float
        The variance of the feature to scale the boundaries.

    Properties
    ----------
    boundaries: Tensor
        The boundaries of the cut in the original feature range.
    directions: Tensor
        The directions of the boundaries.
    case: int
        The case of the cut.
    index: int
        The index of the boundary to use in the experssion.

    Input shape
    -----------
    2D tensor with shape (batch_size, n_features)

    Output shape
    ------------
    At training: 2D tensor with shape (batch_size, 2 * n_features);
    At inference: 2D tensor with shape (batch_size, n_features)
    """

    LEFT = 1
    RIGHT = 2
    MIDDLE = 3
    EDGE = 4

    def __init__(
        self,
        center: float = 0.0,
        threshold: float = 0.5,
        feature: str = "x",
        importance_score: float = 1.0,
        data_mean: float = float("nan"),
        data_variance: float = float("nan"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.center = center
        self.threshold = threshold
        self.feature = feature
        self.importance_score = self.add_weight(
            initializer=Constant(importance_score), trainable=False
        )
        self.data_mean = data_mean
        self.data_variance = data_variance

        self.activation = ops.sigmoid
        self.inverse_activation = lambda x: ops.negative(ops.log(1 / x - 1))

        self.kernels = self.add_weight(shape=(2,))
        self.biases = self.add_weight(shape=(2,))

    def __str__(self) -> str:
        if self.case == self.LEFT:
            boundary = self.boundaries[self.index]
            return f"{self.feature} < {boundary:.4f}"

        elif self.case == self.RIGHT:
            boundary = self.boundaries[self.index]
            return f"{self.feature} > {boundary:.4f}"

        elif self.case == self.MIDDLE:
            lower, upper = self.boundaries
            return f"{lower:.4f} < {self.feature} < {upper:.4f}"

        else:
            lower, upper = self.boundaries
            return f"{self.feature} < {lower:.4f} or {self.feature} > {upper:.4f}"

    @property
    def boundaries(self):
        logits = self.inverse_activation(self.threshold)
        boundaries = ops.squeeze((logits - self.biases) / self.kernels)
        boundaries = ops.divide(boundaries, self.importance_score)
        boundaries = ops.multiply(ops.sqrt(self.data_variance), boundaries)
        boundaries = ops.add(boundaries, self.data_mean)
        return boundaries

    @property
    def directions(self):
        return ops.squeeze(ops.sign(self.kernels))

    @property
    def case(self):
        boundary_lower, boundary_upper = self.boundaries
        direction_lower, direction_upper = self.directions

        return ops.cond(
            # Lower boundary is valid
            ops.less(boundary_lower, self.center),
            true_fn=lambda: ops.cond(
                ops.greater(boundary_upper, self.center),
                # Upper boundary is also valid
                true_fn=lambda: ops.cond(
                    ops.logical_and(
                        ops.less(direction_lower, 0),
                        ops.less(direction_upper, 0),
                    ),
                    true_fn=lambda: self.LEFT,
                    false_fn=lambda: ops.cond(
                        ops.logical_and(
                            ops.greater(direction_lower, 0),
                            ops.greater(direction_upper, 0),
                        ),
                        true_fn=lambda: self.RIGHT,
                        false_fn=lambda: ops.cond(
                            ops.logical_and(
                                ops.greater(direction_lower, 0),
                                ops.less(direction_upper, 0),
                            ),
                            true_fn=lambda: self.MIDDLE,
                            false_fn=lambda: self.EDGE,
                        ),
                    ),
                ),
                # Upper boundary is invalid -> Lower direction determines the case
                false_fn=lambda: ops.cond(
                    ops.less(direction_lower, 0),
                    true_fn=lambda: self.LEFT,
                    false_fn=lambda: self.RIGHT,
                ),
            ),
            # Lower boundary is invalid
            false_fn=lambda: ops.cond(
                # Neither lower nor upper are valid
                ops.less(boundary_upper, self.center),
                true_fn=lambda: ops.cond(
                    ops.logical_and(
                        ops.less(direction_lower, 0),
                        ops.less(direction_upper, 0),
                    ),
                    true_fn=lambda: self.LEFT,
                    false_fn=lambda: ops.cond(
                        ops.logical_and(
                            ops.greater(direction_lower, 0),
                            ops.greater(direction_upper, 0),
                        ),
                        true_fn=lambda: self.RIGHT,
                        false_fn=lambda: self.EDGE,
                    ),
                ),
                false_fn=lambda: ops.cond(
                    # Upper boundary is valid -> Upper direction determines the case
                    ops.less(direction_upper, 0),
                    true_fn=lambda: self.LEFT,
                    false_fn=lambda: self.RIGHT,
                ),
            ),
        )

    @property
    def index(self):
        boundary_lower, boundary_upper = self.boundaries

        return ops.cond(
            ops.equal(self.case, self.LEFT),
            true_fn=lambda: ops.cond(
                # One that is valid determines the index
                ops.logical_and(
                    ops.less(boundary_lower, self.center),
                    ops.less(boundary_upper, self.center),
                ),
                true_fn=lambda: 0,
                false_fn=lambda: ops.cond(
                    ops.logical_and(
                        ops.greater(boundary_lower, self.center),
                        ops.greater(boundary_upper, self.center),
                    ),
                    true_fn=lambda: 1,
                    # Both are valid or invalid -> Left direction determines the index
                    false_fn=lambda: 0,
                ),
            ),
            false_fn=lambda: ops.cond(
                # Same logic as above, but for the right case
                ops.equal(self.case, self.RIGHT),
                true_fn=lambda: ops.cond(
                    ops.logical_and(
                        ops.less(boundary_lower, self.center),
                        ops.less(boundary_upper, self.center),
                    ),
                    true_fn=lambda: 0,
                    false_fn=lambda: ops.cond(
                        ops.logical_and(
                            ops.greater(boundary_lower, self.center),
                            ops.greater(boundary_upper, self.center),
                        ),
                        true_fn=lambda: 1,
                        # Both are valid or invalid -> Right direction determines the index
                        false_fn=lambda: 1,
                    ),
                ),
                # Middle and edge cases are assigned -1 to show both indices are needed
                false_fn=lambda: -1,
            ),
        )

    def call(self, inputs, training=None):
        y_lower = self.activation(inputs * self.kernels[0] + self.biases[0])
        y_upper = self.activation(inputs * self.kernels[1] + self.biases[1])
        y = ops.concatenate([y_lower, y_upper], axis=1)

        # When training, return the probabilities of both sides to compute the loss
        if training:
            return y

        # When inference, a cut layer should return a binary output according to the case
        else:
            return ops.cond(
                ops.logical_or(
                    ops.equal(self.case, self.LEFT),
                    ops.equal(self.case, self.RIGHT),
                ),
                true_fn=lambda: ops.where(
                    # Left or right cases -> The indexed output should pass the threshold
                    ops.greater(ops.take(y, [self.index], axis=1), self.threshold),
                    1.0,
                    0.0,
                ),
                false_fn=lambda: ops.cond(
                    ops.equal(self.case, self.MIDDLE),
                    true_fn=lambda: ops.where(
                        # Middle case -> All outputs should pass the threshold
                        ops.all(ops.greater(y, self.threshold), axis=1, keepdims=True),
                        1.0,
                        0.0,
                    ),
                    false_fn=lambda: ops.where(
                        # Edge case -> At least one output should pass the threshold
                        ops.any(ops.greater(y, self.threshold), axis=1, keepdims=True),
                        1.0,
                        0.0,
                    ),
                ),
            )

    def compute_output_shape(self, input_shape):
        # Since the behavior of the layer is different in training and inference,
        # we have to return the output shape manually for the training
        return (input_shape[0], 2)
