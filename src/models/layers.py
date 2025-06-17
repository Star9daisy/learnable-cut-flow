from keras import ops
from keras.layers import Layer
from keras.saving import register_keras_serializable


@register_keras_serializable()
class Split(Layer):
    """A layer that splits the input according to the number of features.

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
        assert len(input_shape) == 2, "Inputs must be a 2D tensor"
        self.n_features = input_shape[self.axis]

    def call(self, inputs):
        assert len(ops.shape(inputs)) == 2, "Inputs must be a 2D tensor"
        return ops.split(inputs, indices_or_sections=self.n_features, axis=self.axis)


@register_keras_serializable()
class LearnableImportance(Layer):
    """A layer that uses normalized trainable weights to compute the importance
    of each feature.

    Input shape
    -----------
    2D tensor with shape (batch_size, n_features)

    Output shape
    -----------
    2D tensor with shape (batch_size, n_features)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2, "Inputs must be a 2D tensor"
        self.kernel = self.add_weight(shape=input_shape[1:], initializer="ones")

    def call(self, inputs):
        assert len(ops.shape(inputs)) == 2, "Inputs must be a 2D tensor"
        return self.scores * inputs

    @property
    def scores(self):
        return ops.softmax(self.kernel)


@register_keras_serializable()
class LearnableCut(Layer):
    """A layer that learns the cut boundaries and directions.

    Input shape
    -----------
    2D tensor with shape (batch_size, n_features)

    Output shape
    ------------
    When training: 2D tensor with shape (batch_size, 2);
    When inference: 2D tensor with shape (batch_size, 1)
    """

    LEFT = 1
    RIGHT = 2
    MIDDLE = 3
    EDGE = 4

    def __init__(
        self,
        center: float = 0.0,
        threshold: float = 0.5,
        feature_name: str = "x",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.center = center
        self.threshold = threshold
        self.feature_name = feature_name

        self.activation = ops.sigmoid
        self.inverse_activation = lambda x: ops.negative(ops.log(1 / x - 1))

    def __str__(self) -> str:
        if self.case == self.LEFT:
            boundary = self.boundaries[self.index]
            return f"{self.feature_name} < {boundary:.4f}"

        elif self.case == self.RIGHT:
            boundary = self.boundaries[self.index]
            return f"{self.feature_name} > {boundary:.4f}"

        elif self.case == self.MIDDLE:
            lower, upper = self.boundaries
            return f"{lower:.4f} < {self.feature_name} < {upper:.4f}"

        else:
            lower, upper = self.boundaries
            return f"{self.feature_name} < {lower:.4f} or {self.feature_name} > {upper:.4f}"

    @property
    def boundaries(self):
        logits = self.inverse_activation(self.threshold)
        boundaries = ops.squeeze((logits - self.biases) / self.kernels)
        return boundaries

    @property
    def directions(self):
        return ops.squeeze(ops.sign(self.kernels))

    @property
    def case(self):
        boundary_left, boundary_right = self.boundaries
        direction_left, direction_right = self.directions

        return ops.cond(
            ops.greater(boundary_left, self.center),
            true_fn=lambda: ops.cond(
                ops.greater(boundary_right, self.center),
                true_fn=lambda: ops.where(
                    ops.less(direction_right, 0), self.LEFT, self.RIGHT
                ),
                false_fn=lambda: ops.cond(
                    ops.logical_and(
                        ops.less(direction_left, 0),
                        ops.less(direction_right, 0),
                    ),
                    true_fn=lambda: self.LEFT,
                    false_fn=lambda: ops.cond(
                        ops.logical_and(
                            ops.greater(direction_left, 0),
                            ops.greater(direction_right, 0),
                        ),
                        true_fn=lambda: self.RIGHT,
                        false_fn=lambda: self.EDGE,
                    ),
                ),
            ),
            false_fn=lambda: ops.cond(
                ops.less(boundary_right, self.center),
                true_fn=lambda: ops.where(
                    ops.less(direction_left, 0), self.LEFT, self.RIGHT
                ),
                false_fn=lambda: ops.cond(
                    ops.logical_and(
                        ops.less(direction_left, 0),
                        ops.less(direction_right, 0),
                    ),
                    true_fn=lambda: self.LEFT,
                    false_fn=lambda: ops.cond(
                        ops.logical_and(
                            ops.greater(direction_left, 0),
                            ops.greater(direction_right, 0),
                        ),
                        true_fn=lambda: self.RIGHT,
                        false_fn=lambda: ops.cond(
                            ops.logical_and(
                                ops.greater(direction_left, 0),
                                ops.less(direction_right, 0),
                            ),
                            true_fn=lambda: self.MIDDLE,
                            false_fn=lambda: self.EDGE,
                        ),
                    ),
                ),
            ),
        )

    @property
    def index(self):
        boundary_left, boundary_right = self.boundaries
        direction_left, direction_right = self.directions

        return ops.cond(
            ops.greater(boundary_left, self.center),
            true_fn=lambda: ops.cond(
                ops.greater(boundary_right, self.center),
                true_fn=lambda: 1,
                false_fn=lambda: ops.cond(
                    ops.logical_and(
                        ops.less(direction_left, 0),
                        ops.less(direction_right, 0),
                    ),
                    true_fn=lambda: 0,
                    false_fn=lambda: ops.cond(
                        ops.logical_and(
                            ops.greater(direction_left, 0),
                            ops.greater(direction_right, 0),
                        ),
                        true_fn=lambda: 1,
                        false_fn=lambda: -1,
                    ),
                ),
            ),
            false_fn=lambda: ops.cond(
                ops.less(boundary_right, self.center),
                true_fn=lambda: 0,
                false_fn=lambda: ops.cond(
                    ops.logical_and(
                        ops.less(direction_left, 0),
                        ops.less(direction_right, 0),
                    ),
                    true_fn=lambda: 0,
                    false_fn=lambda: ops.cond(
                        ops.logical_and(
                            ops.greater(direction_left, 0),
                            ops.greater(direction_right, 0),
                        ),
                        true_fn=lambda: 1,
                        false_fn=lambda: -1,
                    ),
                ),
            ),
        )

    def build(self, input_shape):
        self.kernels = self.add_weight(shape=(2,))
        self.biases = self.add_weight(shape=(2,))

    def call(self, inputs, training=None):
        # Outputs: (batch_size, 2)
        outputs_left = self.activation(inputs * self.kernels[0] + self.biases[0])
        outputs_right = self.activation(inputs * self.kernels[1] + self.biases[1])
        outputs = ops.concatenate([outputs_left, outputs_right], axis=1)

        if training:
            return outputs
        else:
            return ops.cond(
                ops.logical_or(self.case == self.LEFT, self.case == self.RIGHT),
                true_fn=lambda: ops.where(
                    ops.greater(
                        ops.take(outputs, [self.index], axis=1), self.threshold
                    ),
                    1.0,
                    0.0,
                ),
                false_fn=lambda: ops.cond(
                    self.case == self.MIDDLE,
                    true_fn=lambda: ops.where(
                        ops.all(
                            ops.greater(outputs, self.threshold), axis=1, keepdims=True
                        ),
                        1.0,
                        0.0,
                    ),
                    false_fn=lambda: ops.where(
                        ops.any(
                            ops.greater(outputs, self.threshold), axis=1, keepdims=True
                        ),
                        1.0,
                        0.0,
                    ),
                ),
            )

    def compute_output_shape(self, input_shape):
        return (None, 2)
