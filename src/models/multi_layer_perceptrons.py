from keras.layers import Dense, Input, Normalization
from keras.models import Model
from keras.saving import register_keras_serializable


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
        self.normalization = self.layers[1]
