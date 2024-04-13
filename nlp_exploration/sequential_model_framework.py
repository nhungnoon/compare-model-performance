import tensorflow as tf
from tensorflow.keras.models import Model
from keras import regularizers
from tensorflow.keras.layers import Layer, Dense, Dropout


class NLPSequentialModel(Model, Layer):

    def __init__(self, **kwargs):
        super(NLPSequentialModel, self).__init__(**kwargs)
        self.dense_1 = Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l1(0.001),
        )
        # drop out to avoid overfitting
        self.drop_layer = Dropout(0.5)
        self.dense_2 = Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        )
        # TODO make the class number flexible
        self.final_dense = Dense(46, activation="softmax")

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 64),
            initializer="random_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(64,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        inputs = tf.matmul(inputs, self.w) + self.b
        x = self.dense_1(inputs)
        x = self.drop_layer(x)
        x = self.dense_2(x)
        x = self.final_dense(x)

        return x
