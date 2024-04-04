# based on
# https://www.tensorflow.org/tutorials/customization/custom_layers
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Add,
    Dropout,
)


class ResidualBlock(Layer):

    def __init__(self, out_filters, **kwargs):
        # initialize residualblock
        super(ResidualBlock, self).__init__(**kwargs)
        self.out_filters = out_filters

        # define layers
        self.conv_1 = Conv2D(out_filters, (3, 3), padding="same", activation="tanh")
        self.batch_norm_1 = BatchNormalization()
        self.conv_2 = Conv2D(
            self.out_filters,
            (3, 3),
            padding="same",
        )
        self.batch_norm_2 = BatchNormalization()
        self.conv_3 = Conv2D(self.out_filters, (1, 1))
        self.batch_norm_3 = BatchNormalization()

    def call(self, inputs, training=False):
        # pass inputs through layers
        x = self.batch_norm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.batch_norm_2(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        final = self.conv_3(inputs)

        return tf.add(x, final)


class MainModel(Model):

    def __init__(self, **kwargs):
        super(MainModel, self).__init__(**kwargs)
        self.conv_1 = Conv2D(32, (3, 3), strides=1)
        self.conv_2 = Conv2D(32, (3, 3))
        self.resnet_customized = ResidualBlock(64)
        self.flatten = Flatten()
        self.dense = Dense(10, activation="softmax")

    def call(self, inputs, training=False):

        x = self.conv_1(inputs)
        x = self.conv_2(inputs)
        x = self.resnet_customized(x, training)
        x = self.flatten(x)

        return self.dense(x)
