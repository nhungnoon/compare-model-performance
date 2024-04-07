# based on
# https://www.tensorflow.org/tutorials/customization/custom_layers
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    BatchNormalization,
    Dropout,
    Conv2D,
    Dense,
    Flatten,
    ReLU,
)


class CustomizedBlock(Layer):

    def __init__(self, out_filters, **kwargs):
        # initialize residualblock
        super(CustomizedBlock, self).__init__(**kwargs)
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
        self.drop_1 = Dropout(0.25)
        self.conv_3 = Conv2D(self.out_filters, (1, 1))
        self.batch_norm_3 = BatchNormalization()
        self.relu = ReLU()
        self.batch_norm_4 = BatchNormalization()

    def call(self, inputs, training=True):
        # pass inputs through layers
        x = self.batch_norm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.batch_norm_2(x)
        x = self.drop_1(x)
        x = tf.nn.relu(x)
        x = self.relu(x)
        x = self.batch_norm_4(x)
        x = self.conv_2(x)
        final = self.conv_3(x)
        return tf.add(x, final)


class MainModel(Model):

    def __init__(self, **kwargs):
        super(MainModel, self).__init__(**kwargs)
        self.conv_1 = Conv2D(32, (3, 3), strides=1)
        self.conv_2 = Conv2D(32, (3, 3))
        self.customized_layer = CustomizedBlock(64)
        self.flatten = Flatten()
        self.dense = Dense(10, activation="softmax")

    def call(self, inputs, training=True):

        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.customized_layer(x, training)
        x = self.flatten(x)

        return self.dense(x)
