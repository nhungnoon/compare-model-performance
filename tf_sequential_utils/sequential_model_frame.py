"""
A sequential model  
Based on https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout

class TFSequentialModel(Model):

    def __init__(self, **kwargs):
        super(TFSequentialModel, self).__init__(**kwargs)

        self.conv_1 = Conv2D(32, (3, 3), strides=2, activation="relu")
        self.drop_layer = Dropout(0.5)
        self.conv_2 = Conv2D(32, (3, 3))
        self.flatten = Flatten()
        self.dense = Dense(128, activation="relu")
        self.final_dense = Dense(10, activation="softmax")

    def call(self, inputs):

        x = self.conv_1(inputs)
        x = self.drop_layer(x)
        x = self.conv_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.final_dense(x)

        return x
