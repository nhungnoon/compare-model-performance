"""
A sequential model  
Based on https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout


class SequentialModel(Model, Layer):

    def __init__(self, rate=1e-2, **kwargs):
        super(SequentialModel, self).__init__(**kwargs)
        self.rate = rate
        self.conv_1 = Conv2D(32, (3, 3), strides=2, activation="relu")
        self.drop_layer = Dropout(0.5)
        self.conv_2 = Conv2D(32, (3, 3))
        self.flatten = Flatten()
        self.dense = Dense(128, activation="relu")
        self.final_dense = Dense(10, activation="softmax")

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 32),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(32,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        inputs = tf.matmul(inputs, self.w) + self.b
        x = self.conv_1(inputs)
        self.add_loss(self.rate * tf.reduce_mean(inputs))
        x = self.drop_layer(x)
        x = self.conv_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.final_dense(x)

        return x


def find_loss_gradients(model, inputs, targets, loss):
    with tf.GradientTape() as t:
        predictions = model(inputs)
        loss_update = loss(targets, predictions)
        # Add extra losses
        loss_update += sum(model.losses)
    grad_value = t.gradient(loss_update, model.trainable_variables)

    return loss_update, grad_value


def train_model(model, num_epochs, dataset, optimizer, loss, grad_fn):

    train_loss_results = []
    train_accuracy_results = []
    for epoch in range(num_epochs):

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for x, y in train_dataset:

            loss_value, grads = grad_fn(model, x, y, loss)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg(loss_value)
            epoch_accuracy(to_categorical(y), model(x))

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

    return train_loss_results, train_accuracy_results
