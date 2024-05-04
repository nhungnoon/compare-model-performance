import os
import random
import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


def sequential_model(
  x_train, x_test,
  y_train, y_test,
):
  model = Sequential([
      Dense(100, input_dim=784, kernel_initializer=glorot_uniform(seed=42)),
      Activation("relu"),
      Dropout(0.3),
      BatchNormalization()
  ])

  for n_neurons in [200, 100]:
    model.add(Dense(n_neurons, activation="relu", kernel_initializer=glorot_uniform(seed=42)))
    model.add(Dropout(0.45, seed=42))
    model.add(BatchNormalization())

  model.add(Dense(10, activation="softmax", kernel_initializer=glorot_uniform(seed=42)))
  model.compile(optimizer=Adam(decay = 1.3e-6), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

  #early stopping
  early_stop_check = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=200)

  #create checkpoint and choose to save the best model
  model_c = ModelCheckpoint(
    'best_model.h5.keras', 
    monitor='val_accuracy', 
    mode='max', 
    verbose=1, 
    save_best_only=True
  )


  model.fit(x_train, y_train, batch_size=256, epochs=3, 
          validation_data=(x_test, y_test),
          callbacks=[model_c, early_stop_check ])

