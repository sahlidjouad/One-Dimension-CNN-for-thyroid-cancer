from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dropout,  MaxPooling1D, Dense, AveragePooling1D, Flatten, Input, concatenate
import tensorflow as tf


def make_model(input_names, batch_size):
    input_layers = []
    for k in input_names:
        input_layers.append(Input(shape=(1, 1), batch_size=batch_size, name=k))

    x = concatenate(input_layers, axis=1)

    x = Conv1D(padding="same", activation=tf.keras.activations.tanh,
               filters=8, kernel_size=6)(x)

    x = Conv1D(padding="valid", activation=tf.keras.activations.tanh,
               filters=10, kernel_size=6)(x)
    x = Dropout(0.2)(x)

    x = AveragePooling1D(padding="valid", strides=1, pool_size=6)(x)

    x = Conv1D(padding="valid", activation=tf.keras.activations.tanh,
               filters=13, kernel_size=6)(x)

    x = Conv1D(padding="same", activation=tf.keras.activations.tanh,
               filters=20, kernel_size=6)(x)

    x = Conv1D(padding="valid", activation=tf.keras.activations.tanh,
               filters=25, kernel_size=6)(x)
    x = Dropout(0.2)(x)

    x = MaxPooling1D(padding="valid", strides=1, pool_size=6)(x)

    x = Flatten()(x)

    x = Dense(units=100, activation=tf.keras.activations.sigmoid)(x)
    x = Dropout(0.2)(x)

    x = Dense(units=1, activation=tf.keras.activations.sigmoid)(x)
    return Model(input_layers, x)
