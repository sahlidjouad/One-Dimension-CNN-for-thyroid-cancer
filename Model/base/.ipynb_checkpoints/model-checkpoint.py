from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, AveragePooling1D, Flatten, Input, concatenate
import tensorflow as tf


def make_model(input_names, batch_size):
    input_layers = []
    for k in input_names:
        input_layers.append(Input(shape=(1, 1), batch_size=batch_size, name=k))

    x = concatenate(input_layers, axis=1)

    x = Conv1D( padding= "same", activation=tf.math.tanh, kernel_size=2, filters=2)(x)
    x = Conv1D( padding= "same", activation=tf.math.tanh, kernel_size=2, filters=4)(x)
    x = Conv1D( padding= "valid", activation=tf.math.tanh, kernel_size=2, filters=8)(x)
    x = AveragePooling1D(pool_size=2, strides=1)(x)
    
    x = Conv1D( padding= "same", activation=tf.math.tanh, kernel_size=2, filters=2)(x)
    x = Conv1D( padding= "same", activation=tf.math.tanh, kernel_size=2, filters=4)(x)
    x = Conv1D( padding= "valid", activation=tf.math.tanh, kernel_size=2, filters=8)(x)
    x = AveragePooling1D(pool_size=2, strides=1)(x)
    
    x = Flatten()(x)

    x = Dense(units=44, activation=tf.math.tanh)(x)
    x = Dense(units=44, activation=tf.math.tanh)(x)

    x = Dense(units=1, activation= tf.keras.activations.sigmoid )(x)

    return Model(input_layers, x)
