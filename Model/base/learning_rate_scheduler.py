import tensorflow as tf


def scheduler(epoch, lr):

    return lr * tf.math.exp(-0.01)


def Get_scheduler_callbacks(scheduler):
    return tf.keras.callbacks.LearningRateScheduler(scheduler)
