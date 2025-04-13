from keras.saving import register_keras_serializable
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def jaccard_index(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return intersection / total

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))

import keras.backend as K

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=0.25, gamma=2.0):
    def focal(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)  # Evita log(0)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.math.pow(y_pred, gamma)
        return tf.reduce_mean(weight * bce)
    return focal



@tf.keras.utils.register_keras_serializable()
def combined_loss(alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5):
    def loss(y_true, y_pred):
        return dice_weight * dice_loss(y_true, y_pred) + focal_weight * focal_loss(alpha, gamma)(y_true, y_pred)
    return loss

