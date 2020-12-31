import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models


inputs = keras.Input(shape=(48, 48, 3))
paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])

x0 = tf.pad(inputs, paddings, "SYMMETRIC")
x1_1 = layers.Conv2D(64, (5,5), activation='relu')(x0) #48 64

paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

x1 = tf.pad(x1_1, paddings, "SYMMETRIC")
x2_1 = layers.Conv2D(64, (3,3), activation='relu')(x1) #48 64

x1n2 = layers.Concatenate()([x1_1, x2_1]) #48 128

x2 = tf.pad(x1n2, paddings, "SYMMETRIC")
x3_1 = layers.Conv2D(128, (3,3), activation='relu')(x2) #48 128

x1n2n3 = layers.Concatenate() ([x1_1, x2_1, x3_1]) #48 256

x3 = tf.pad(x1n2n3, paddings, "SYMMETRIC")
x4 = layers.Conv2D(3 * (2 ** 2) * 4, (3,3), activation='relu')(x3)

# outputs = tf.nn.depth_to_space(x4, 2, data_format='NHWC', name=None)
x5 = tf.nn.depth_to_space(x4, 2, data_format='NHWC', name=None)


x6 = tf.pad(x5, paddings, "SYMMETRIC")
outputs = layers.Conv2D(3, (3, 3), activation='sigmoid')(x6)

model = keras.Model(inputs, outputs)