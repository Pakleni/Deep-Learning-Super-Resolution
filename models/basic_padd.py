import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models


paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])


Input_img = keras.Input(shape=(48, 48, 3)) #48

x1 = tf.pad(Input_img, paddings, "SYMMETRIC") #98
x2 = layers.Conv2D(256, (3, 3), activation='relu') (x1) #96

x3 = layers.UpSampling2D(size=(2, 2))(x2) #96

x4 = tf.pad(x3, paddings, "SYMMETRIC") #98
x5 = layers.Conv2D(128, (3, 3), activation='relu') (x4) #96

x6 = tf.pad(x5, paddings, "SYMMETRIC") #98
x7 = layers.Conv2D(64, (3, 3), activation='relu') (x6) #96

x8 = tf.pad(x7, paddings, "SYMMETRIC") #98
x9 = layers.Conv2D(32, (3, 3), activation='relu') (x8) #96

x10 = tf.pad(x9, paddings, "SYMMETRIC") #98
x11 = layers.Conv2D(16, (3, 3), activation='relu') (x10) #96

x12 = tf.pad(x11, paddings, "SYMMETRIC") #98
x13 = layers.Conv2D(8, (3, 3), activation='relu') (x12) #96

x14 = tf.pad(x13, paddings, "SYMMETRIC") #98
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid')(x14) #96


#model done
model = keras.Model(Input_img, decoded)