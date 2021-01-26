import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models


paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

def basic(x, num):
    x = tf.pad(x, paddings, "SYMMETRIC")
    x = layers.Conv2D(num*2, (3, 3), activation='relu') (x)

    x = tf.pad(x, paddings, "SYMMETRIC")
    x = layers.Conv2D(num*2, (3, 3), activation='relu') (x)

    x = tf.pad(x, paddings, "SYMMETRIC")
    x = layers.Conv2D(num*2, (3, 3), activation='relu') (x)

    x = layers.Conv2D(num, (1, 1), activation='relu') (x)

    return x

def down(x, num):
    x = tf.pad(x, paddings, "SYMMETRIC")
    x = layers.Conv2D(num/2, (3, 3), activation='relu') (x)

    x = tf.pad(x, paddings, "SYMMETRIC")
    x = layers.Conv2D(num/2, (3, 3), activation='relu') (x)

    x = tf.pad(x, paddings, "SYMMETRIC")
    x = layers.Conv2D(num, (3, 3), activation='relu') (x)

    return x

Input_img = keras.Input(shape=(48, 48, 3)) #48

x1 =    down(x = Input_img, num = 512)

x2 =        layers.MaxPooling2D((2, 2))(x1) #24

x3 =        basic (x = x2, num = 256)

x4 =    layers.UpSampling2D((2, 2))(x3) #48

x4 =    layers.Concatenate() ([x1, x4]) #48

x5 =    basic (x = x4, num = 128)

x6 = layers.UpSampling2D((2, 2))(x5) #96

x7 = basic (x = x6, num = 64)


x8 = tf.pad(x7, paddings, "SYMMETRIC") #98
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid')(x8) #96

#model done
model = keras.Model(Input_img, decoded)