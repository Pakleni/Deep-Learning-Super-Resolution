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


x1 =    down (x = Input_img, num = 256) #48

x2 =        layers.MaxPooling2D((2, 2))(x1) #24

x3 =        down (x = x2, num = 512) #24

x4 =            layers.MaxPooling2D((2, 2))(x3) #12

x5 =            basic (x = x4, num = 256) #12

x6 =        layers.UpSampling2D((2, 2))(x5) #24

x3_alt =    basic (x = x3, num = 256) #24

x6 =        layers.Concatenate() ([x3_alt, x6]) #24 (512)

x7 =        basic (x = x6, num = 256) #24

x8 =    layers.UpSampling2D((2, 2))(x7) #48

x9 =    layers.Concatenate() ([x1, x8]) #48 (512)

x10 =   basic (x = x9, num = 256)

x11 = layers.UpSampling2D((2, 2))(x10) #96

x12 = basic (x = x11, num = 128)

x13 = basic (x = x12, num = 64)

x14 = tf.pad(x13, paddings, "SYMMETRIC") #98
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid')(x14) #96

#model done
model = keras.Model(Input_img, decoded)