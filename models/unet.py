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

x1 =    down(x = Input_img, num = 16)

x2 =        layers.MaxPooling2D((2, 2))(x1) #24

x2 =        down (x = x2, num = 32)

x3 =            layers.MaxPooling2D((2, 2))(x2) #12

x3 =            down (x = x3, num = 64)

x4 =                layers.MaxPooling2D((2, 2))(x3) #6

x4 =                down (x = x4, num = 128)

x5 =                    layers.MaxPooling2D((2, 2))(x4) #3

x5 =                    down (x = x5, num = 256)

x6 =                layers.Conv2DTranspose(128, (2,2),strides=(2,2), activation = 'relu')(x5) #6 (128)

x6 =                layers.Concatenate() ([x4, x6]) #(256)

x6 =                basic (x = x6, num = 128)

x7 =            layers.Conv2DTranspose(64, (2,2),strides=(2,2), activation = 'relu')(x6) #12

x7 =            layers.Concatenate() ([x3, x7])

x7 =            basic (x = x7, num = 64)

x8 =        layers.Conv2DTranspose(32, (2,2),strides=(2,2), activation = 'relu')(x7) #24

x8 =        layers.Concatenate() ([x2, x8])

x8 =        basic (x = x8, num = 32)

x9 =    layers.Conv2DTranspose(16, (2,2),strides=(2,2), activation = 'relu')(x8) #48

x9 =    layers.Concatenate() ([x1, x9]) #48

x9 =    basic (x = x9, num = 16)

x10 = layers.Conv2DTranspose(8, (2,2),strides=(2,2), activation = 'relu')(x9) #96

x10 = basic (x = x10, num = 8)


x11 = tf.pad(x10, paddings, "SYMMETRIC") #98
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid')(x11) #96

#model done
model = keras.Model(Input_img, decoded)