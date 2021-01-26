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


def inception(x, num):
    x1 = x
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    x1_1 = tf.pad(x1, paddings, "SYMMETRIC")
    x1_1 = layers.Conv2D(num/2, (1, 1), activation='relu') (x1_1)

    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    x1_2 = tf.pad(x1, paddings, "SYMMETRIC") 
    x1_2 = layers.Conv2D(num/2, (1, 1), activation='relu') (x1_2)

    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    x2_2 = layers.Conv2D(num/2, (3, 3), activation='relu') (x1_1)
    x2_3 = layers.Conv2D(num/2, (5, 5), activation='relu') (x1_2)

    x2 = layers.Concatenate() ([ x2_2, x2_3])

    return x2


#input layer
Input_img = keras.Input(shape=(48, 48, 3)) #48

x =     basic (x= Input_img, num= 32) #48 (32)

x1 =    inception (x= x, num= 32) #48 (32)

x2 =        layers.MaxPooling2D((2, 2))(x1) #24 (32)

x3 =        inception (x= x2, num= 32) #24 (32)

x4 =        layers.Concatenate() ([x2, x3]) #24 (32 + 32)

x5 =    layers.UpSampling2D(size=(2, 2))(x4) #48 (64)

x6 =    layers.Conv2D(32, (1, 1), activation='relu') (x5) #48 (32)

x7 =    inception (x= x6, num= 32) #48 (32)

x8 =    layers.Concatenate() ([x, x5, x7]) #48 (32 + 64 + 32)

x9 =    layers.Conv2D(64, (1, 1), activation='relu') (x8) #48 (64)

x10 = layers.UpSampling2D(size=(2, 2))(x9) #96 (64)

x11 = basic (x= x10, num= 32) #96 (32)

x12 = tf.pad(x11, paddings, "SYMMETRIC") #98

decoded = layers.Conv2D(3, (3, 3), activation='sigmoid')(x12) #96


#model done
model = keras.Model(Input_img, decoded)