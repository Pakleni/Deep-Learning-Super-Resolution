import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models


paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])


Input_img = keras.Input(shape=(48, 48, 3)) #48

def basic(x, num):
    x = tf.pad(x, paddings, "SYMMETRIC") #98
    x = layers.Conv2D(num/2, (3, 3), activation='relu') (x) #96

    x = tf.pad(x, paddings, "SYMMETRIC") #98
    x = layers.Conv2D(num/2, (3, 3), activation='relu') (x) #96

    x = tf.pad(x, paddings, "SYMMETRIC") #98
    x = layers.Conv2D(num/2, (3, 3), activation='relu') (x) #96

    x = tf.pad(x, paddings, "SYMMETRIC") #98
    x = layers.Conv2D(num, (3, 3), activation='relu') (x) #96

    return x


x = basic(x= Input_img, num= 64)

x = basic(x= x, num= 32)

x = layers.UpSampling2D(size=(2, 2))(x) #96

x = basic(x= x, num= 16)


x = tf.pad(x, paddings, "SYMMETRIC") #98
x = layers.Conv2D(16, (3, 3), activation='relu') (x) #96

x = tf.pad(x, paddings, "SYMMETRIC") #98
x = layers.Conv2D(16, (3, 3), activation='relu') (x) #96

x = tf.pad(x, paddings, "SYMMETRIC") #98
x = layers.Conv2D(16, (3, 3), activation='relu') (x) #96



x = tf.pad(x, paddings, "SYMMETRIC") #98
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid')(x) #96


#model done
model = keras.Model(Input_img, decoded)