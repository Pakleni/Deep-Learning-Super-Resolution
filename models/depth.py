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


#input layer
Input_img = keras.Input(shape=(48, 48, 3)) #48

x = basic (x= Input_img, num= 256)
x = basic (x= x, num= 256)

x = tf.pad(x, paddings, "SYMMETRIC")
x = layers.Conv2D(3 * (2 ** 2), (3,3), activation='sigmoid')(x)

decoded = tf.nn.depth_to_space(x12, 2, data_format='NHWC', name=None)

#model done
model = keras.Model(Input_img, decoded)