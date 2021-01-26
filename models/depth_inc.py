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
    x1_1 = layers.Conv2D(num/4, (1, 1), activation='relu') (x1_1) #48

    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    x1_2 = tf.pad(x1, paddings, "SYMMETRIC") 
    x1_2 = layers.Conv2D(num/4, (1, 1), activation='relu') (x1_2) #48

    paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
    x1_3 = tf.pad(x1, paddings, "SYMMETRIC")
    x1_3 = layers.Conv2D(num/4, (1, 1), activation='relu') (x1_3) #48

    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    x2_1 = layers.Conv2D(num/4, (1, 1), activation='relu') (x1) #48
    x2_2 = layers.Conv2D(num/4, (3, 3), activation='relu') (x1_1) #48
    x2_3 = layers.Conv2D(num/4, (5, 5), activation='relu') (x1_2) #48
    x2_4 = layers.Conv2D(num/4, (7, 7), activation='relu') (x1_3) #48

    x2 = layers.Concatenate() ([x2_1, x2_2, x2_3, x2_4]) #48 (128)

    return x2


#input layer
Input_img = keras.Input(shape=(48, 48, 3)) #48

x =     basic (x= Input_img, num= 128) #48 (128)

x1 =    inception (x= x, num= 128) #48 (128)

x2 =        layers.MaxPooling2D((2, 2))(x1) #24 (128)

x3 =        inception (x= x2, num= 128) #24 (128)

x4 =        layers.Concatenate() ([x2, x3]) #24 (128 + 128)

x5 =    layers.UpSampling2D(size=(2, 2))(x4) #48 (256)

x6 =    layers.Conv2D(128, (1, 1), activation='relu') (x5) #48 (128)

x7 =    inception (x= x6, num= 128) #48 (128)

x8 =    layers.Concatenate() ([x, x5, x7]) #48 (128 + 256 + 128)

x9 =    layers.Conv2D(256, (1, 1), activation='relu') (x8) #48 (256)

x10 =   basic (x= x9, num= 128)

x11 = tf.pad(x10, paddings, "SYMMETRIC")
x12 = layers.Conv2D(3 * (2 ** 2), (3,3), activation='sigmoid')(x11)

decoded = tf.nn.depth_to_space(x12, 2, data_format='NHWC', name=None)

#model done
model = keras.Model(Input_img, decoded)