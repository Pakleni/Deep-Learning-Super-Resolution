import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models


paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        
        
#input layer
Input_img = keras.Input(shape=(48, 48, 3)) #48

#pool-gang-1
x1 = tf.pad(Input_img, paddings, "SYMMETRIC") #50
x2 = layers.Conv2D(32, (3, 3), activation='relu') (x1) #48

x3 = tf.pad(x2, paddings, "SYMMETRIC") #50
x4 = layers.Conv2D(64, (3, 3), activation='relu') (x3) #48

x5 = layers.MaxPooling2D((2, 2))(x4) #24

#pool-gang-2
x6 = tf.pad(x5, paddings, "SYMMETRIC") #26
x7 = layers.Conv2D(128, (3, 3), activation='relu') (x6) #24

x8 = tf.pad(x7, paddings, "SYMMETRIC") #26
x9 = layers.Conv2D(128, (3, 3), activation='relu') (x8) #24

x10 = layers.MaxPooling2D((2, 2))(x9) #12

#uppers-gang-1
x11 = tf.pad(x10, paddings, "SYMMETRIC") #1
x12 = layers.Conv2D(256, (3, 3), activation='relu') (x11) #12

x13 = tf.pad(x12, paddings, "SYMMETRIC") #14
x14 = layers.Conv2D(256, (3, 3), activation='relu') (x13) #12
x15 = layers.Conv2D(128, (1, 1), activation='relu') (x14) #12

x16 = layers.UpSampling2D(size=(2, 2))(x15) #24

#con to the cat
x17 = tf.pad(x16, paddings, "SYMMETRIC") #26
x18 = layers.Concatenate() ([x6, x17]) #26

#uppers-gang-2 + con to the cat2 -> ???
x19 = layers.Conv2D(64, (3, 3), activation='relu') (x18) #24

x20 = tf.pad(x19, paddings, "SYMMETRIC") #26
x21 = layers.Conv2D(64, (3, 3), activation='relu') (x20) #24
x22 = layers.Conv2D(32, (1, 1), activation='relu') (x21) #24

x23 = layers.Concatenate() ([x5, x22]) #24
x24 = layers.UpSampling2D(size=(2, 2))(x23) #48

#uppers-gang-1
x25 = tf.pad(x24, paddings, "SYMMETRIC") #50
x26 = layers.Conv2D(32, (3, 3), activation='relu') (x25) #48

x27 = tf.pad(x26, paddings, "SYMMETRIC") #50
x28 = layers.Conv2D(32, (3, 3), activation='relu') (x27) #48
x29 = layers.Conv2D(16, (1, 1), activation='relu') (x28) #48

x30 = layers.Concatenate() ([Input_img, x29]) #48
x31 = layers.UpSampling2D(size=(2, 2))(x30) #96

#output-gangaroos
x32 = tf.pad(x31, paddings, "SYMMETRIC") #98
x33 = layers.Conv2D(16, (3, 3), activation='relu') (x32) #96

x34 = tf.pad(x33, paddings, "SYMMETRIC") #98
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid')(x34) #96


#model done
model = keras.Model(Input_img, decoded)