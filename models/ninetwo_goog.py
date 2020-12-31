import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models


paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        
        
#input layer
Input_img = keras.Input(shape=(48, 48, 3)) #48

#pool-gang-1
x1 = Input_img
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
x1_1 = tf.pad(x1, paddings, "SYMMETRIC")
paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
x1_2 = tf.pad(x1, paddings, "SYMMETRIC") 
paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
x1_3 = tf.pad(x1, paddings, "SYMMETRIC")
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
x2_1 = layers.Conv2D(16, (1, 1), activation='relu') (x1) #48
x2_2 = layers.Conv2D(3, (3, 3), activation='relu') (x1_1) #48
x2_3 = layers.Conv2D(32, (5, 5), activation='relu') (x1_2) #48
x2_4 = layers.Conv2D(32, (7, 7), activation='relu') (x1_3) #48

x2 = layers.Concatenate() ([x2_1, x2_2, x2_3, x2_4])

x3 = tf.pad(x2, paddings, "SYMMETRIC") #50
x4 = layers.Conv2D(64, (3, 3), activation='relu') (x3) #48

x5 = layers.MaxPooling2D((2, 2))(x4) #24

#pool-gang-2
x6 = x5
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
x6_1 = tf.pad(x6, paddings, "SYMMETRIC")
paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
x6_2 = tf.pad(x6, paddings, "SYMMETRIC") 
paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
x6_3 = tf.pad(x6, paddings, "SYMMETRIC")
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
x7_1 = layers.Conv2D(32, (1, 1), activation='relu') (x6) #24
x7_2 = layers.Conv2D(32, (3, 3), activation='relu') (x6_1) #24
x7_3 = layers.Conv2D(32, (5, 5), activation='relu') (x6_2) #24
x7_4 = layers.Conv2D(32, (7, 7), activation='relu') (x6_3) #24

x7 = layers.Concatenate() ([x7_1, x7_2, x7_3, x7_4])

x8 = tf.pad(x7, paddings, "SYMMETRIC") #26
x9 = layers.Conv2D(128, (3, 3), activation='relu') (x8) #24

# x10 = layers.MaxPooling2D((2, 2))(x9) #12

# uppers-gang-1
# x11 = x10

# x12_1 = layers.Conv2D(64, (1, 1), activation='relu') (x11) #12
# x12_2 = layers.Conv2D(64, (3, 3), activation='relu') (x11) #12
# x12_3 = layers.Conv2D(64, (5, 5), activation='relu') (x11) #12
# x12_4 = layers.Conv2D(64, (7, 7), activation='relu') (x11) #12

# x12 = layers.Concatenate() ([x12_1, x12_2, x12_3, x12_4])

# x13 = tf.pad(x12, paddings, "SYMMETRIC") #14
# x14 = layers.Conv2D(256, (3, 3), activation='relu') (x13) #12
# x15 = layers.Conv2D(128, (1, 1), activation='relu') (x14) #12

# x16 = layers.UpSampling2D(size=(2, 2))(x15) #24

# #con to the cat
# x17 = layers.Concatenate() ([x6, x16]) #24

# x18 = tf.pad(x17, paddings, "SYMMETRIC") #26
x18 = x9

#uppers-gang-2 + con to the cat2 -> ???
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
x18_1 = tf.pad(x18, paddings, "SYMMETRIC")
paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
x18_2 = tf.pad(x18, paddings, "SYMMETRIC") 
paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
x18_3 = tf.pad(x18, paddings, "SYMMETRIC")
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
x19_1 = layers.Conv2D(32, (1, 1), activation='relu') (x18) #24
x19_2 = layers.Conv2D(32, (3, 3), activation='relu') (x18_1) #24
x19_3 = layers.Conv2D(32, (5, 5), activation='relu') (x18_2) #24
x19_4 = layers.Conv2D(32, (7, 7), activation='relu') (x18_3) #24

x19 = layers.Concatenate() ([x19_1, x19_2, x19_3, x19_4])

x20 = tf.pad(x19, paddings, "SYMMETRIC") #26
x21 = layers.Conv2D(64, (3, 3), activation='relu') (x20) #24
x22 = layers.Conv2D(32, (1, 1), activation='relu') (x21) #24

x23 = layers.Concatenate() ([x5, x22]) #24
x24 = layers.UpSampling2D(size=(2, 2))(x23) #48

#uppers-gang-1
x25 = x24

paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
x25_1 = tf.pad(x25, paddings, "SYMMETRIC")
paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
x25_2 = tf.pad(x25, paddings, "SYMMETRIC") 
paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
x25_3 = tf.pad(x25, paddings, "SYMMETRIC")
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

x26_1 = layers.Conv2D(8, (1, 1), activation='relu') (x25) #48
x26_2 = layers.Conv2D(8, (3, 3), activation='relu') (x25_1) #48
x26_3 = layers.Conv2D(8, (5, 5), activation='relu') (x25_2) #48
x26_4 = layers.Conv2D(8, (7, 7), activation='relu') (x25_3) #48

x26 = layers.Concatenate() ([x26_1, x26_2, x26_3, x26_4])

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