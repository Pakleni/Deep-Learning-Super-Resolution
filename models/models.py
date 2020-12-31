model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))

model.add(layers.Conv2D(3, (3, 3), activation=None, padding='same'))






model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3), dtype='float32'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Conv2D(64, (3, 3)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.UpSampling2D(size=(2, 2)))

model.add(layers.Conv2D(3, (3, 3), activation=None, padding='same'))









model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation=None, input_shape=(48, 48, 3)))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=None))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None))
model.add(layers.LeakyReLU(alpha=0.3))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=None))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None))
model.add(layers.LeakyReLU(alpha=0.3))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(layers.LeakyReLU(alpha=0.3))

model.add(layers.UpSampling2D(size=(2, 2)))

model.add(layers.Conv2D(3, (3, 3), activation=None, padding='same'))





model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))

model.add(layers.Conv2D(3 , (3, 3), activation=None, padding='same'))





model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.LayerNormalization())

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.LayerNormalization())

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.LayerNormalization())

# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu'))
model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu'))
model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu'))
model.add(layers.LayerNormalization())

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu'))
model.add(layers.LayerNormalization())

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu'))
model.add(layers.LayerNormalization())

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu'))
model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu'))
model.add(layers.LayerNormalization())

model.add(layers.Conv2DTranspose(3 , (3, 3), activation=None, padding='same'))




model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Conv2D(1024, (3, 3), activation='relu'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))

model.add(layers.Conv2D(3 , (3, 3), activation=None, padding='same'))




model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Conv2D(1024, (3, 3), activation='relu'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))

model.add(layers.UpSampling2D(size=(2, 2)))

model.add(layers.Conv2D(3 , (3, 3), activation=None, padding='same'))






model = models.Sequential()
        
model.add(layers.Conv2D(128, (9, 9), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#####

model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu'))

model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu'))
model.add(layers.UpSampling2D(size=(2, 2)))

model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu'))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu'))

model.add(layers.Conv2DTranspose(3, (3, 3),
                activation='linear', padding='valid'))





x1 = layers.Conv2D(64, (9, 9), activation='relu')(Input_img) # 40 40 128
x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1) # 40 40 64

x3 = layers.Conv2D(128, (5, 5), activation='relu')(x2) # 36 36 128

x4 = layers.MaxPooling2D((2, 2))(x3) # 18 18 128
x5 = layers.Conv2D(256, (3, 3), activation='relu')(x4) # 16 16 256
x6 = layers.Conv2D(512, (7, 7), activation='relu')(x5) # 10 10 512

x7 = layers.UpSampling2D(size=(2, 2))(x6) # 20 20 512
x8 = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(x7) # 20 20 64
x9 = layers.UpSampling2D(size=(2, 2))(x8) # 40 40 64

x10 = layers.Add() ([x2,x9]) # 40 40 64
x11 = layers.Conv2DTranspose(64, (9, 9), activation='relu')(x10) # 48 48 64

x12 = layers.UpSampling2D(size=(2, 2))(x11) # 96 96 64

decoded = layers.Conv2D(3, (3, 3), padding='same',activation='relu')(x12)



Input_img = keras.Input(shape=(48, 48, 3))


x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(Input_img)
x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
x3 = layers.MaxPooling2D((2, 2))(x2)


x4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
x5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x4)
x6 = layers.MaxPooling2D((2, 2))(x5)


x7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x6)
x8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x7)
x9 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x8)
x10 = layers.UpSampling2D(size=(2, 2))(x9)


x11 = layers.Add() ([x5,x10])

x12 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x11)
x13 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x12)
x14 = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x13)


x16 = layers.Add() ([x3,x14])

x17 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x16)
x18 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x17)
x19 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x18)
x20 = layers.UpSampling2D(size=(2, 2))(x19)


x21 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x20)
x22 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x21)
x23 = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(x22)
x24 = layers.UpSampling2D(size=(2, 2))(x23)

x25 = layers.Conv2DTranspose(16, (3, 3), activation='relu' )(x24)
decoded = layers.Conv2D(3, (3, 3), activation=tf.keras.activations.sigmoid)(x25)

model = keras.Model(Input_img, decoded)








Input_img = keras.Input(shape=(48, 48, 3)) #48

        
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

x0 = tf.pad(Input_img, paddings, "SYMMETRIC")
x1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(x0) #48

x1 = tf.pad(x1_1, paddings, "SYMMETRIC")
x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(x1) #48

x3_1 = layers.MaxPooling2D((2, 2))(x2) #24

x3 = tf.pad(x3_1, paddings, "SYMMETRIC")
x4 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x3) #24

x4 = tf.pad(x4, paddings, "SYMMETRIC")
x5 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(x4) #24


x6 = layers.MaxPooling2D((2, 2))(x5) #12

x6 = tf.pad(x6, paddings, "SYMMETRIC")
x7 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(x6) #12

x7 = tf.pad(x7, paddings, "SYMMETRIC")
x8 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(x7) #12
x10 = tf.nn.depth_to_space(x8, 2, data_format='NHWC', name=None)
#x10 = layers.UpSampling2D(size=(2, 2))(x9) #24


x10 = tf.pad(x10, paddings, "SYMMETRIC")
x11 = layers.Concatenate() ([x3,x10]) #24

# x12 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x11) #24
# x13 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x12) #24
# x14 = layers.Conv2D(64, (1, 1), activation='relu', padding='valid')(x13) #24
# x15 = MaxPooling2D()(x14)

# x16 = layers.Concatenate() ([x3,x14]) #24

x17 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x11) #24

x17 = tf.pad(x17, paddings, "SYMMETRIC")
x18 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(x17) #24
x18 = tf.nn.depth_to_space(x18, 2, data_format='NHWC', name=None)


x3_2 = tf.nn.depth_to_space(x3_1, 2, data_format='NHWC', name=None)
x19p = layers.Concatenate() ([x3_2,x18]) #24

# x19 = tf.pad(x19, paddings, "SYMMETRIC")


x20 = tf.pad(x19p, paddings, "SYMMETRIC")
x21 = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x20) #48

x21 = tf.pad(x21, paddings, "SYMMETRIC")
x22 = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x21) #48


x23p = layers.Concatenate()([x1_1,x22]) #48

x24 = tf.nn.depth_to_space(x23p, 2, data_format='NHWC', name=None) #96


x24 = tf.pad(x24, paddings, "SYMMETRIC") #98    
x25 = layers.Conv2D(24, (3, 3), activation='relu', padding='valid')(x24) #96

paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
x26 = tf.pad(x25, paddings, "SYMMETRIC") #98    
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid')(x26) #96

model = keras.Model(Input_img, decoded[0:,1:-1,1:-1])





Input_img = keras.Input(shape=(48, 48, 3)) #48


x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(Input_img) #48
x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1) #48

x3 = layers.MaxPooling2D((2, 2))(x2) #24


x4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x3) #24
x5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x4) #24
x6 = layers.MaxPooling2D((2, 2))(x5) #12


x7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x6) #12
x8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x7) #12
x9 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x8) #12
x10 = layers.UpSampling2D(size=(2, 2))(x9) #24


x11 = layers.Concatenate() ([x3,x10]) #24

x12 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x11) #24
x13 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x12) #24
x14 = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x13) #24


x16 = layers.Concatenate() ([x3,x14]) #24

x17 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x16) #24
x18 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x17) #24
x19 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x18) #24

x19p = layers.Concatenate() ([x3,x19]) #24

x20 = layers.UpSampling2D(size=(2, 2))(x19p) #48


x21 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x20) #48
x22 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x21) #48
x23 = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(x22) #48

x23p = layers.Concatenate() ([Input_img,x23]) #48

# x23pp = layers.DepthwiseConv2D(1, (1, 1), activation='relu')(x23p)

x24 = layers.UpSampling2D(size=(2, 2))(x23p) #96


decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x24) #96

model = keras.Model(Input_img, decoded)