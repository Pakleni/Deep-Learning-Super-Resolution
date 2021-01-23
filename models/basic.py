import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models

model = models.Sequential()

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 3)))

model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(128, (3, 3),activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(8, (3, 3),  activation='relu', padding='same'))


model.add(layers.Conv2D(3 , (3, 3), activation='sigmoid', padding='same'))