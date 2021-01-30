import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from data import DIV2K
from tensorflow import keras 
from tensorflow.keras import layers, models
from keras import backend as K

#config stuff
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#end config stuff


#UNCOMMENT THESE TWO IF USING VGG LOSS CAUSES PROBLEMS
# tf.config.experimental_run_functions_eagerly(True)

# tf.config.run_functions_eagerly(True)



from losses import *



os.system('clear')

def norm(x):
    return (x/255)

def denorm(x):
    return (x*255).astype("int32")




create = False
rerun = True
patience = 3
batch_size = 20
n = 0.00001
epochs = 400
num = 3200
optimizer = keras.optimizers.Adam(learning_rate=n)
loss_fn = keras.losses.MeanSquaredError()


#LOAD DATA
valid_loader = DIV2K(type='valid')

valid_ds = valid_loader.dataset(batch_size=1, repeat_count=1, random_transform=True)

train_loader = DIV2K(type='train')

train_ds = train_loader.dataset(batch_size=1, random_transform=True)


main = tf.keras.models.load_model('./saved-models/model.h5',
                                    custom_objects=custom_objects)

if(create):
    from models.srgan import model

    model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy'])
else:
    model = tf.keras.models.load_model('./saved-models/srgan.h5')
    model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy'])




#PREPARE DATA

the_lr_tr = []
the_hr_tr = []

the_lr_vl = []
the_hr_vl = []

c = 0
for x in train_ds.take(num):
    os.system('clear')
    c+=1
    print(f"{c}/{num}")
    #FAKE
    fake = main(np.array( [ norm(x[0][0]) ] ) )
    the_lr_tr.extend(fake)
    the_hr_tr.extend([1])
    #REAL
    the_lr_tr.extend([norm(x[1][0])])
    the_hr_tr.extend([0])

#we don't need this anymore
del(train_ds)

c = 0
for x in valid_ds.take(100):
    os.system('clear')
    c+=1
    print(f"{c}/100")
    #FAKE
    fake = main(np.array( [ norm(x[0][0]) ] ) )
    the_lr_vl.extend(fake)
    the_hr_vl.extend([1])
    #REAL
    the_lr_vl.extend([norm(x[1][0])])
    the_hr_vl.extend([0])


del(main)

the_lr_tr = np.array(the_lr_tr)
the_hr_tr = np.array(the_hr_tr)

the_lr_vl = np.array(the_lr_vl)
the_hr_vl = np.array(the_hr_vl)

#END PREPARE


#TRAIN

early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)

history = model.fit(x = the_lr_tr,
                    y = the_hr_tr,
                    batch_size=batch_size,
                    epochs=epochs, 
                    validation_data=(the_lr_vl, the_hr_vl),
                    callbacks=[early_stopping])

#SAVE THE NN
model.save("./saved-models/srgan.h5")

if (rerun):
    sys.exit()

# PLOT DATA
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')


plt.show()