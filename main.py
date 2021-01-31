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
tf.config.experimental_run_functions_eagerly(True)

tf.config.run_functions_eagerly(True)


os.system('clear')

valid_loader = DIV2K(type='valid')

valid_ds = valid_loader.dataset(batch_size=1, random_transform=True)

from losses import *


create = False
rerun = True

batch_size = 10
#3e-4
n = 1e-5
epochs = 400
num = 3200
valid_num = 100


optimizer = keras.optimizers.Adam(learning_rate=n)
loss_fn = SRGANVGGLoss
metrics = ['accuracy',
            SRGAN,
            SSIMLoss]

patience = 1
monitor = 'val_loss'
early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, monitor=monitor, restore_best_weights=True)

#LOAD DATA
train_loader = DIV2K(type='train')

train_ds = train_loader.dataset(batch_size=1, random_transform=True)

#END LOAD

if(create):

    
    from models.basic import model
    # from models.inception import model
    # from models.unet import model
    # from models.depth import model
    # from models.resnet import model

    # print(model.summary())
    # exit(1)

    model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics)
else:
    model = tf.keras.models.load_model('./saved-models/model.h5', 
                                        custom_objects=custom_objects)
    model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics)




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
    the_lr_tr.extend([norm(x[0][0])])
    the_hr_tr.extend([norm(x[1][0])])

#we don't need this anymore
del(train_ds)

c = 0
for x in valid_ds.take(valid_num):
    os.system('clear')
    c+=1
    print(f"{c}/{valid_num}")
    the_lr_vl.extend([norm(x[0][0])])
    the_hr_vl.extend([norm(x[1][0])])


the_lr_tr = np.array(the_lr_tr)
the_hr_tr = np.array(the_hr_tr)

the_lr_vl = np.array(the_lr_vl)
the_hr_vl = np.array(the_hr_vl)

#END PREPARE


#TRAIN


history = model.fit(x = the_lr_tr,
                    y = the_hr_tr,
                    batch_size=batch_size,
                    epochs=epochs, 
                    validation_data=(the_lr_vl, the_hr_vl),
                    callbacks=[early_stopping])

#SAVE THE NN
model.save("./saved-models/model.h5")

if (rerun):
    sys.exit()

# PLOT DATA
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')


plt.show()