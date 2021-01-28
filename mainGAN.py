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
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

#UNCOMMENT THESE TWO IF USING VGG LOSS CAUSES PROBLEMS
# tf.config.experimental_run_functions_eagerly(True)

# tf.config.run_functions_eagerly(True)


def vggLoss(X,Y):
    vgg_model = VGG19(include_top=False, input_shape=(96,96,3))

    Xt = preprocess_input(X*255)
    Yt = preprocess_input(Y*255)
    
    vggX = vgg_model(Xt)
    vggY = vgg_model(Yt)

    return tf.reduce_mean(tf.square(vggY-vggX))

def VGGStyleLoss(X,Y):
    Xt = preprocess_input(X*255)
    Yt = preprocess_input(Y*255)
    
    vgg_model = VGG19(include_top=False, input_shape=(96,96,3))

    Xx = vgg_model.input
    
    layerNames = [ [1,2], [2,2], [3,4], [4,4], [5,4] ]

    ret = 0;
    for i in layerNames:

        Yy = vgg_model.get_layer(name= f'block{i[0]}_conv{i[1]}').output

        curr = keras.models.Model(Xx,Yy)


        vggX = curr(Xt)
        vggY = curr(Yt)
        
        ret +=  tf.reduce_mean(tf.square(vggY-vggX))/(curr.output_shape[1]*curr.output_shape[2]*curr.output_shape[3])

    return ret

def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, norm(255)))

def psnr(y_true,y_pred):
    max_pixel = norm(255)
    return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true + 1e-8), axis=-1)))) / 2.303

def psnr_abs(y_true,y_pred):
    max_pixel = norm(255)
    return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.abs(y_pred - y_true + 1e-8), axis=-1)))) / 2.303




os.system('clear')

def norm(x):
    return (x/255)

def denorm(x):
    return (x*255).astype("int32")




create = True
rerun = False
patience = 10
batch_size = 20
factorStride = 1
n = 0.00003
epochs = 400
num = 3200
optimizer = keras.optimizers.Adam(learning_rate=n)
loss_fn = keras.losses.MeanSquaredError()


#LOAD DATA
valid_loader = DIV2K(type='valid')

valid_ds = valid_loader.dataset(batch_size=1, repeat_count=1, random_transform=True)

train_loader = DIV2K(type='train')

train_ds = train_loader.dataset(batch_size=1, random_transform=True)

#END LOAD
custom_objects = {'SSIMLoss': SSIMLoss,
                'VGGStyleLoss': VGGStyleLoss,
                'vggLoss': vggLoss,
                'psnr': psnr}

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