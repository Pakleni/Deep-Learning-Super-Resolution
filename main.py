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

#vgg stuff
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

#UNCOMMENT THESE TWO IF USING VGG LOSS CAUSES PROBLEMS
tf.config.experimental_run_functions_eagerly(True)

tf.config.run_functions_eagerly(True)

def vggLoss(X,Y):
    vgg_model = VGG19(include_top=False)

    Xt = preprocess_input(X*255)
    Yt = preprocess_input(Y*255)
    
    vggX = vgg_model(Xt)
    vggY = vgg_model(Yt)

    return tf.reduce_mean(tf.square(vggY-vggX))

def VGGFeatureLoss(X,Y):
    vgg_model = VGG19(include_top=False)

    vgg_model = vgg_model.get_layer(name= 'block1_conv2')

    Xt = preprocess_input(X*255)
    Yt = preprocess_input(Y*255)

    vggX = vgg_model(Xt)
    vggY = vgg_model(Yt)
    
    return tf.reduce_mean(tf.square(vggY-vggX))
#end vgg stuff

def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, norm(255)))

def psnr(y_true,y_pred):
    max_pixel = norm(255)
    return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true + 1e-8), axis=-1)))) / 2.303

def psnr_abs(y_true,y_pred):
    max_pixel = norm(255)
    return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.abs(y_pred - y_true + 1e-8), axis=-1)))) / 2.303




os.system('clear')

valid_loader = DIV2K(type='valid')

valid_ds = valid_loader.dataset(batch_size=1, repeat_count=1, random_transform=True)


def norm(x):
    return (x/255)

def denorm(x):
    return (x*255).astype("int32")




train = True
create = True

patience = 10
batch_size = 20
n = 0.0003
epochs = 400
num = 3600
optimizer = keras.optimizers.Adam(learning_rate=n)
loss_fn = VGGFeatureLoss




if (train):

    #LOAD DATA
    train_loader = DIV2K(type='train')

    train_ds = train_loader.dataset(batch_size=1, random_transform=True)

    #END LOAD

    if(create):

        
        # from models.basic import model
        # from models.inception import model
        # from models.inception_no7 import model
        from models.unet import model
        # from models.depth import model
        # from models.resnet import model

        # print(model.summary())
        # exit(1)

        model.compile(optimizer=optimizer,
                    loss=loss_fn,
                    metrics=['accuracy'])
    else:
        model = tf.keras.models.load_model('./saved-models/model.h5', custom_objects={'SSIMLoss': SSIMLoss, 'vggLoss': vggLoss, 'psnr': psnr})
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
        the_lr_tr.extend([norm(x[0][0])])
        the_hr_tr.extend([norm(x[1][0])])

    #we don't need this anymore
    del(train_ds)

    c = 0
    for x in valid_ds.take(100):
        os.system('clear')
        c+=1
        print(f"{c}/100")
        the_lr_vl.extend([norm(x[0][0])])
        the_hr_vl.extend([norm(x[1][0])])


    the_lr_tr = np.array(the_lr_tr)
    the_hr_tr = np.array(the_hr_tr)

    the_lr_vl = np.array(the_lr_vl)
    the_hr_vl = np.array(the_hr_vl)
    
    #END PREPARE


    #TRAIN DA BITCH

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)

    history = model.fit(x = the_lr_tr,
                        y = the_hr_tr,
                        batch_size=batch_size,
                        epochs=epochs, 
                        validation_data=(the_lr_vl, the_hr_vl),
                        callbacks=[early_stopping])

    #SAVE THE NN
    model.save("./saved-models/model.h5")

    sys.exit()

    # PLOT DATA
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(top=10)
    plt.legend(loc='lower right')


    plt.show()


else:
    model = tf.keras.models.load_model('./saved-models/model.h5', custom_objects={'SSIMLoss': SSIMLoss,'vggLoss': vggLoss, 'psnr': psnr})





valid_ds = valid_loader.dataset(batch_size=1, repeat_count=1, random_transform=False)


lr_crop_size = 48

for lr_img, hr_img in valid_ds.take(40):

    # lr_img = hr_img

    #OVDE SE SMANJUJE SLIKA ULAZNA
    lr_img_shape = tf.shape(lr_img[0])[:2]
    lr_img = lr_img.numpy()

    lr_img_temp = lr_img[0][0:240, 0:240]
    lr_img = [lr_img_temp]
    #KRAJ

    out = tf.zeros((480,480, 3)).numpy().astype('int32')
    
    #dimenzije ulaza
    lr_img_shape = tf.shape(lr_img[0])[:2]

    width = int(lr_img_shape[1].numpy())
    height = int(lr_img_shape[0].numpy())

    factorStride = 1
    stride = lr_crop_size // factorStride
    # stride = 48
    
    for i in range(0, width//stride - factorStride + 1):
        lr_w = stride*i

        for j in range(0, height//stride - factorStride + 1):
            lr_h = stride*j
            
            #get the cropped img
            lr_img_cropped = lr_img[0][lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
            
            lr_img_cropped = norm(lr_img_cropped[None])

            #now predict
            predictions = model.predict(lr_img_cropped)
            x=[denorm(y) for y in predictions[0]]

            #add predicted to out
            out[lr_h*2:(lr_h + lr_crop_size)*2, lr_w*2:(lr_w + lr_crop_size)*2] += x

    for i in range(0, width//stride):


        for j in range(0, height//stride):

            nesto = 1


            nesto *= min(min(i + 1,width//stride - i), factorStride)

            nesto *= min(min(j + 1,height//stride - j), factorStride)

            out[j * stride * 2 :(j+1) * stride * 2, i * stride * 2 :(i+1) *  stride * 2] //= nesto
        
    
    
    #print
    plt.imshow(lr_img[0], cmap=plt.cm.binary)
    plt.show()

    plt.imshow(out, cmap=plt.cm.binary)
    plt.show()

    plt.imshow(hr_img[0,0:480,0:480], cmap=plt.cm.binary)
    plt.show()