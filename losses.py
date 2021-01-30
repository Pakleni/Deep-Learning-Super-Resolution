from tensorflow import keras 
from tensorflow.keras import layers, models
from keras import backend as K
import tensorflow as tf

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

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

srgan = tf.keras.models.load_model('./saved-models/srgan.h5')
def SRGANLoss(y_true,y_pred):
    return tf.clip_by_value(t = srgan(y_pred)
                            , clip_value_min = 0
                            , clip_value_max = 1) * 0.01 + SSIMLoss(y_true,y_pred)

def SRGANVGGLoss(y_true,y_pred):
    return tf.clip_by_value(t = srgan(y_pred)
                            , clip_value_min = 0
                            , clip_value_max = 1) + vggLoss(y_true,y_pred)


custom_objects = {'SSIMLoss': SSIMLoss,
                'VGGStyleLoss': VGGStyleLoss,
                'vggLoss': vggLoss,
                'psnr': psnr,
                'SRGANLoss': SRGANLoss,
                'SRGANVGGLoss': SRGANVGGLoss}