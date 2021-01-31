from data import DIV2K
import matplotlib.pyplot as plt

#config stuff
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#end config stuff


from losses import *



model = tf.keras.models.load_model('./saved-models/model.h5',
                                    custom_objects=custom_objects)

model._layers[0]._batch_input_shape = (None,None,None,3)

new_model = keras.models.model_from_json(model.to_json())

for layer,new_layer in zip(model.layers,new_model.layers):
    new_layer.set_weights(layer.get_weights())

model= new_model


valid_loader = DIV2K(type='valid')

valid_ds = valid_loader.dataset(batch_size=1, repeat_count=1, random_transform=False)


size = 240

for lr_img, hr_img in valid_ds.take(40):

    # lr_img = hr_img

    #OVDE SE SMANJUJE SLIKA ULAZNA
    lr_img_shape = tf.shape(lr_img[0])[:2]
    lr_img = lr_img.numpy()

    lr_img_temp = lr_img[0][0:size, 0:size]
    lr_img = [lr_img_temp]
    #KRAJ

    lr_img_temp = norm(lr_img[0][None])

    predictions = model.predict(lr_img_temp)
    x=[denorm(y) for y in predictions[0]]


        
    
    #print
    plt.imshow(lr_img[0], cmap=plt.cm.binary)
    plt.show()

    plt.imshow(x, cmap=plt.cm.binary)
    plt.show()

    plt.imshow(hr_img[0,0:2*size,0:2*size], cmap=plt.cm.binary)
    plt.show()