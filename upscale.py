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


import os
images_dir = os.path.join('upscale-in')
image_files = [os.path.join(images_dir, filename) for filename in sorted(os.listdir(images_dir))]

ds = tf.data.Dataset.from_tensor_slices(image_files)
ds = ds.map(tf.io.read_file)

from tensorflow.python.data.experimental import AUTOTUNE
ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

ds = ds.batch(1)


from PIL import Image
import numpy as np

i = 0

for img in ds.take(1):

    img_temp = norm(img)

    predictions = model.predict(img_temp)
    x=[denorm(y) for y in predictions[0]]

    im = Image.fromarray(np.uint8(x))

    im.save(os.path.join("upscale-out", f"converted{i}.png"), format="png")
    
    i += 1