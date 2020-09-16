from keras.models import load_model
import numpy as np
from skimage import color, exposure, transform

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img

from skimage import io
import os
import glob


def get_class(img_path):
    return int(img_path.split('/')[-2])


root_dir = 'GTSRB/Final_Training/Images/'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
np.random.shuffle(all_img_paths)
print(type(all_img_paths), len(all_img_paths))
img_path = all_img_paths[0]
img = preprocess_img(io.imread(img_path))
label = get_class(img_path)
imgs.append(img)
labels.append(label)

X = np.array(imgs, dtype='float32')
# Make one hot targets
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

print(X.shape)


from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
image = X[0].transpose(1,2,0)
imshow(image)
plt.show()





model = load_model('model.h5')
fc_weights = np.asarray(model.weights[-2].numpy())
x_0 = X[0:1]

import keras
for i in range(3, len(model.layers)):
	inter_output_model = keras.Model(model.input, model.get_layer(index = 3).output)
	inter_output = inter_output_model.predict(x_0)
	#get_output = K.function([model.layers[0].input], [model.layers[i].output])
	#output = get_output([x_0])[0]
	print(inter_output.shape)
# cam_weights = model.
import IPython; IPython.embed(); exit(1)

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_34 (Conv2D)           (None, 32, 48, 48)        896       
# _________________________________________________________________
# conv2d_35 (Conv2D)           (None, 32, 48, 48)        9248      
# _________________________________________________________________
# max_pooling2d_14 (MaxPooling (None, 32, 24, 24)        0         
# _________________________________________________________________
# dropout_20 (Dropout)         (None, 32, 24, 24)        0         
# _________________________________________________________________
# conv2d_36 (Conv2D)           (None, 64, 24, 24)        18496     
# _________________________________________________________________
# conv2d_37 (Conv2D)           (None, 64, 24, 24)        36928     
# _________________________________________________________________
# max_pooling2d_15 (MaxPooling (None, 64, 12, 12)        0         
# _________________________________________________________________
# dropout_21 (Dropout)         (None, 64, 12, 12)        0         
# _________________________________________________________________
# conv2d_38 (Conv2D)           (None, 128, 12, 12)       73856     
# _________________________________________________________________
# conv2d_39 (Conv2D)           (None, 128, 12, 12)       147584    
# _________________________________________________________________
# global_average_pooling2d_6 ( (None, 128)               0         
# _________________________________________________________________
# dense_12 (Dense)             (None, 43)                5547      
# =================================================================

# Conv
# (3, 3, 3, 32)
# (32,)

# Conv
# (3, 3, 32, 32)
# (32,)

# Conv
# (3, 3, 32, 64)
# (64,)

# Conv
# (3, 3, 64, 64)
# (64,)

# Conv
# (3, 3, 64, 128)
# (128,)

# Conv
# (3, 3, 128, 128)
# (128,)

# Dropout
# (128, 43)
# (43,)
