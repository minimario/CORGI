
import numpy as np
from skimage import color, exposure, transform
from skimage import io
import os
import glob

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


def get_class(img_path):
    return int(img_path.split('/')[-2])

root_dir = '/home/minimario/Research/gtsrb/GTSRB/Final_Training/Images'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
all_img_paths.sort()

# np.random.seed(1234)
# np.random.shuffle(all_img_paths)
# for img_path in all_img_paths[0:500]:
#     img = preprocess_img(io.imread(img_path))
#     label = get_class(img_path)
#     imgs.append(img)
#     labels.append(label)

for i in range(0, len(all_img_paths), 50): # len(all_img_paths)
  img_path = all_img_paths[i]
  img = preprocess_img(io.imread(img_path))
  label = get_class(img_path)
  imgs.append(img)
  labels.append(label)

X = np.array(imgs, dtype='float32').transpose(0, 2, 3, 1) # n 48 48 3
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

## load test data

import pandas as pd
test = pd.read_csv('../GTSRB/GT-final_test.csv', sep=';')

# Load test dataset
X_test = []
Y_test = []
i = 0
for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('../GTSRB/Final_Test/Images/', file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    Y_test.append(class_id)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_test = X_test.transpose(0, 2, 3, 1)