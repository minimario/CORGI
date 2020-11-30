from load_test import X_test, Y_test
from get_corgi_bounds import get_interpretability_bound
from get_attack_bounds import get_attack
from keras.models import load_model
import numpy as np

model = load_model('../model_gtsrb.h5')

k = 15


import pickle

for i in range(X_test.shape[0]):
    bounds = pickle.load(open('data_final.pkl', 'rb'))
    x_0 = X_test[i:i+1]
    image = np.squeeze(X_test[i])
    correct_class = Y_test[i]
    corgi = get_interpretability_bound(model, image, correct_class, k)
    attack = get_attack(model, x_0, correct_class, k)
    bounds.append((i, corgi, attack))
    pickle.dump(bounds, open('data_final.pkl', 'wb'))
