from load_test import X_test, Y_test
from get_corgi_bounds import get_interpretability_bound
from get_attack_bounds import get_attack
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('../model_gtsrb.h5')

k = 15

import pickle
d = {}
lut = pickle.load(open('data_final.pkl', 'rb'))
for (i, x, y) in lut:
  d[i] = (x, y)
indices = [[] for _ in range(43)]
Y_pred = np.argmax(model.predict(X_test), axis=1)
for i in range(Y_test.shape[0]):
  if (Y_pred[i] != Y_test[i]):
    continue
  if len(indices[Y_test[i]]) > 100:
    continue
  indices[Y_test[i]].append(i)

out = [[] for _ in range(10)]
for c in range(4, 10):
  ct = 0
  for i in indices[c]:
    out = pickle.load(open('data_final_4.pkl', 'rb'))
    if i <= lut[-1][0]:
      corgi, attack = d[i]
    else:
      x_0 = X_test[i:i+1]
      image = np.squeeze(X_test[i])
      correct_class = Y_test[i]
      corgi = get_interpretability_bound(model, image, correct_class, k)
      attack = get_attack(x_0, correct_class, k)
    print(f"class {c}, image {ct}")
    ct+=1
    print(out)
    out[c].append((i, corgi, attack))
    pickle.dump(out, open('data_final_4.pkl', 'wb'))
