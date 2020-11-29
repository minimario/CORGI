import tensorflow as tf
import keras
from keras.models import load_model
from keras.activations import relu, softplus
import numpy as np
from gtsrb import get_cam_map
from load_images import X, Y

def get_maximum_indices(array, num_indices):
    """
    Given any 2d array and a constant k, returns the top k
    indices of that array in the format (r_array, c_array),
    both of which are length k arrays
    """
    sorted_r, sorted_c = np.unravel_index(np.argsort(-array, axis=None), array.shape)
    topk_r, topk_c = sorted_r[:num_indices], sorted_c[:num_indices]
    not_topk_r, not_topk_c = sorted_r[num_indices:], sorted_c[num_indices:]
    return (topk_r, topk_c)

def get_top_k_overlap(array_1, array_2, k):
  """
  Given two nxn 2d arrays, returns the number of indices
  in the top k values of both array_1 and array_2
  """
  n = array_1.shape[0]
  assert array_1.shape == (n, n)
  assert array_2.shape == (n, n)
  r_max_1, c_max_1 = get_maximum_indices(array_1, k)
  r_max_2, c_max_2 = get_maximum_indices(array_2, k)
  return len(np.intersect1d(n*r_max_1+c_max_1, n*r_max_2+c_max_2))

def helper_orig(x_0, correct_class, r_max, c_max):
  # CAM map calculation with TF tracking
  cam_map = tf.zeros([CAM_DIM, CAM_DIM], dtype=tf.dtypes.float32)
  last_conv_layer_output = last_conv_layer_output_model_relu(x_0)
  for channel in range(NUM_CHANNELS):
    cam_map += last_conv_layer_output[0, :, :, channel] * fc_weights[channel, correct_class]
  dissimilarity = 0
  for i in range(top_k):
    dissimilarity -= cam_map[r_max[i]][c_max[i]]
  return cam_map, dissimilarity

def helper_softplus(x_0, correct_class, r_max, c_max):
  # CAM map calculation with TF tracking
  cam_map = tf.zeros([CAM_DIM, CAM_DIM], dtype=tf.dtypes.float32)
  last_conv_layer_output = last_conv_layer_output_model_softplus(x_0)
  for channel in range(NUM_CHANNELS):
    cam_map += last_conv_layer_output[0, :, :, channel] * fc_weights[channel, correct_class]
  dissimilarity = 0
  for i in range(top_k):
    dissimilarity -= cam_map[r_max[i]][c_max[i]]
  return cam_map, dissimilarity

def interpretability_attack(x_0, correct_class, k, epsilon):
  # get original CAM map and top k pixels
  image = np.squeeze(x_0)
  cam_map_orig = get_cam_map(model, image, correct_class)
  r_max, c_max = get_maximum_indices(cam_map_orig, k)

  NUM_STEPS = 100
  x = tf.Variable(x_0, dtype=tf.float32)
  for cycle in range(NUM_STEPS):
    with tf.GradientTape(persistent=True) as tape:
      # get the cam_map and dissimilarity score for the new image
      cam_map, dissimilarity = helper_orig(x, correct_class, r_max, c_max)
      # cam_map_softplus, dissimilarity_softplus = helper_softplus(x, correct_class, r_max, c_max)

    # check if it's a successful attack
    if np.argmax(model(x)) == correct_class:
      num_overlap = get_top_k_overlap(cam_map_orig, cam_map.numpy(), top_k)
      if num_overlap != top_k:
        print("attacked on", cycle)
        return True

    grads = tape.gradient(dissimilarity, [x])
    candidate_inp = np.clip(x + epsilon / 30 * tf.math.sign(grads[0]).numpy(), 0, 1)
    pert = np.clip(candidate_inp - image, -epsilon, epsilon)
    x.assign(image + pert)

  return False
  
def get_attack(image, correct_class, k):
  lo = 0
  hi = 0.02
  for steps in range(10):
    mid = (lo + hi) / 2
    if interpretability_attack(image, correct_class, k, mid):
      hi = mid
    else:
      lo = mid
  return hi

## START THE EXPERIMENT

NUM_CHANNELS = 16
CAM_DIM = 24

model = load_model('../model_gtsrb.h5')
model_sp = load_model('../model_gtsrb.h5')
for layer in model_sp.layers:
  if hasattr(layer, 'activation') and layer.activation == relu:
    layer.activation = softplus
model_sp.compile()

last_conv_layer_output_model_softplus = keras.Model(model_sp.input, model_sp.get_layer(index = -3).output) 
last_conv_layer_output_model_relu = keras.Model(model.input, model.get_layer(index = -3).output) 

fc_weights = np.asarray(model.weights[-2].numpy())

bounds = {}
top_k = 15
for i in range(120):
  x_0 = X[i:i+1]
  correct_class = np.argmax(Y[i])
  if np.argmax(model.predict(X[i:i+1])) != np.argmax(Y[i]):
    print("wrong")
    continue
  new_bound = get_attack(x_0, correct_class, top_k)
  bounds[i] = new_bound
  print(bounds)
