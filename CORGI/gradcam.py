from load_images import X, Y
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model
from cnn_bounds_full import Model, run_gtsrb
from matplotlib import pyplot as plt
from get_corgi_bounds import check_top_k

# load the model, last activation of the model is a softmax so remove it
model = load_model('../model_dense.h5')
model.layers[-1].activation = None
model.compile()

def get_last_conv_layer_output(model, x_0):
  """
  Gets actual value for the last convolutional layer
  Output shape: (12, 12, 16)
  """
  conv_model = tf.keras.models.Model(
      [model.inputs], [model.get_layer(index = -4).output]
  )
  conv_output = conv_model(x_0).numpy()
  return conv_output.squeeze()

def get_last_conv_layer_bounds(model, x_0, eps):
  """
  Gets lower and upper bounds for the last convolutional layer
  Output shape: [(12, 12, 16), (12, 12, 16)]
  """

  # squeeze x_0 to get image for consistency
  image = np.squeeze(x_0)

  # compute last conv layer upper and lower bounds
  cnn_model = Model(model, inp_shape=(48, 48, 3))
  LBs, UBs = run_gtsrb(cnn_model, image, 0, eps, 105)
  return LBs[-3], UBs[-3]

def get_twolayer_fc_bounds(x_l, x_u, w1, b1, w2):
    """
    Gets gradient bounds for a 2-layer FC network
    with ReLU activation function.

    [x: d neurons] -> [hidden: n neurons] -> [out: 1 neuron]
    x_l: (d,)
    x_u: (d,)
    w1: (n, d)
    b1: (n,)
    w2: (1, n)

    Output shape: [(d, ), (d, )]
    """

    # make sure shapes are correct, d-dim input, n hidden neurons
    d = x_l.shape[0]
    n = w1.shape[0]
    assert x_l.shape == x_u.shape == (d,)
    assert w1.shape == (n, d)
    assert b1.shape == (n,)
    assert w2.shape == (1, n)
    
    # calculate the minimum and maximum values of the hidden layer neurons
    w1_pos = np.maximum(w1, 0)
    w1_neg = np.minimum(w1, 0)
    layer1_min = w1_pos.dot(x_l) + w1_neg.dot(x_u) + b1
    layer1_max = w1_pos.dot(x_u) + w1_neg.dot(x_l) + b1   
    
    # create diagonal array for unsure, positive, and negative neurons
    unsure = np.logical_and(layer1_min <= 0, layer1_max >= 0)
    positive = layer1_min >= 0
    negative = layer1_max <= 0
    
    unsure_diag = np.diag(unsure).astype(np.float32)
    pos_diag = np.diag(positive).astype(np.float32)
    neg_diag = np.diag(negative).astype(np.float32)
    
    # ensure each hidden neuron only in one category
    assert(np.allclose(unsure_diag + pos_diag + neg_diag, np.eye(n)))

    # calculate the lower and upper bounds of the gradient
    prod_matrix = np.diag(w2[0]).dot(w1)
    prod_neg = np.minimum(0, prod_matrix)
    prod_pos = np.maximum(0, prod_matrix)
    grad_lo = np.sum(unsure_diag.dot(prod_neg) + pos_diag.dot(prod_matrix), axis = 0)
    grad_hi = np.sum(unsure_diag.dot(prod_pos) + pos_diag.dot(prod_matrix), axis = 0)

    return grad_lo, grad_hi    

def get_fc_gradient(model, x_0, target_class):
  """
  Given a model, an input image x_0, and a target class,
  returns the exact gradient of the FC portion of
  that model with respect to the last conv layer output

  Output shape: [(12, 12, 16)]
  """
  grad_model = tf.keras.models.Model(
      [model.inputs], [model.get_layer(index = -4).output, model.output]
  )
  # get true gradient of output with respect to convolutional layer outputs
  with tf.GradientTape() as tape:
      inputs = tf.cast(x_0, tf.float64)
      conv_outputs, predictions = grad_model(x_0)
      loss = predictions[:, target_class]

  grads = tape.gradient(loss, conv_outputs)
  return grads.numpy().squeeze()

def get_fc_gradient_bounds(model, x_0, target_class, eps):
  """
  Given a model, an input image x_0, and a target class,
  returns bounds on the gradient of the FC portion of
  that model with respect to the last conv layer output.

  Output shape: [(12, 12, 16), (12, 12, 16)]
  """

  # compute last conv layer upper and lower bounds
  last_conv_lb, last_conv_ub = get_last_conv_layer_bounds(model, x_0, eps)
  last_conv_lb = last_conv_lb.flatten()
  last_conv_ub = last_conv_ub.flatten()

  # extract fc layer weights and biases
  w1 = model.layers[-2].get_weights()[0]
  b1 = model.layers[-2].get_weights()[1]
  w2 = model.layers[-1].get_weights()[0][:, target_class:target_class+1]
  b2 = model.layers[-1].get_weights()[1][target_class]

  # get the fc gradient bounds using last conv layer bounds and weights
  grad_lo, grad_hi = get_twolayer_fc_bounds(last_conv_lb, last_conv_ub, w1.T, b1, w2.T)
  return grad_lo.reshape(12, 12, 16), grad_hi.reshape(12, 12, 16)

def get_gradcam(model, x_0, target_class):
  # get last conv output 
  last_conv_output = get_last_conv_layer_output(model, x_0)

  # compute weights for each conv map
  fc_gradients = get_fc_gradient(model, x_0, target_class)
  gradcam_weights = fc_gradients.mean(axis = (0, 1))

  # get gradcam and make sure it's positive
  gradcam = last_conv_output.dot(gradcam_weights)
  gradcam = np.maximum(gradcam, 0)

  return gradcam

def get_gradcam_bounds(model, x_0, target_class, eps):
  last_conv_lb, last_conv_ub = get_last_conv_layer_bounds(model, x_0, eps)
  fc_grad_lb, fc_grad_ub = get_fc_gradient_bounds(model, x_0, target_class, eps)
  
  assert(fc_grad_lb.shape == (12, 12, 16))
  grad_weights_lb = fc_grad_lb.mean(axis = (0, 1))
  grad_weights_ub = fc_grad_ub.mean(axis = (0, 1))

  prod_1 = last_conv_lb.dot(np.diag(grad_weights_lb))
  prod_2 = last_conv_lb.dot(np.diag(grad_weights_ub))
  prod_3 = last_conv_ub.dot(np.diag(grad_weights_lb))
  prod_4 = last_conv_ub.dot(np.diag(grad_weights_ub))
  
  mini = np.minimum(np.minimum(prod_1, prod_2), np.minimum(prod_3, prod_4))
  maxi = np.maximum(np.maximum(prod_1, prod_2), np.maximum(prod_3, prod_4))

  gradcam_lb = np.sum(mini, axis=2)
  gradcam_ub = np.sum(maxi, axis=2)
  return np.maximum(gradcam_lb, 0), np.maximum(gradcam_ub, 0)

def get_interpretability_bound_gradcam(model, x_0, correct_class, num_indices):
    pred_class = np.argmax(model.predict(x_0))
    assert(pred_class == correct_class)

    image = x_0.squeeze()
    gradcam_map = get_gradcam(model, x_0, correct_class)

    eps_min = 0
    eps_max = 0.01
    num_iterations = 12
    for it in range(num_iterations):
        print("Iteration {}, LB: {}, UB: {}".format(it, eps_min, eps_max))
        eps_mid = (eps_min + eps_max) / 2
        gradcam_LB, gradcam_UB = get_gradcam_bounds(model, image, correct_class, eps_mid)
        if check_top_k(gradcam_map, gradcam_LB, gradcam_UB, num_indices):
            eps_min = eps_mid
        else:
            eps_max = eps_mid
    return eps_min

def test_last_conv_bounds():
  # model, X needs to be defined
  x_0 = X[0:1]
  last_conv_lb, last_conv_ub = get_last_conv_layer_bounds(model, x_0, 0.001)
  conv_output = get_last_conv_layer_output(model, x_0)
  eps_error = 0.00001
  assert(np.all(conv_output >= last_conv_lb - eps_error))
  assert(np.all(conv_output <= last_conv_ub + eps_error))

def test_fc_gradient_bounds():
  # model, X, Y need to be defined
  x = X[0:1]
  y = Y[0:1]
  target = np.argmax(y)
  grads_real = get_fc_gradient(model, x, target)
  grads_lb, grads_ub = get_fc_gradient_bounds(model, x, target, 0.01)
  eps = 0.00001
  assert(np.all(grads_lb - eps <= grads_real))
  assert(np.all(grads_ub + eps >= grads_real))
  
def test_gradcam_bounds():
  gradcam = get_gradcam(model, X[0:1], np.argmax(Y[0]))
  gradcam_lb, gradcam_ub = get_gradcam_bounds(model, X[0:1], np.argmax(Y[0]), 0.00001)
  eps_error = 0.0001
  assert(np.alltrue(gradcam_lb - eps_error <= gradcam))
  assert(np.alltrue(gradcam <= gradcam_ub + eps_error))

# dummy function for testing purposes
def f(x, model):
  w1 = model.layers[-2].get_weights()[0].T
  b1 = model.layers[-2].get_weights()[1]
  w2 = model.layers[-1].get_weights()[0].T
  b2 = model.layers[-1].get_weights()[1]
  return w2.dot(np.maximum(0, w1.dot(x) + b1)) + b2

# test all the functions
def test():
  test_fc_gradient_bounds()
  test_last_conv_bounds()
  test_gradcam_bounds()

# image_index = 250
# gradcam = get_gradcam(model, X[image_index:image_index+1], np.argmax(Y[image_index]))

# gradcam = get_gradcam(model, X[image_index:image_index+1], np.argmax(Y[image_index]))
# plt.imshow(X[image_index])
# plt.show()
# plt.imshow(gradcam)
# plt.show()

# # # takes in 12 x 12
# heatmap = np.maximum(gradcam, 0)
# heatmap /= np.max(heatmap)
# heatmap = cv2.resize(heatmap, (48, 48))
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# plt.imshow(heatmap)
