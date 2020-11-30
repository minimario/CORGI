from matplotlib import pyplot as plt
import keras
import numpy as np
from keras.models import load_model
from cnn_bounds_full import Model, run_gtsrb

NUM_CHANNELS = 16
CAM_DIM = 24

def get_cam_map(model, image, correct_class):
    """
    Given a Keras model, an input image (48, 3, 3), and
    the correct class, gets the CAM map of the image
    """

    # calculate the last convolutional layer output
    fc_weights = model.weights[-2].numpy()  # (NUM_CHANNELS, num_classes)
    last_conv_layer_output_model = keras.Model(model.input, model.get_layer(index=-3).output)
    last_conv_layer_output = last_conv_layer_output_model(np.expand_dims(image, 0))

    # calculate the CAM as a weighted sum of last convolutional layer outputs
    cam_map = np.zeros((CAM_DIM, CAM_DIM))
    for channel in range(NUM_CHANNELS):
        cam_map += (last_conv_layer_output[0, :, :, channel].numpy()
                    * fc_weights[channel, correct_class])

    return cam_map

def get_all_indices_split(array, num_indices):
    """
    Given an array and a value num_indices, splits the indices
    into the top k indices and the rest of the indices. 
    Returns ((topk_r, topk_c), (not_topk_r, not_topk_c))
    """
    sorted_r, sorted_c = np.unravel_index(np.argsort(-array, axis=None), array.shape)
    topk_r, topk_c = sorted_r[:num_indices], sorted_c[:num_indices]
    not_topk_r, not_topk_c = sorted_r[num_indices:], sorted_c[num_indices:]
    return ((topk_r, topk_c), (not_topk_r, not_topk_c))

def calculate_cam_bounds(model, image, correct_class, eps):
    """
    Given a Keras model, an input image (48, 3, 3), and
    a value of epsilon, calculate the LB and UB of the model's
    CAM map for all x in B(image, eps).
    """

    # don't calculate bounds if predicted class is not correct class
    pred_class = np.argmax(model.predict(np.expand_dims(image, axis=0)))
    assert(pred_class == correct_class)

    # get the LB's and UB's for the last convolutional layer from CNN-Cert
    cnn_model = Model(model, inp_shape=(48, 48, 3))
    LBs, UBs = run_gtsrb(cnn_model, image, correct_class, eps, 105)
    last_conv_lb = LBs[-3]  # (CAM_DIM, CAM_DIM, NUM_CHANNELS)
    last_conv_ub = UBs[-3]
    fc_weights = model.weights[-2].numpy()  # NUM_CHANNELS x 43

    cam_LB = np.zeros((CAM_DIM, CAM_DIM))
    cam_UB = np.zeros((CAM_DIM, CAM_DIM))

    for channel in range(NUM_CHANNELS):
        channel_weight = fc_weights[channel, pred_class]
        if channel_weight < 0:
            cam_LB += last_conv_ub[:, :, channel] * channel_weight
            cam_UB += last_conv_lb[:, :, channel] * channel_weight
        else:
            cam_LB += last_conv_lb[:, :, channel] * channel_weight
            cam_UB += last_conv_ub[:, :, channel] * channel_weight

    return cam_LB, cam_UB

def check_top_k(cam_map, cam_LB, cam_UB, num_indices):
    """
    Given a cam_map and lower/upper bounds for the CAM,
    checks if it satisfies the top-k interpretability
    robustness criteria.
    """
    (topk_r, topk_c), (not_topk_r, not_topk_c) = get_all_indices_split(cam_map, num_indices)
     # minimum lower bound of top k indices
    min_LB_top_k = np.min(cam_LB[topk_r, topk_c])

    # maximum upper bound of indices not in top k
    max_UB_not_top_k = np.max(cam_UB[not_topk_r, not_topk_c])  
    return min_LB_top_k > max_UB_not_top_k

def get_interpretability_bound(model, image, correct_class, num_indices):
    pred_class = np.argmax(model.predict(image[np.newaxis, :]))
    assert(pred_class == correct_class)

    cam_map = get_cam_map(model, image, correct_class)

    eps_min = 0
    eps_max = 0.005
    num_iterations = 15
    for it in range(num_iterations):
        print("Iteration {}, LB: {}, UB: {}".format(it, eps_min, eps_max))
        eps_mid = (eps_min + eps_max) / 2
        cam_LB, cam_UB = calculate_cam_bounds(model, image, correct_class, eps_mid)
        if check_top_k(cam_map, cam_LB, cam_UB, num_indices):
            eps_min = eps_mid
        else:
            eps_max = eps_mid
    print(eps_min, eps_max)
    return eps_min

def check_top_k_close(cam_map, cam_LB, cam_UB, k1, k2):
    (topk_r, topk_c), (not_topk_r, not_topk_c) = get_all_indices_split(cam_map, k1)
    not_top_k_values = cam_UB[not_topk_r, not_topk_c]
    top_k_necessary = np.partition(-not_top_k_values, k2-k1)[k2-k1]

    not_top_k_values.sort()
    not_top_k_values = np.flip(not_top_k_values)
    assert(-top_k_necessary == not_top_k_values[k2-k1])
    min_LB_top_k = np.min(cam_LB[topk_r, topk_c])
    return min_LB_top_k > -top_k_necessary

def get_interpretability_bound_rank(model, image, correct_class, k1, k2):
    correct_class = np.argmax(model.predict(image[np.newaxis, :]))
    cam_map = get_cam_map(model, image, correct_class)

    eps_min = 0
    eps_max = 0.008
    num_iterations = 12
    for it in range(num_iterations):
        print("Iteration {}, LB: {}, UB: {}".format(it, eps_min, eps_max))
        eps_mid = (eps_min + eps_max) / 2
        cam_LB, cam_UB = calculate_cam_bounds(model, image, correct_class, eps_mid)
        if check_top_k_close(cam_map, cam_LB, cam_UB, k1, k2):
            eps_min = eps_mid
        else:
            eps_max = eps_mid

    return eps_min

def run_experiment():
    # BEGIN THE EXPERIMENT
    i = 496
    model = load_model("../model_gtsrb.h5")
    top_k = 20
    from load_test import X_test, Y_test
    image = X_test[i]
    correct_class = np.argmax(Y_test[i])
    bound = get_interpretability_bound(model, image, correct_class, top_k)
    print(bound)

# run_experiment()