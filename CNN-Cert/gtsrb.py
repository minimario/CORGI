import numpy as np
import keras
from keras.models import load_model
from model import X, Y
from cnn_bounds_full import Model, run_gtsrb

# model = load_model("../model_clast.h5")
# cnn_model = Model(model, inp_shape=(48, 48, 3))

# load the model
model = load_model("../model_small.h5")

# load an image
image = X[0]
# 1) calculate the CAM
def get_cam_map(model, image, target_class):
    fc_weights = model.weights[-2].numpy()  # 128 x 43
    last_conv_layer_output_model = keras.Model(  # model to get the last conv layer outputs
        model.input, model.get_layer(index=-3).output
    )
    last_conv_layer_output = last_conv_layer_output_model(
        np.expand_dims(image, 0)
    )  # last conv layer output
    cam_map = np.zeros((24, 24))
    for channel in range(128):
        cam_map += (
            last_conv_layer_output[0, :, :, channel].numpy()
            * fc_weights[channel, target_class]
        )  # weighted sum of last conv layers
    return cam_map


# 2) calculate the top k indices of the CAM
def get_maximum_indices(array, num_indices):
    sorted_r, sorted_c = np.unravel_index(np.argsort(-array, axis=None), array.shape)
    topk_r, topk_c = sorted_r[:num_indices], sorted_c[:num_indices]
    not_topk_r, not_topk_c = sorted_r[num_indices:], sorted_c[num_indices:]
    return ((topk_r, topk_c), (not_topk_r, not_topk_c))


def get_cam_top_k(cam_map, k):
    return get_maximum_indices(cam_map, k)


# 3) calculate the bounds for the CAM and some given epsilon
def calculate_cam_bounds(model, cnn_model, image, eps):
    # make sure predicted class is correct
    pred_class = np.argmax(model.predict(np.expand_dims(image, axis=0)))
    correct_class = np.argmax(Y[0])
    print("Pred: {}, Correct: {}".format(pred_class, correct_class))

    # get the LB's and UB's for the CAM
    LBs, UBs = run_gtsrb(cnn_model, image, correct_class, eps, 105)
    last_conv_lb = LBs[-3]  # (24, 24, 128)
    last_conv_ub = UBs[-3]
    fc_weights = model.weights[-2].numpy()  # 128 x 43

    cam_LB = np.zeros((24, 24))
    cam_UB = np.zeros((24, 24))

    for channel in range(128):
        channel_weight = fc_weights[channel, pred_class]
        if channel_weight < 0:
            cam_LB += last_conv_ub[:, :, channel] * channel_weight
            cam_UB += last_conv_lb[:, :, channel] * channel_weight
        else:
            cam_LB += last_conv_lb[:, :, channel] * channel_weight
            cam_UB += last_conv_ub[:, :, channel] * channel_weight

    return cam_LB, cam_UB


def check_top_k(cam_map, cam_LB, cam_UB, num_indices):
    (topk_r, topk_c), (not_topk_r, not_topk_c) = get_cam_top_k(cam_map, num_indices)
    min_LB_top_k = np.min(
        cam_LB[topk_r, topk_c]
    )  # minimum lower bound of top k indices
    max_UB_not_top_k = np.max(
        cam_UB[not_topk_r, not_topk_c]
    )  # maximum upper bound of indices not in tosp k
    return min_LB_top_k > max_UB_not_top_k


def get_interpretability_bound(model, image, num_indices):
    correct_class = np.argmax(model.predict(image[np.newaxis, :]))
    cam_map = get_cam_map(model, image, correct_class)
    cnn_model = Model(model, inp_shape=(48, 48, 3))

    eps_min = 0
    eps_max = 0.05
    num_iterations = 15
    for it in range(num_iterations):
        print("Iteration {}, LB: {}, UB: {}".format(it, eps_min, eps_max))
        eps_mid = (eps_min + eps_max) / 2
        cam_LB, cam_UB = calculate_cam_bounds(model, cnn_model, image, eps_mid)
        if check_top_k(cam_map, cam_LB, cam_UB, num_indices):
            eps_min = eps_mid
        else:
            eps_max = eps_mid

    return eps_min


print(get_interpretability_bound(model, image, 10))
