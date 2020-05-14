# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import os
from keras import backend as K
from keras.preprocessing.image import load_img, save_img, img_to_array
import matplotlib.pyplot as plt
from keras.applications import vgg19
from keras.models import Model
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf
import time

import os.path
from os import path


def get_content_loss(base_content, target):
    return K.sum(K.square(target - base_content))


# +
# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(input_tensor):
    assert K.ndim(input_tensor)==3
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram#/tf.cast(n, tf.float32)

def get_style_loss(style, combination, img_nrows, img_ncols):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows*img_ncols
    return K.sum(K.square(S - C))#/(4.0 * (channels ** 2) * (size ** 2))
    
def total_variation_loss(x, img_nrows, img_ncols):
    a = K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# -

def eval_loss_and_grads(x, img_nrows, img_ncols, f_outputs):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
        
    outs = f_outputs([x])
    loss_value = outs[0]
    
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
        
    return loss_value, grad_values


# this function will open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, img_nrows, img_ncols):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x, img_nrows, img_ncols):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# +
#set up a Python class named Evaluator that computes both the loss value and the gradients value at once,
#returns the loss value when called the first time, and caches the gradients for the next call.

class Evaluator():
    def __init__(self, img_nrows, img_ncols, f_outputs):
        self.loss_value = None
        self.grads_values = None
        self.img_nrows = img_nrows
        self.img_ncols = img_ncols
        self.f_outputs = f_outputs

    def loss(self, x):
        assert self.loss_value is None

        loss_value, grad_values = eval_loss_and_grads(x, self.img_nrows, self.img_ncols, self.f_outputs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


# -

def plot_results(best_img, base_image_path, style_image_path, img_nrows, img_ncols, CombinedPath):

    imgx = deprocess_image(best_img.copy(), img_nrows, img_ncols)

    plt.figure(figsize=(50,50))
    plt.subplot(5,5,1)
    plt.title("Base Image",fontsize=20)
    img_base = load_img(base_image_path)
    plt.imshow(img_base)

    plt.subplot(5,5,1+1)
    plt.title("Style Image",fontsize=20)
    img_style = load_img(style_image_path)
    plt.imshow(img_style)

    plt.subplot(5,5,1+2)
    plt.title("Final Image",fontsize=20)
    plt.imshow(imgx)
    
    plt.savefig(CombinedPath+'_3_results.png')



# # FUNCTION

def run_style_transfer(combined_folder_name, base_image_name, style_image_name, 
                       content_weight, style_weight, total_variation_weight,
                       iterations):

    combined_folder_name = combined_folder_name 
    dir = os.path.join("images","combined_images", combined_folder_name)
    if not os.path.exists(dir):
        os.mkdir(dir)

    ContentPath = 'images/base_images/'
    StylePath = 'images/style_images/'
    CombinedPath = dir

    base_image_path = ContentPath+base_image_name
    style_image_path = StylePath+style_image_name

    # dimensions of the generated picture.
    width, height = load_img(base_image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    # get tensor representations of our images
    base_image = K.variable(preprocess_image(base_image_path, img_nrows, img_ncols))
    style_reference_image = K.variable(preprocess_image(style_image_path, img_nrows, img_ncols))

    # this will contain our generated image
    if K.image_data_format() == 'channels_first':
        combination_image = K.placeholder((1,3,img_nrows, img_ncols))
    else:
        combination_image = K.placeholder((1,img_nrows, img_ncols,3))
    
    # combine the 3 images into a single Keras tensor
    input_tensor = K.concatenate([base_image,
                                  style_reference_image,
                                  combination_image
                                  ], axis=0)
    
    from keras.applications.vgg19 import VGG19

    model = VGG19(input_tensor=input_tensor,
                  include_top = False,
                  weights='imagenet')

    # Content layer where will pull our feature maps
    content_layers = 'block5_conv2' 

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1',
                    'block5_conv1'
                   ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    content_weight=content_weight
    style_weight=style_weight
    total_variation_weight=total_variation_weight

    # combine these loss functions into a single scalar
    loss = K.variable(0.0)
    layer_features = outputs_dict[content_layers]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * get_content_loss(base_image_features,combination_features)

    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = get_style_loss(style_reference_features, combination_features, img_nrows, img_ncols)
        loss = loss + (style_weight / len(style_layers)) * sl

    loss = loss + total_variation_weight * total_variation_loss(combination_image, img_nrows, img_ncols)

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)

    outputs = [loss]
    if isinstance(grads, (list,tuple)):
        outputs += grads
    else:
        outputs.append(grads)
    
    f_outputs = K.function([combination_image], outputs)

    x_opt = preprocess_image(base_image_path, img_nrows, img_ncols)
    
    evaluator = Evaluator(img_nrows, img_ncols, f_outputs)

    iterations= iterations

    #storing best results here
    best_loss, best_img = float('inf'), None
    
    for i in range(iterations):
        # run the gradient-ascent process using SciPy’s L-BFGS algorithm, 
        # saving the current generated image at each iteration of the algorithm
        x_opt, min_val, info= fmin_l_bfgs_b(evaluator.loss, 
                                            x_opt.flatten(), #must be flattened for L-BFGS 
                                            fprime=evaluator.grads,
                                            maxfun=20,
                                            disp=True,
                                           )
        
        # saving results every 5 interations
        if i % 5 == 0:
            imgx = deprocess_image(x_opt.copy(), img_nrows, img_ncols)
            plt.imsave(CombinedPath+'/combined_result_%d.png' % i, imgx)

        if min_val < best_loss:
            # Update best loss and best image from total loss. 
            best_loss = min_val
            best_img = x_opt.copy()

    best = deprocess_image(best_img.copy(), img_nrows, img_ncols)
    plt.imsave(CombinedPath+'/BEST.png', best)

    plot_results(best_img, base_image_path, style_image_path, img_nrows, img_ncols, CombinedPath)


# # RESULTS

run_style_transfer('Alex_2_Van_Gogh', 'base_image.jpg', 'van_gogh.jpg', 0.5, 10.0, 10.0, 1)

run_style_transfer('Alex_2_Greek_Statue', 'base_image.jpg', 'greek_style_image.jpg', 0.5, 10.0, 10.0, 10)

run_style_transfer('Forest_2_Monet', 'forestry.jpg', 'monet.jpg', 0.5, 10.0, 10.0, 10)

run_style_transfer('London_Bridge_2_Cubism', 'london_bridge.jpg', 'cubist_buildings.jpg', 0.5, 10.0, 10.0, 100)

run_style_transfer('Chia_2_ChiaArt', 'chia.png', 'chia_art.png', 0.5, 10.0, 10.0, 100)


