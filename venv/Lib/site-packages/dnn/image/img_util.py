#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import numpy as np
import random as rnd
from PIL import Image
from scipy import ndimage
from .rect_util import Rect
from .transform import contrast_stretching
import cv2

def compute_norm_mat(base_width, base_height): 
    # normalization matrix used in image pre-processing 
    x      = np.arange(base_width)
    y      = np.arange(base_height)
    X, Y   = np.meshgrid(x, y)
    X      = X.flatten()
    Y      = Y.flatten() 
    A      = np.array([X*0+1, X, Y]).T 
    A_pinv = np.linalg.pinv(A)
    return A, A_pinv

def preproc_img(img, A, A_pinv):
    # compute image histogram 
    img_flat = img.flatten()
    img_hist = np.bincount(img_flat, minlength = 256)

    # cumulative distribution function 
    cdf = img_hist.cumsum() 
    cdf = cdf * (2.0 / cdf[-1]) - 1.0 # normalize 

    # histogram equalization 
    img_eq = cdf[img_flat] 

    diff = img_eq - np.dot(A, np.dot(A_pinv, img_eq))

    # after plane fitting, the mean of diff is already 0 
    std = np.sqrt(np.dot(diff,diff)/diff.size)
    if std > 1e-6: 
        diff = diff/std
    return diff.reshape(img.shape)

NORM_MATRIX = {}
def normalize (img_arr, reshape = None):
    global NORM_MATRIX

    w, h = img_arr.shape [:2]
    if (w, h) not in NORM_MATRIX:
        NORM_MATRIX [(w, h)] = compute_norm_mat (w, h)
    A, A_pinv = NORM_MATRIX [(w, h)]
    final_image = preproc_img (img_arr, A = A, A_pinv = A_pinv)
    if reshape:
       return final_image.reshape (reshape)
    return final_image   

def distort_img(img, roi = None, out_width = None, out_height = None, max_shift = 0.0, max_scale = 1.0, max_angle = 0.0, max_skew = 0.0, flip = False, noise = False, random_contrast = False):
    default_width, default_height = img.width, img.height
    
    roi = roi or Rect([0, 0, default_width, default_height])
    out_width = out_width or default_width
    out_height = out_height or default_height

    shift_y = out_height*max_shift*rnd.uniform(-1.0,1.0)
    shift_x = out_width*max_shift*rnd.uniform(-1.0,1.0)

    # rotation angle 
    angle = max_angle*rnd.uniform(-1.0,1.0)

    #skew 
    sk_y = max_skew*rnd.uniform(-1.0, 1.0)
    sk_x = max_skew*rnd.uniform(-1.0, 1.0)

    # scale 
    scale_y = rnd.uniform(1.0, max_scale) 
    if rnd.choice([True, False]): 
        scale_y = 1.0/scale_y 
    scale_x = rnd.uniform(1.0, max_scale) 

    if rnd.choice([True, False]): 
        scale_x = 1.0/scale_x 
    T_im = crop_img(img, roi, out_width, out_height, shift_x, shift_y, scale_x, scale_y, angle, sk_x, sk_y)
    if flip and rnd.choice([True, False]): 
        T_im = np.fliplr(T_im)
    if noise and rnd.choice([True, False]):
        T_im = add_noise (T_im, rnd.choice (['gauss', 's&p', 'poisson', 'speckle']))
        T_im = np.clip (T_im.astype ('int64'), 0, 255)                
    if random_contrast and rnd.choice([True, False]):
        T_im = contrast_stretching (T_im, rnd.randrange (20), rnd.randrange (80, 101))
    return T_im

def crop_img(img, roi, crop_width, crop_height, shift_x, shift_y, scale_x, scale_y, angle, skew_x, skew_y):
    # current face center 
    ctr_in = np.array((roi.center().y, roi.center().x))
    ctr_out = np.array((crop_height/2.0+shift_y, crop_width/2.0+shift_x))
    out_shape = (crop_height, crop_width)
    s_y = scale_y*(roi.height()-1)*1.0/(crop_height-1)
    s_x = scale_x*(roi.width()-1)*1.0/(crop_width-1)
    
    # rotation and scale 
    ang = angle*np.pi/180.0 
    transform = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    transform = transform.dot(np.array([[1.0, skew_y], [0.0, 1.0]]))
    transform = transform.dot(np.array([[1.0, 0.0], [skew_x, 1.0]]))
    transform = transform.dot(np.diag([s_y, s_x]))
    offset = ctr_in-ctr_out.dot(transform)
    
    # each point p in the output image is transformed to pT+s, where T is the matrix and s is the offset
    T_im = ndimage.interpolation.affine_transform(input = img, 
                                                  matrix = np.transpose(transform), 
                                                  offset = offset, 
                                                  output_shape = out_shape, 
                                                  order = 1,   # bilinear interpolation 
                                                  mode = 'reflect', 
                                                  prefilter = False)
    return T_im

def add_noise (image, noise_typ = 'gauss'):    
   if noise_typ == "gauss":
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,image.shape)      
      gauss = gauss.reshape(*image.shape)
      noisy = image + gauss
      return noisy

   elif noise_typ == "s&p":
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out

   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy

   elif noise_typ =="speckle":      
      gauss = np.random.randn(*image.shape)
      gauss = gauss.reshape(*image.shape)        
      noisy = image + image * gauss
      return noisy
