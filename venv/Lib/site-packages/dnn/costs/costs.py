#-------------------------------------------------------------------------------
# Original source from: 
#   https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/cost.py
#-------------------------------------------------------------------------------


import logging
import tensorflow as tf

_FLOATX = tf.float32
_EPSILON = 1e-8

def sparse_softmax_cross_entropy (output, target, name=None):    
    return tf.reduce_mean (tf.nn.sparse_softmax_cross_entropy_with_logits (labels=target, logits=output, name=name))

def categorical_crossentropy (logits, labels):    
    logits = tf.nn.softmax (logits)
    logits /= tf.reduce_sum(logits,
                            reduction_indices = len (logits.get_shape()) - 1,
                            keepdims=True)
    # manual computation of crossentropy
    logits = tf.clip_by_value(logits, tf.cast(_EPSILON, dtype=_FLOATX),
                              tf.cast(1.-_EPSILON, dtype=_FLOATX))
    cross_entropy = - tf.reduce_sum(labels * tf.log(logits),
                           reduction_indices=len(logits.get_shape())-1)
    return tf.reduce_mean(cross_entropy)


def sigmoid_cross_entropy (output, target, name=None):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output, name=name))

def binary_cross_entropy (output, target, epsilon=1e-8, name='bce_loss'):
    with tf.name_scope(name):
        return tf.reduce_mean(tf.reduce_sum(-(target * tf.log(output + epsilon) + (1. - target) * tf.log(1. - output + epsilon)), axis=1))

def mean_squared_error(output, target, is_mean=False, name="mean_squared_error"):    
    with tf.name_scope(name):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), 1))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), 1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]))
        else:
            raise Exception("Unknow dimension")
        return mse

def normalized_mean_squared_error(output, target):
    with tf.name_scope("mean_squared_error_loss"):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=1))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=[1, 2]))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=[1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=[1, 2, 3]))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=[1, 2, 3]))
        nmse = tf.reduce_mean(nmse_a / nmse_b)
    return nmse

def absolute_difference_error(output, target, is_mean=False):
    # is_mean : boolean Whether compute the mean or sum for each example.
    with tf.name_scope("mean_squared_error_loss"):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            if is_mean:
                loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target), 1))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(tf.abs(output - target), 1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            if is_mean:
                loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target), [1, 2]))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(tf.abs(output - target), [1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            if is_mean:
                loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target), [1, 2, 3]))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(tf.abs(output - target), [1, 2, 3]))
        else:
            raise Exception("Unknow dimension")
        return loss

def cosine_distance_error (output, target):
    return tf.reduce_mean (
        tf.losses.cosine_distance (
            tf.nn.l2_normalize(output, 1), 
            tf.nn.l2_normalize(target, 1),
            1 # dim / axis
        )
    )

def dice_coe (output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    return tf.reduce_mean((2. * inse + smooth) / (l + r + smooth))

def dice_hard_coe (output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    return tf.reduce_mean((2. * inse + smooth) / (l + r + smooth))
    
def iou_coe (output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    return tf.reduce_mean((inse + smooth) / (union + smooth))
    
def cross_entropy_seq (logits, target_seqs, batch_size=None):  #, batch_size=1, num_steps=None):
    sequence_loss_by_example_fn = tf.contrib.legacy_seq2seq.sequence_loss_by_example
    loss = sequence_loss_by_example_fn([logits], [tf.reshape(target_seqs, [-1])], [tf.ones_like(tf.reshape(target_seqs, [-1]), dtype=tf.float32)])
    # [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_sum(loss)  #/ batch_size
    if batch_size is not None:
        cost = cost / batch_size
    return cost

def cosine_similarity (v1, v2):
    return (tf.reduce_sum(tf.multiply(v1, v2), 1) 
        / (tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1)) 
        * tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1))))

def mean_seq (logits, target_seqs, batch_size, sequence_length):
    weights = tf.ones ([batch_size, sequence_length])
    sequence_loss = tf.contrib.seq2seq.sequence_loss (
        logits = logits, targets = target_seqs, weights = weights
    )
    return tf.reduce_mean (sequence_loss)

def cross_entropy_seq_with_mask (logits, target_seqs, input_mask, return_details=False, name=None):
    targets = tf.reshape(target_seqs, [-1])  # to one vector
    weights = tf.to_float(tf.reshape(input_mask, [-1]))  # to one vector like targets
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name=name) * weights
    
    loss = tf.divide(
        tf.reduce_sum(losses),  # loss from mask. reduce_sum before element-wise mul with mask !!
        tf.reduce_sum(weights),
        name="seq_loss_with_mask")
    if return_details:
        return loss, losses, weights, targets
    else:
        return loss
