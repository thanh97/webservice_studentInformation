# refer from,
# https://github.com/haeusser/learning_by_association/blob/master/semisup/backend.py

import numpy as np
import tensorflow as tf

def loss (a, b, labels, visit_weight=1.0, walker_weight=1.0):
    """Add semi-supervised classification loss to the model.

    The loss constist of two terms: "walker" and "visit".

    Args:
      a: [N, emb_size] tensor with supervised embedding vectors.
      b: [M, emb_size] tensor with unsupervised embedding vectors.
      labels : [N] tensor with labels for supervised embeddings.
      walker_weight: Weight coefficient of the "walker" loss.
      visit_weight: Weight coefficient of the "visit" loss.
    """

    equality_matrix = tf.equal (tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast (equality_matrix, tf.float32)
    p_target = equality_matrix / tf.reduce_sum (equality_matrix, [1], keepdims=True)
    
    match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
    p_ab = tf.nn.softmax(match_ab, name='p_ab')
    p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
    p_aba = tf.matmul(p_ab, p_ba, name='p_aba')
   
    loss_walker = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (
        labels = p_target,
        logits = tf.log (1e-8 + p_aba)        
    )) * walker_weight
    
    return loss_walker, visit_loss (p_ab, visit_weight), estimate_error (p_aba, equality_matrix)

def visit_loss (p, weight=1.0):
    """Add the "visit" loss to the model.

    Args:
      p: [N, M] tensor. Each row must be a valid probability distribution
          (i.e. sum to 1.0)
      weight: Loss weight.
    """
    visit_probability = tf.reduce_mean(p, [0], keepdims=True, name='visit_prob')
    t_nb = tf.shape(p)[1]

    return tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (
        labels = tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
        logits = tf.log(1e-8 + visit_probability)
    )) * weight
    
    
def estimate_error (p_aba, equality_matrix):
    """Adds "walker" loss statistics to the graph.
    Args:
      p_aba: [N, N] matrix, where element [i, j] corresponds to the
          probalility of the round-trip between supervised samples i and j.
          Sum of each row of 'p_aba' must be equal to one.
      equality_matrix: [N, N] boolean matrix, [i,j] is True, when samples
          i and j belong to the same class.
    """
    # Using the square root of the correct round trip probalilty as an estimate
    # of the current classifier accuracy.
    per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba), 1)**0.5
    return tf.reduce_mean(1.0 - per_row_accuracy, name=p_aba.name[:-2] + '_esterr')
    