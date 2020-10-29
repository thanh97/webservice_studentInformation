import tensorflow as tf


def grad (cost, learning_rate, global_step, **karg):
    return tf.train.GradientDescentOptimizer (learning_rate).minimize (cost, global_step = global_step, **karg)

def momentum (cost, learning_rate, global_step, use_nesterov = False, momentum = 0.9, **karg):    
    return tf.train.MomentumOptimizer (learning_rate, momentum, use_nesterov = use_nesterov).minimize (cost, global_step = global_step, **karg)

def nesterov_momentum (cost, learning_rate, global_step, **karg):
    return momentum (cost, learning_rate, global_step, True, **karg) 
nag = nesterov_momentum

def adagrad (cost, learning_rate, global_step, initial_accumulator_value = 0.1, **karg):
    return tf.train.AdagradOptimizer (learning_rate, initial_accumulator_value).minimize(cost, global_step = global_step, **karg)

def adadelta (cost, learning_rate, global_step, rho = 0.95, **karg):
    return tf.train.AdadeltaOptimizer (learning_rate, rho).minimize(cost, global_step = global_step, **karg)

def rmsprop (cost, learning_rate, global_step, decay = 0.9, momentum = 0.0, **karg):
    return tf.train.RMSPropOptimizer (learning_rate, decay, momentum).minimize(cost, global_step = global_step, **karg)

def adam (cost, learning_rate, global_step, beta1 = 0.9, beta2 = 0.999, **karg):
    return tf.train.AdamOptimizer (learning_rate, beta1, beta2).minimize (cost, global_step = global_step, **karg)

def ftrl (cost, learning_rate, global_step, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, **karg):
    return tf.train.FtrlOptimizer (learning_rate, global_step, initial_accumulator_value, l1_regularization_strength, l2_regularization_strength).minimize(cost, global_step = global_step, **karg)

def proxadagrad (cost, learning_rate, global_step, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, **karg):
    return tf.train.ProximalAdagradOptimizer (learning_rate, global_step, initial_accumulator_value, l1_regularization_strength, l2_regularization_strength).minimize(cost, global_step = global_step, **karg)

def proxgrad (cost, learning_rate, global_step, l1_regularization_strength=0.0, l2_regularization_strength=0.0, **karg):
    return tf.train.ProximalAdagradOptimizer (learning_rate, global_step, l1_regularization_strength, l2_regularization_strength).minimize(cost, global_step = global_step, **karg)

def clip (cost, learning_rate, global_step, min_, max_):
    train_op = tf.train.AdamOptimizer (learning_rate = learning_rate)
    gradients = train_op.compute_gradients (cost)
    capped_gradients = [(tf.clip_by_value (grad, min_, max_), var) for grad, var in gradients]
    return train_op.apply_gradients (capped_gradients, global_step = global_step)
