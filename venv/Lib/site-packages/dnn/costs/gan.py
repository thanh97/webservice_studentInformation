import tensorflow as tf

def vae_loss (y, x, mu, sigma):
    # https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py
    marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return loss, -marginal_likelihood, KL_divergence

def gan_loss (discriminator, logit, x):    
    # https://github.com/hwalsuklee/tensorflow-mnist-AAE/blob/master/aae.py    
    D_real_logit = discriminator (x)
    D_fake_logit = discriminator (logit)
        
    D_real_loss = tf.reduce_mean (tf.nn.sigmoid_cross_entropy_with_logits (logits = D_real_logit, labels = tf.ones_like(D_real_logit)))
    D_fake_loss = tf.reduce_mean (tf.nn.sigmoid_cross_entropy_with_logits (logits = D_fake_logit, labels = tf.zeros_like(D_fake_logit)))
    
    D_loss =  D_real_loss + D_fake_loss                            
    G_loss = tf.reduce_mean (tf.nn.sigmoid_cross_entropy_with_logits (logits = D_fake_logit, labels = tf.ones_like (D_fake_logit)))
    
    return D_loss, G_loss
