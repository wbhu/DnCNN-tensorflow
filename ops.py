import tensorflow as tf
import math
from six.moves import xrange


def batch_normalization(logits, scale, offset, isCovNet=False, name="bn"):
    if isCovNet:
        mean, var = tf.nn.moments(logits, [0, 1, 2])
    else:
        mean, var = tf.nn.moments(logits, [0])
    output = tf.nn.batch_normalization(logits, mean, var, offset, scale, variance_epsilon=1e-5)
    return output


def get_conv_weights(weight_shape, sess, name="get_conv_weights"):
    return math.sqrt(2 / (9.0 * 64)) * sess.run(tf.truncated_normal(weight_shape))


def get_bn_weights(weight_shape, clip_b, sess, name="get_bn_weights"):
    weights = get_conv_weights(weight_shape, sess)
    return clipping(weights, clip_b)


def clipping(A, clip_b, name="clipping"):
    h, w = A.shape
    for i in xrange(h):
        for j in xrange(w):
            if A[i, j] >= 0 and A[i, j] < clip_b:
                A[i, j] = clip_b
            elif A[i, j] > -clip_b and A[i, j] < 0:
                A[i, j] = -clip_b
    return A
