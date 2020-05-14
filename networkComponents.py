import tensorflow as tf
#Comment the above line, and uncomment the below line in case of Tensorflow 2.0 environment
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()


def SpectralNormalization(name, w, iteration=1):
    wShape = w.shape.as_list()
    w = tf.reshape(w, [-1, wShape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, wShape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    uHat = u
    vHat = None

    for i in range(iteration):
        v =  tf.matmul(uHat, tf.transpose(w))
        vHat = v / (tf.reduce_sum(v ** 2) ** 0.5 + 1e-12)
        u_ = tf.matmul(vHat, w)
        uHat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + 1e-12)
    sigma = tf.matmul(tf.matmul(vHat, w), tf.transpose(uHat))
    wNorm = w / sigma
    with tf.control_dependencies([u.assign(uHat)]):
        wNorm = tf.reshape(wNorm, wShape)
    return wNorm

def LeakyRelu(x, slope=0.2):
    return tf.maximum(x, slope*x)


def Normalization(inputs, name):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        shift = tf.get_variable("shift", shape=mean.shape[-1], initializer=tf.constant_initializer([0.]))
        scale = tf.get_variable("scale", shape=mean.shape[-1], initializer=tf.constant_initializer([1.]))
        return (inputs - mean) * scale / tf.sqrt(variance + 1e-10) + shift

def Convolution(name, inputs, numOut, kernelSize, strides, padding="SAME", SpecNorm=False):
    with tf.variable_scope(name):
        Weight = tf.get_variable("W", shape=[kernelSize, kernelSize, int(inputs.shape[-1]), numOut], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable("b", shape=[numOut], initializer=tf.constant_initializer(0.))
        if SpecNorm:
            return tf.nn.conv2d(inputs, SpectralNormalization(name, Weight), [1, strides, strides, 1], padding) + bias
        else:
            return tf.nn.conv2d(inputs, Weight, [1, strides, strides, 1], padding) + bias

def UpConvolution(name, inputs, numOut, kernelSize, strides, padding="SAME"):
    with tf.variable_scope(name):
        Weight = tf.get_variable("W", shape=[kernelSize,kernelSize, numOut, int(inputs.shape[-1])], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable("b", [numOut], initializer=tf.constant_initializer(0.))
        
    return tf.nn.conv2d_transpose(inputs, Weight, [tf.shape(inputs)[0], int(inputs.shape[1])*strides, int(inputs.shape[2])*strides, numOut], [1, strides, strides, 1], padding=padding) + bias


def FullyConnected(name, inputs, numOut):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        Weight = tf.get_variable("W", [int(inputs.shape[-1]), numOut], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable("b", [numOut], initializer=tf.constant_initializer(0.))
        return tf.matmul(inputs, Weight) + bias



