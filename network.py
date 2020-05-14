from networkComponents import *
from DcganConstants import *

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, Z):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = tf.reshape(tf.nn.relu((FullyConnected("linear", Z, 4*4*512))), [-1, 4, 4, 512])
            inputs = tf.nn.relu(Normalization(UpConvolution("deconv1", inputs, 256, 5, 2), "IN1"))
            inputs = tf.nn.relu(Normalization(UpConvolution("deconv2", inputs, 128, 5, 2), "IN2"))
            inputs = tf.nn.relu(Normalization(UpConvolution("deconv3", inputs, 64, 5, 2), "IN3"))
            inputs = tf.nn.tanh(UpConvolution("deconv4", inputs, IMG_C, 5, 2))
            return inputs

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = LeakyRelu(Convolution("conv1", inputs, 64, 5, 2, SpecNorm=True))
            inputs = LeakyRelu(Normalization(Convolution("conv2", inputs, 128, 5, 2, SpecNorm=True), "IN2"))
            inputs = LeakyRelu(Normalization(Convolution("conv3", inputs, 256, 5, 2, SpecNorm=True), "IN3"))
            inputs = LeakyRelu(Normalization(Convolution("conv4", inputs, 512, 5, 2, SpecNorm=True), "IN4"))
            inputs = tf.layers.flatten(inputs)
            return FullyConnected("liner", inputs, 1)

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
