from network import *
from TrainDCGAN import DCGAN
from PIL import Image
import numpy as np
import DcganConstants

def getMask(imgHeight, imgWidth, maskWidth, maskHeight):
    Y = np.random.randint(0, maskWidth + 1)
    X = np.random.randint(0, maskHeight + 1)
    patch = np.ones([maskHeight, maskWidth])
    mask = np.zeros([imgHeight, imgWidth])
    mask[X:X+maskHeight, Y:Y+maskWidth] = patch
    return mask,X, Y


class ImageInpaint:
    def __init__(self):
        self.img = tf.placeholder(tf.float32, [1, IMG_H, IMG_W, IMG_C])
        self.mask = tf.placeholder(tf.float32, [1, IMG_H, IMG_W, IMG_C])
        self.y = self.img * (1 - self.mask) + self.mask
        self.z = tf.get_variable("z", [1, Z_DIM], initializer=tf.random_normal_initializer())
        G = Generator("generator")
        D = Discriminator("discriminator")
        self.output = G(self.z)
        self.logits = D(self.output)
        self.L_p = LAMBDA * tf.log(1 - tf.sigmoid(self.logits) + EPSILON)
        self.L_c = tf.reduce_sum(tf.abs(self.output - self.y) * (1 - self.mask))
        self.Loss = self.L_p + self.L_c
        self.Opt = tf.train.AdamOptimizer(1e-1).minimize(self.Loss, var_list=self.z)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator") +\
                               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminator"))
        dcgan = DCGAN()
        dcgan.train()
            
    def test(self):
        saved_model = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator") +\
                               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminator"))
        saved_model.restore(self.sess, "./save_para//.\\dcgan.ckpt")
        img = np.array(Image.open("./test//celeba//032895.jpg"))
        h = img.shape[0]
        w = img.shape[1]
        #img = misc.imresize(img[(h // 2 - 70):(h // 2 + 70), (w // 2 - 70):(w // 2 + 70), :], [64, 64]) / 127.5 - 1.0
        img = np.array(Image.fromarray(img).crop(((w // 2 - 70),(h // 2 - 70),(w // 2 + 70),(h // 2 + 70))).resize((64,64))) / 127.5 -1.0
        mask = getMask(IMG_H, IMG_W,MASK_W, MASK_H)[0]
        mask = np.dstack((mask, mask, mask))
        img = np.reshape(img, [1, 64, 64, 3])
        mask = np.reshape(mask, [1, 64, 64, 3])
        for i in range(20):
            self.sess.run(self.Opt, feed_dict={self.img: img, self.mask: mask})
            self.sess.run(tf.clip_by_value(self.z, -1, 1))
            if i % 10 == 0:
                [Loss, output, y] = self.sess.run([self.Loss, self.output, self.y], feed_dict={self.img: img, self.mask: mask})
                print("Step: %d, Loss: %f"%(i, Loss))
                poisson= output * mask + img * (1 - mask)
                out = np.concatenate((y[0, :, :, :], output[0, :, :, :], poisson[0, :, :, :]), 1)
                Image.fromarray(np.uint8((out + 1.0) * 127.5)).save("./completed_img//"+str(i)+".jpg")

if __name__ == "__main__":
    ii = ImageInpaint()
    if IS_DCGAN_TRAINED:
        ii.test()
    else:
        ii.train()
        ii.test()
