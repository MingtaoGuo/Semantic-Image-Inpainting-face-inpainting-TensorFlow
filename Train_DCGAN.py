from networks import *
from PIL import Image
import numpy as np
import scipy.misc as misc
import os

class DCGAN:
    def __init__(self):
        self.img = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
        self.z = tf.placeholder(tf.float32, [None, Z_DIM])
        G = Generator("generator")
        D = Discriminator("discriminator")
        self.real_logits = D(self.img)
        self.fake_img = G(self.z)
        self.fake_logits = D(self.fake_img)
        self.D_loss = - tf.reduce_mean(tf.log(tf.sigmoid(self.real_logits + EPSILON))) - tf.reduce_mean(tf.log(1 - tf.sigmoid(self.fake_logits) + EPSILON))
        self.G_loss = - tf.reduce_mean(tf.log(tf.sigmoid(self.fake_logits) + EPSILON))
        self.Opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.G_loss, var_list=G.var)
        self.Opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.D_loss, var_list=D.var)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        file_path = "./celeba//"
        file_names = os.listdir(file_path)
        saver = tf.train.Saver()
        batch = np.zeros([BATCHSIZE, IMG_H, IMG_W, IMG_C])
        for i in range(8000):
            for c in range(1):
                batch_filenames = np.random.random_integers(0, file_names.__len__()-1, BATCHSIZE)
                for j, filename in enumerate(batch_filenames):
                    img = np.array(Image.open(file_path + file_names[filename]))
                    h = img.shape[0]
                    w = img.shape[1]
                    batch[j, :, :, :] = misc.imresize(img[(h // 2 - 70):(h // 2 + 70), (w // 2 - 70):(w // 2 + 70), :], [64, 64]) / 127.5 - 1.0
                z = np.random.standard_normal([BATCHSIZE, Z_DIM])
                self.sess.run(self.Opt_D, feed_dict={self.img: batch, self.z: z})
            z = np.random.standard_normal([BATCHSIZE, Z_DIM])
            self.sess.run(self.Opt_G, feed_dict={self.z: z})
            if i % 10 == 0:
                [D_loss, G_loss, fake_img] = self.sess.run([self.D_loss, self.G_loss, self.fake_img], feed_dict={self.img: batch, self.z: z})
                print("Step: %d, D_loss: %f, G_loss: %f"%(i, D_loss, G_loss))
                Image.fromarray(np.uint8((fake_img[0, :, :, :] + 1.0) * 127.5)).save("./result//"+str(i)+".jpg")
            if i % 100 == 0:
                saver.save(self.sess, "./save_para//dcgan.ckpt")

if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.train()
