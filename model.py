import tensorflow as tf
import numpy as np
from glob import glob
from ops import *
from utils import *
from six.moves import xrange
import time


class DnCNN(object):
    def __init__(self, sess, patch_size=40, batch_size=128,
                 output_size=40, input_c_dim=1, output_c_dim=1,
                 sigma=25, clip_b=0.025, lr=0.001, epoch=50,
                 ckpt_dir='./checkpoint', sample_dir='./sample', test_save_dir='./test',
                 dataset='BSD400', testset='BSD68', evalset='Set12'):
        self.sess = sess
        self.is_gray = (input_c_dim == 1)
        self.batch_size = batch_size
        self.patch_sioze = patch_size
        self.output_size = output_size
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.sigma = sigma
        self.clip_b = clip_b
        self.lr = lr
        self.numEpoch = epoch
        self.ckpt_dir = ckpt_dir
        self.trainset = dataset
        self.testset = testset
        self.evalset = evalset
        self.sample_dir = sample_dir
        self.test_save_dir = test_save_dir
        self.epoch = epoch
        # Fixed params
        self.save_every_epoch = 10
        self.eval_every_epoch = 10
        # Adam setting (default setting)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = 0.01
        self.epsilon = 1e-8

        self.build_model()

    def build_model(self):
        self.X_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.X = self.X_ + tf.truncated_normal(shape=tf.shape(self.X_), stddev=self.sigma / 255.0) # noisy batches
        # layer 1
        with tf.variable_scope('conv1'):
            layer_1_output = self.layer(self.X, [3, 3, self.input_c_dim, 64], useBN=False)
        # layer 2 to 16
        with tf.variable_scope('conv2'):
            layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64])
        with tf.variable_scope('conv3'):
            layer_3_output = self.layer(layer_2_output, [3, 3, 64, 64])
        with tf.variable_scope('conv4'):
            layer_4_output = self.layer(layer_3_output, [3, 3, 64, 64])
        with tf.variable_scope('conv5'):
            layer_5_output = self.layer(layer_4_output, [3, 3, 64, 64])
        with tf.variable_scope('conv6'):
            layer_6_output = self.layer(layer_5_output, [3, 3, 64, 64])
        with tf.variable_scope('conv7'):
            layer_7_output = self.layer(layer_6_output, [3, 3, 64, 64])
        with tf.variable_scope('conv8'):
            layer_8_output = self.layer(layer_7_output, [3, 3, 64, 64])
        with tf.variable_scope('conv9'):
            layer_9_output = self.layer(layer_8_output, [3, 3, 64, 64])
        with tf.variable_scope('conv10'):
            layer_10_output = self.layer(layer_9_output, [3, 3, 64, 64])
        with tf.variable_scope('conv11'):
            layer_11_output = self.layer(layer_10_output, [3, 3, 64, 64])
        with tf.variable_scope('conv12'):
            layer_12_output = self.layer(layer_11_output, [3, 3, 64, 64])
        with tf.variable_scope('conv13'):
            layer_13_output = self.layer(layer_12_output, [3, 3, 64, 64])
        with tf.variable_scope('conv14'):
            layer_14_output = self.layer(layer_13_output, [3, 3, 64, 64])
        with tf.variable_scope('conv15'):
            layer_15_output = self.layer(layer_14_output, [3, 3, 64, 64])
        with tf.variable_scope('conv16'):
            layer_16_output = self.layer(layer_15_output, [3, 3, 64, 64])
        # layer 17
        with tf.variable_scope('conv17'):
            self.Y = self.layer(layer_16_output, [3, 3, 64, self.output_c_dim], useBN=False,
                                useReLU=False)  # predicted noise
        # L2 loss
        self.Y_ = self.X - self.X_  # noisy image - clean image
        self.loss = (1.0 / self.batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_step = optimizer.minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        # create this init op after all variables specified
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        print("[*] Initialize model successfully...")

    def conv_layer(self, inputdata, weightshape, b_init, stridemode):
        # weights
        W = tf.get_variable('weights', weightshape,
                            initializer=tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
        b = tf.get_variable('biases', [1, weightshape[-1]], initializer=tf.constant_initializer(b_init))
        # convolutional layer
        return tf.add(tf.nn.conv2d(inputdata, W, strides=stridemode, padding="SAME"), b)  # SAME with zero padding

    def bn_layer(self, logits, output_dim, b_init=0.0):
        alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer=
        tf.constant_initializer(get_bn_weights([1, output_dim], self.clip_b, self.sess)))
        beta = tf.get_variable('bn_beta', [1, output_dim], initializer=
        tf.constant_initializer(b_init))
        return batch_normalization(logits, alpha, beta, isCovNet=True)

    def layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True, useReLU=True):
        logits = self.conv_layer(inputdata, filter_shape, b_init, stridemode)
        if useReLU == False:
            output = logits
        else:
            if useBN:
                output = tf.nn.relu(self.bn_layer(logits, filter_shape[-1]))
            else:
                output = tf.nn.relu(logits)
        return output

    def train(self):
        # init the variables
        self.sess.run(self.init)
        # get data
        eval_files = glob('./data/test/{}/*.png'.format(self.evalset))
        eval_data = load_images(eval_files)  # list of array of different size, 4-D, pixel value range is 0-255
        data = load_data(filepath='./data/img_clean_pats.npy')
        numBatch = int(data.shape[0] / self.batch_size)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        iter_num = 0
        print("[*] Start training : ")
        start_time = time.time()
        self.evaluate(iter_num, eval_data)  # eval_data value range is 0-255
        for epoch in xrange(self.epoch):
            np.random.shuffle(data)
            for batch_id in xrange(numBatch):
                batch_images = data[batch_id * self.batch_size:(batch_id + 1) * self.batch_size, :, :, :]
                batch_images = np.array(batch_images / 255.0, dtype=np.float32)  # normalize the data to 0-1
                _, loss, summary = self.sess.run([self.train_step, self.loss, merged],
                                                 feed_dict={self.X_: batch_images})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, self.eval_every_epoch) == 0:
                self.evaluate(epoch, eval_data)  # eval_data value range is 0-255
            # save the model
            if np.mod(epoch + 1, self.save_every_epoch) == 0:
                self.save(iter_num)
        print("[*] Finish training.")

    def save(self, iter_num):
        model_name = "DnCNN.model"
        model_dir = "%s_%s_%s" % (self.trainset,
                                  self.batch_size, self.patch_sioze)
        checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print("[*] Saving model...")
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        model_dir = "%s_%s_%s" % (self.trainset, self.batch_size, self.patch_sioze)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        test_files = glob('./data/test/{}/*.png'.format(self.testset))
        assert len(test_files) != 0, 'No testing data!'
        if self.load(self.ckpt_dir):
            print(" [*] Load weights SUCCESS...")
        else:
            print(" [!] Load weights FAILED...")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in xrange(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            predicted_noise, noisy_image = self.sess.run([self.Y, self.X],
                                                         feed_dict={self.X_: clean_image})
            output_clean_image = noisy_image - predicted_noise
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(self.test_save_dir, 'noisy%d.png' % idx), noisyimage)
            save_images(os.path.join(self.test_save_dir, 'denoised%d.png' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)

    def evaluate(self, iter_num, test_data):
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0
        for idx in xrange(len(test_data)):
            clean_image = test_data[idx].astype(np.float32) / 255.0
            predicted_noise, noisy_image = self.sess.run([self.Y, self.X],
                                                         feed_dict={self.X_: clean_image})
            output_clean_image = noisy_image - predicted_noise
            groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(self.sample_dir, 'test%d_%d.png' % (idx, iter_num)),
                        groundtruth, noisyimage, outputimage)
        avg_psnr = psnr_sum / len(test_data)
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
