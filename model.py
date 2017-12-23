import time
from glob import glob

from ops import *
from utils import *


def dncnn(input, is_training=True, output_channels=1):
    output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu, name='conv1')
    for layers in xrange(2, 16 + 1):
        output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers)
        output = tf.layers.batch_normalization(output, training=is_training, name='bn%d' % layers)
        output = tf.nn.relu(output)
    output = tf.layers.conv2d(output, output_channels, 3, padding='same', name='conv17')
    return input - output


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
        self.save_every_epoch = 2
        self.eval_every_epoch = 2
        # Adam setting (default setting)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = 0.01
        self.epsilon = 1e-8

        self.build_model()

    def build_model(self):
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_trning')
        self.X = self.Y_ + tf.truncated_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy batches
        self.Y = dncnn(self.X, is_training=self.is_training)
        self.loss = (1.0 / self.batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        # create this init op after all variables specified
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        print("[*] Initialize model successfully...")

    # def conv_layer(self, inputdata, weightshape, b_init, stridemode):
    #     # weights
    #     W = tf.get_variable('weights', weightshape,
    #                         initializer=tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
    #     b = tf.get_variable('biases', [1, weightshape[-1]], initializer=tf.constant_initializer(b_init))
    #     # convolutional layer
    #     return tf.add(tf.nn.conv2d(inputdata, W, strides=stridemode, padding="SAME"), b)  # SAME with zero padding
    #
    # def bn_layer(self, logits, output_dim, b_init=0.0):
    #     alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer=
    #     tf.constant_initializer(get_bn_weights([1, output_dim], self.clip_b, self.sess)))
    #     beta = tf.get_variable('bn_beta', [1, output_dim], initializer=
    #     tf.constant_initializer(b_init))
    #     return batch_normalization(logits, alpha, beta, isCovNet=True)
    #
    # def layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True, useReLU=True):
    #     logits = self.conv_layer(inputdata, filter_shape, b_init, stridemode)
    #     if useReLU == False:
    #         output = logits
    #     else:
    #         if useBN:
    #             output = tf.nn.relu(self.bn_layer(logits, filter_shape[-1]))
    #         else:
    #             output = tf.nn.relu(logits)
    #     return output

    def train(self):
        # init the variables
        self.sess.run(self.init)
        # get data
        eval_files = glob('./data/test/{}/*.png'.format(self.evalset))
        eval_data = load_images(eval_files)  # list of array of different size, 4-D, pixel value range is 0-255
        data = load_data(filepath='./data/img_clean_pats.npy')
        numBatch = int(data.shape[0] / self.batch_size)
        load_model_status, global_step = self.load(self.ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_data)  # eval_data value range is 0-255
        for epoch in xrange(start_epoch, self.epoch):
            np.random.shuffle(data)
            for batch_id in xrange(start_step, numBatch):
                batch_images = data[batch_id * self.batch_size:(batch_id + 1) * self.batch_size, :, :, :]
                batch_images = np.array(batch_images / 255.0, dtype=np.float32)  # normalize the data to 0-1
                _, loss, summary = self.sess.run([self.train_step, self.loss, merged],
                                                 feed_dict={self.Y_: batch_images, self.is_training: True})
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
        model_name = "DnCNN-tensorflow"
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
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            self.saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        test_files = glob('./data/test/{}/*.png'.format(self.testset))
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(self.ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in xrange(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image, self.is_training: False})
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
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image, self.is_training: False})
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
