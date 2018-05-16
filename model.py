import time
from glob import glob
import numpy as np
import tensorflow as tf
import random
import os
import cv2

def dncnn(input, is_training=True, output_channels=3):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in xrange(2, 19+1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))   
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same',use_bias=False)
    return input - output

filepaths = glob('./data/train/original/*.png') #takes all the paths of the png files in the train folder
filepaths = sorted(filepaths)                           #Order the list of files
filepaths_noisy = glob('./data/train/noisy/*.png')
filepaths_noisy = sorted(filepaths_noisy)
ind = range(len(filepaths))

class denoiser(object):
    def __init__(self, sess, input_c_dim=3, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim])
        self.Y = dncnn(self.X, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.dataset = dataset(sess)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, eval_files, noisy_files, sample_dir, summary_writer):
        print("[*] Evaluating...")
        psnr_sum = 0
        
        for i in xrange(10):
            clean_image = cv2.imread(eval_files[i])
            clean_image = clean_image.astype('float32') / 255.0
            clean_image = clean_image[np.newaxis, ...]
            noisy = cv2.imread(noisy_files[i])
            noisy = noisy.astype('float32') / 255.0
            noisy = noisy[np.newaxis, ...]
            
            output_clean_image = self.sess.run(
                [self.Y],feed_dict={self.Y_: clean_image,
                           self.X: noisy,
                           self.is_training: False})
            psnr = psnr_scaled(clean_image, output_clean_image)
            print("img%d PSNR: %.2f" % (i + 1, psnr))
            psnr_sum += psnr

        avg_psnr = psnr_sum / 10

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)


    def train(self, eval_files, noisy_files, batch_size, ckpt_dir, epoch, lr, sample_dir, eval_every_epoch=1):

        numBatch = int(len(filepaths) * 2)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
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
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        clip_all_weights = tf.get_collection("max_norm")        

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_files, noisy_files, sample_dir=sample_dir,summary_writer=writer)  # eval_data value range is 0-255
        for epoch in xrange(start_epoch, epoch):
            batch_noisy = np.zeros((128,64,64,3),dtype='float32')
            batch_images = np.zeros((128,64,64,3),dtype='float32')
            for batch_id in xrange(start_step, numBatch):
              try:
                res = self.dataset.get_batch() # If we get an error retrieving a batch of patches we have to reinitialize the dataset
              except KeyboardInterrupt:
                raise
              except:
                self.dataset = dataset(self.sess) # Dataset re init
              if batch_id==0:
                batch_noisy = np.zeros((128,64,64,3),dtype='float32')
                batch_images = np.zeros((128,64,64,3),dtype='float32')
              ind1 = range(res.shape[0]/2)
              ind1 = np.multiply(ind1,2)
              for i in range(128):
                random.shuffle(ind1)
                ind2 = random.randint(0,8-1)
                batch_noisy[i] = res[ind1[0],ind2]
                batch_images[i] = res[ind1[0]+1,ind2]
#              for i in range(64):
#                cv2.imshow('raw',batch_images[i])
#                cv2.imshow('noisy',batch_noisy[i])
              _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                                                 feed_dict={self.Y_: batch_images, self.X: batch_noisy, self.lr: lr[epoch],
                                                            self.is_training: True})
              self.sess.run(clip_all_weights)          
              
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                    % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
              iter_num += 1
              writer.add_summary(summary, iter_num)
              
            if np.mod(epoch + 1, eval_every_epoch) == 0: ##Evaluate and save model
                self.evaluate(iter_num, eval_files, noisy_files, sample_dir=sample_dir,summary_writer=writer)
                self.save(iter_num, ckpt_dir)
        print("[*] Training finished.")

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, eval_files, noisy_files, ckpt_dir, save_dir, temporal):
        """Test DnCNN"""
        # init variables
        tf.global_variables_initializer().run()
        assert len(eval_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        psnr_sum2 = 0
        if temporal:
            createBackSub = False
            
        for i in xrange(len(eval_files)):
            clean_image = cv2.imread(eval_files[i])
            clean_image = clean_image.astype('float32') / 255.0
            clean_image = clean_image[np.newaxis, ...]
            
            noisy = cv2.imread(noisy_files[i])
            noisy = noisy.astype('float32') / 255.0
            noisy = noisy[np.newaxis, ...] 
          
            output_clean_image = self.sess.run(
                [self.Y],feed_dict={self.Y_: clean_image, self.X: noisy,
                                    self.is_training: False})
            
            out1 = np.asarray(output_clean_image)
               
            psnr = psnr_scaled(clean_image, out1[0,0])
            psnr1 = psnr_scaled(clean_image, noisy)
            
            print("img%d PSNR: %.2f , noisy PSNR: %.2f" % (i + 1, psnr, psnr1))
            psnr_sum += psnr

            cv2.imwrite('./data/denoised/%04d.png'%(i),out1[0,0]*255.0)

        avg_psnr = psnr_sum / len(eval_files)
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
        avg_psnr2 = psnr_sum2 / (len(eval_files)-1)
        print("--- Test ---- Average temporal PSNR %.2f ---" % avg_psnr2)

    
class dataset(object):
  def __init__(self,sess):
    self.sess = sess
    seed = time.time()
    random.seed(seed)

    random.shuffle(ind)
    
    filenames = list()
    for i in xrange(len(filepaths)):
        filenames.append(filepaths_noisy[ind[i]])
        filenames.append(filepaths[ind[i]])

    # Parameters
    num_patches = 8   # number of patches to extract from each image
    patch_size = 64                 # size of the patches
    num_parallel_calls = 1          # number of threads
    batch_size = 32                # size of the batch
    get_patches_fn = lambda image: get_patches(image, num_patches=num_patches, patch_size=patch_size)
    dataset = (
        tf.data.Dataset.from_tensor_slices(filenames)
        .map(im_read, num_parallel_calls=num_parallel_calls)
        .map(get_patches_fn, num_parallel_calls=num_parallel_calls)
        .batch(batch_size)
        .prefetch(batch_size)
    )
    
    iterator = dataset.make_one_shot_iterator()
    self.iter = iterator.get_next()
  

    def get_batch(self):
        res = self.sess.run(self.iter)
        return res
        
def im_read(filename):
    """Decode the png image from the filename and convert to [0, 1]."""
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image
    
def get_patches(image, num_patches=128, patch_size=64):
    """Get `num_patches` from the image"""
    patches = []
    for i in range(num_patches):
      point1 = random.randint(0,116) # 116 comes from the image source size (180) - the patch dimension (64)
      point2 = random.randint(0,116)
      patch = tf.image.crop_to_bounding_box(image, point1, point2, patch_size, patch_size)
      patches.append(patch)
    patches = tf.stack(patches)
    assert patches.get_shape().dims == [num_patches, patch_size, patch_size, 3]
    return patches
    
def cal_psnr(im1, im2): # PSNR function for 0-255 values
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr
    
def psnr_scaled(im1, im2): # PSNR function for 0-1 values
    mse = ((im1 - im2) ** 2).mean()
    mse = mse * (255 ** 2)
    psnr = 10 * np.log10(255 **2 / mse)
    return psnr