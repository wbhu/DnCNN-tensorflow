import argparse
from glob import glob
import tensorflow as tf
import time
from model import denoiser
import os
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./data/denoised', help='denoised sample are saved here')
parser.add_argument('--temporal', dest='temporal', type=bool, default=False, help='Activate the temporal denoising for video sequences')
args = parser.parse_args()


def denoiser_train(denoiser, lr):
        noisy_eval_files = glob('./data/train/noisy/*.png')
        noisy_eval_files = sorted(noisy_eval_files)
        eval_files = glob('./data/train/original/*.png')
        eval_files = sorted(eval_files)
        denoiser.train(eval_files, noisy_eval_files, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr,
                       sample_dir=args.sample_dir)


def denoiser_test(denoiser):

    noisy_eval_files = glob('./data/test/noisy/*.png')
#    n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.png')[0], noisy_eval_files)]
#    noisy_eval_files = [x for (y, x) in sorted(zip(n, noisy_eval_files))]
    noisy_eval_files = sorted(noisy_eval_files)
    eval_files = glob('./data/test/original/*.png')
    eval_files = sorted(eval_files)
    start = time.time()
    denoiser.test(eval_files, noisy_eval_files, ckpt_dir=args.ckpt_dir, save_dir='./data/denoised', temporal=args.temporal)
    end = time.time()
    print "Elapsed time:", end-start

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[30:] = lr[0] / 10.0
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)

def test():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      model = denoiser(sess)
      denoiser_test(model)

  
if __name__ == '__main__':
    tf.app.run()
