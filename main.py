import argparse
import os
import numpy as np
from model import DnCNN
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=bool, default=True, help='gpu flag')
parser.add_argument('--sigma', dest='sigma', type=int, default=25, help='noise level')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
args = parser.parse_args()


def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = DnCNN(sess, sigma=args.sigma, lr=args.lr, epoch=args.epoch,
                          batch_size=args.batch_size)
            if args.phase == 'train':
                model.train()
            else:
                model.test()
    
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = DnCNN(sess, sigma=args.sigma, lr=args.lr,
                          epoch=args.epoch, batch_size=args.batch_size)
            if args.phase == 'train':
                model.train()
            else:
                model.test()


if __name__ == '__main__':
    tf.app.run()
