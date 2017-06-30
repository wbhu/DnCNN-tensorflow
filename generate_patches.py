import argparse
import glob
import os
from PIL import Image
import PIL
import math
import numpy as np 
import random
from utils import *

# macro
DATA_AUG_TIMES = 1 # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/Train400', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=40, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=30, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
# check output arguments
parser.add_argument('--from_file', dest='from_file', default="./data/img_clean_pats.npy", help='get pic from file')
parser.add_argument('--num_pic', dest='num_pic', type=int, default=10, help='number of pic to pick')

args = parser.parse_args()

def generate_patches(isDebug = False):
	global DATA_AUG_TIMES
	count = 0
	filepaths = glob.glob(args.src_dir + '/*.png')
	if isDebug:
		filepaths = filepaths[:10]
	print "number of training data %d" % len(filepaths)

	scales = [1, 0.9, 0.8, 0.7]

	size_dic = {}
	# calculate the number of patches
	for i in xrange(len(filepaths)):
		img = Image.open(filepaths[i]).convert('L') # convert RGB to gray
		for s in xrange(len(scales)):
			newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
			img_s = img.resize(newsize, resample=PIL.Image.BICUBIC) # do not change the original img
			im_h, im_w = img_s.size
			for x in range(0 + args.step, (im_h - args.pat_size + 2), args.stride):
				for y in range(0 + args.step, (im_w - args.pat_size + 2), args.stride):
					count += 1
	origin_patch_num = count * DATA_AUG_TIMES
	
	if origin_patch_num % args.bat_size != 0:
		# if the final batch is not complete, make it complete
		# totaly (origin_patch_num/args.bat_size + 1) patches
		numPatches = (origin_patch_num / args.bat_size + 1) * args.bat_size
	else:
		numPatches = origin_patch_num
	#numPatches = math.ceil(count/args.bat_size) * args.bat_size * 8
	print "total patches = %d , batch size = %d, total batches = %d" % \
			(numPatches, args.bat_size, numPatches/args.bat_size)

	# data matrix 4-D
	# inputs = np.zeros((args.pat_size, args.pat_size, 1, numPatches), dtype="uint8")
	inputs = np.zeros(( numPatches,args.pat_size, args.pat_size, 1), dtype="uint8")

	count = 0
	# generate patches
	for i in xrange(len(filepaths)):
		img = Image.open(filepaths[i]).convert('L')
		for s in xrange(len(scales)):
			newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
			#print newsize
			img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
			img_s = np.reshape(np.array(img_s, dtype = "uint8"), (img_s.size[0], img_s.size[1], 1)) # extend one dimension

			for j in xrange(DATA_AUG_TIMES):
				im_h, im_w, _ = img_s.shape
				for x in range(0 + args.step, im_h - args.pat_size + 2, args.stride):
					for y in range(0 + args.step, im_w - args.pat_size + 2, args.stride):
						inputs[ count,:, :, :] = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, :], \
						random.randint(0, 7))
						count += 1
	# pad the batch
	if count < numPatches:
		to_pad = numPatches - count
		inputs[ -to_pad:,:, :, :] = inputs[:to_pad,:, :, :]

	inputs = inputs / 255.0 # normalize to [0, 1]

	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)
	np.save(os.path.join(args.save_dir, "img_clean_pats"), inputs)
	print "size of inputs tensor = " + str(inputs.shape)

# '''
# get_pictures : to get the patch in the generated file to check
# '''
def get_pictures():
	if not os.path.exists(args.from_file):
		print "no such file"
		return

	inputs = np.load(args.from_file)
	inputs = inputs[:args.num_pic,:, :, :]
	for i in xrange(args.num_pic):
		im = 255 * inputs[i,:, :, 0] # rescale to [0,255]
		im = Image.fromarray(im).convert('RGB')
		im.show()
		raw_input("Press Enter to continue...")
		# im.save(os.path.join(args.save_dir, "%d.png" % i), "png")


if __name__ == '__main__':
	generate_patches()
	# get_pictures()