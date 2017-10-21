import numpy as np
import os, sys
from PIL import Image
import tensorflow as tf


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def load_data(filepath='./data/image_clean_pat.npy'):
    assert '.npy' in filepath
    if not os.path.exists(filepath):
        print("[!] Data file not exists")
        sys.exit(1)
    
    print("[*] Loading data...")
    data = np.load(filepath)
    np.random.shuffle(data)
    print("[*] Load successfully...")
    return data


def add_noise(data, sigma, sess):
    # target image pixel value range is 0-1
    noise = sigma / 255.0 * sess.run(tf.truncated_normal(data.shape))
    return (data + noise)


def load_images(filelist):
    # pixel value range 0-255
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[0], im.size[1], 1))
    return data


def save_images(ground_truth, noisy_image, clean_image, filepath):
    # assert the pixel value range is 0-255
    _, im_h, im_w, _ = noisy_image.shape
    ground_truth = ground_truth.reshape((im_h, im_w))
    noisy_image = noisy_image.reshape((im_h, im_w))
    clean_image = clean_image.reshape((im_h, im_w))
    cat_image = np.column_stack((noisy_image, clean_image))
    cat_image = np.column_stack((ground_truth, cat_image))
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'png')


def cal_psnr(im1, im2):
    # assert pixel value range is 0-255
    mse = (np.abs(im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr
