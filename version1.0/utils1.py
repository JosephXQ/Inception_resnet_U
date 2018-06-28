# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:24:48 2018

@author: Joseph
"""

from six.moves import xrange
import random
import glob
import dicom
import argparse
import numpy as np
import os, sys
from PIL import Image
import PIL
import tensorflow as tf
import pylab
parser = argparse.ArgumentParser(description='')
parser.add_argument('--lowsrc_dir', dest='lowsrc_dir', default='./data/train/nmu/nmu_60angles_128patches', help='dir of lowdata')
parser.add_argument('--highsrc_dir', dest='highsrc_dir', default='./data/train/nmu/nmu_1200angles_128patches', help='dir of highdata')
args = parser.parse_args()
mean =12.0#613.0vphantom#12.0nmu#9.3body #2400.0head
var = 13.0#289.0vphantom#13.0nmu#10.3body #3200.0head
high_mean = 10000
high_var = 11500
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
        return np.rot90(image, k=3)

def load_train_data(filepath='./data'):
    #shuffle img
    Filepath_init = glob.glob(args.highsrc_dir + '/*.dcm')
    print(len(Filepath_init))
    index_arr = np.arange(len(Filepath_init))
    index_arr2 = np.arange(8)
    np.random.shuffle(index_arr)
    np.random.shuffle(index_arr2)
    print("[*] Loading data...")
    #lowdata
    lowfilepath = glob.glob(args.lowsrc_dir + '/*.dcm')
    lowinputs = []
    #highdata
    highfilepath = glob.glob(args.highsrc_dir + '/*.dcm')
    compareimginput = []
    highimginput = []
    #preprocess images and add them into lists
    for i in xrange(360):
        highimg = dicom.read_file(highfilepath[index_arr[i]])
        lowimg = dicom.read_file(lowfilepath[index_arr[i]])
        highimg2 = (highimg.pixel_array).astype(np.single)
        lowimg2 = (lowimg.pixel_array).astype(np.single)
        compareimg = lowimg2 - highimg2
        lowimg1 = (lowimg2-mean)/var
        highimg1 = (highimg2-mean)/var
        compareimg = compareimg/var
        #data_augment and add_noise
        lowimg1 = data_augmentation(lowimg1,index_arr2[i%8])
        compareimg = data_augmentation(compareimg,index_arr2[i%8])
        highimg1 = data_augmentation(highimg1,index_arr2[i%8])
        #get_data and add_noise
        lowinputs.append(np.array(lowimg1).astype(np.single))#.reshape(lowimg.Rows,lowimg.Columns))
        compareimginput.append(np.array(compareimg).astype(np.single))#.reshape( highimg.Rows, highimg.Columns))
        highimginput.append(np.array(highimg1).astype(np.single))
    #send epochsamplenum images for train
    print("[*] Load successfully...")
    return lowinputs,highimginput

def load_images(filelist):
    print("test loading...")
    data = []
    count = 0
    for file in filelist:
        im = dicom.read_file(file)
        im1 = (im.pixel_array).astype(np.single)
        #zeros the negative value
        #im1 = add_noise(im1,1.0,sess=tf.Session())
        #for n, val in enumerate(im1.flat):
            #if val<0:
                #im1.pixel_array.flat[n] = 0
            #if val>17000 and val<30000:
                #im.pixel_array.flat[n] = im.pixel_array.flat[n]
        im2 = (im1-mean)/var
        data.append(np.array(im2).reshape(1, im.Rows, im.Columns, 1))
    print(len(data))
    print("load success")
    return data
def test_save_images(noisy_image, predicted_noise, clean_image1, filepath, idx):
    _, im_h, im_w, _ = noisy_image.shape
    print("test saving!!!!")
    clean_image = clean_image1.reshape([im_h,im_w])
    clean_image = clean_image*var+mean
    noisy_image = noisy_image.reshape((im_h, im_w))
    noisy_image = noisy_image*var+mean
    predicted_noise = predicted_noise.reshape((im_h,im_w))
    predicted_noise = predicted_noise*var+mean
    test = predicted_noise-noisy_image
    pylab.imshow(noisy_image)
    pylab.show("noisy image")
    pylab.imshow(clean_image)
    pylab.show("clean image")
    pylab.imshow(predicted_noise)
    pylab.show("predicted noise")
    #clipp the final result
    for n,val in enumerate(clean_image.flat):
        if val < 0:
            clean_image.flat[n] = 0
        if val >65534 :
            clean_image.flat[n] = 65534
    for n,val in enumerate(predicted_noise.flat):
        if val < 0:
            predicted_noise.flat[n] = 0
        if val > 65534:
            predicted_noise.flat[n] = 65534
    print("ojbk a !")
    #im_new = Image.fromarray(clean_image*255/65535)
    #if im_new.mode != 'RGB':
        #im_new = im_new.convert('RGB')
    #im_new.save('./data/test/test_save/im_test.bmp')
    Filepath = glob.glob('./data/test/test_after_zeros/*.dcm')
    im = dicom.read_file(Filepath[idx])
    for n,val in enumerate(im.pixel_array.flat):
        im.pixel_array.flat[n]=clean_image.flat[n]
    im.PixelData = im.pixel_array.tostring()
    im.save_as('./data/test/test_save/tested%d.dcm' % (idx+2))
    im2 = dicom.read_file(Filepath[idx])
    for n,val in enumerate(im2.pixel_array.flat):
        im2.pixel_array.flat[n]=predicted_noise.flat[n]
    im2.PixelData = im2.pixel_array.tostring()
    im2.save_as('./data/test/test_save/tested%d.dcm' % (idx+3))
    print("test saving successfully!.............")

def cal_psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).sum() / (im1.shape[0] * im1.shape[1])
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr

def add_noise(data, sigma, sess):
	noise = sigma / 255.0 * sess.run(tf.truncated_normal(data.shape))
	return (data + noise)


