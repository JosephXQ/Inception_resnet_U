# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:27:06 2018

@author: Joseph
"""
import tensorflow as tf
import numpy as np
import U_net
import glob.glob
from six.moves import xrange


batch_size = 32
lr_init = 0.001
epoch_num = 80
epoch_decay = 20
weight_decay = 0.99
train_dir_low = ''
train_dir_high = ''
test_dir = ''
valid_dir = ''


def conv_layer(input,weight_shapes,stride_mode):
    with tf.variable_scope('conv'):
        W = tf.get_variable('weights',weight_shapes,initializer = tf.contrib.layers.variance_scaling_initializer()\
                            ,regularizer = tf.contrib.layers.l2_regularizer(scale= weight_decay),dtype = np.single)
        output = tf.nn.relu(tf.nn.conv2d(input,W,stride_mode,padding = 'SAME'))
    return output
    
def up_conv_layer(input,weight_shapes,output_shape,stride_mode):
    with tf.variable_scope('up_conv'):
        W = tf.get_variable()
    return tf.nn.conv2d_transpose(input,W,output_shape,[1,2,2,1],padding = 'SAME')#output_shape should be %2==0
    
def max_pool(input):
    return tf.nn.max_pool(input,[1,2,2,1],[1,2,2,1],padding = 'VALID')
    
def dense_block(input,name):
    shape = input.get_shape().as_list()
    input_channel = shape[3]
    with tf.variable_scope(name):
        output = bn_relu_conv(input)
        output = tf.concat([output,input],3)
    
def load_images(filepath='./data'):
        #shuffle img
    Filepath_init = glob.glob(train_dir_low + '/*.dcm')
    index_arr = np.arange(len(Filepath_init))
    np.random.shuffle(index_arr)
    print("[*] Loading data...")
    #lowdata
    lowfilepath = glob.glob(train_dir_low + '/*.dcm')
    lowinputs1 = []
    lowinputs = []
    #highdata
    highfilepath = glob.glob(train_dir_high + '/*.dcm')
    compareimginput = []
    compareimginput1 = []
    #preprocess images and add them into lists
    for i in xrange(len(lowfilepath)):
        highimg = dicom.read_file(highfilepath[i])
        lowimg = dicom.read_file(lowfilepath[i])
        highimg2 = (highimg.pixel_array).astype(np.single)
        lowimg2 = (lowimg.pixel_array).astype(np.single)
        compareimg = lowimg2 - highimg2
        lowimg1 = (lowimg2-mean)/var
        compareimg = compareimg/var
        lowinputs1.append(np.array(lowimg1).astype(np.single))#.reshape(lowimg.Rows,lowimg.Columns))
        lowinputs.append(np.array(lowimg1).astype(np.single))#.reshape(lowimg.Rows,lowimg.Columns))
        compareimginput1.append(np.array(compareimg).astype(np.single))#.reshape(highimg.Rows, highimg.Columns))
        compareimginput.append(np.array(compareimg).astype(np.single))#.reshape( highimg.Rows, highimg.Columns))
    #send epochsamplenum images for train
    for j in xrange(480):
        lowinputs[j]= lowinputs1[index_arr[j]]
        compareimginput[j] = compareimginput1[index_arr[j]]
    print("[*] Load successfully...")
    return lowinputs,compareimginput

def load_test_images(filepath=''):
    print("test loading...")
    data = []
    count = 0
    for file in filelist:
        im = dicom.read_file(file)
        im1 = (im.pixel_array).astype(np.single)
        #zeros the negative value
        for n, val in enumerate(im1.flat):
            if val<0:
                im1.pixel_array.flat[n] = 0
            #if val>17000 and val<30000:
                #im.pixel_array.flat[n] = im.pixel_array.flat[n]
        im2 = (im1-mean)/(var)
        data.append(np.array(im2).reshape(1, im.Rows, im.Columns, 1))
    print(len(data))
    print("load success")
    return data