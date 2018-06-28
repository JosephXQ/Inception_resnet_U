# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:16:36 2018

@author: Joseph
"""

''' resFCN Dn net train'''
import tensorflow as tf
import numpy as np
import math, os
from glob import glob
from ops import *
from utils1 import *
from six.moves import xrange
import time
import matplotlib.pylab as plt


class DnCNN(object):
    def __init__(self, sess, image_size=128, batch_size=48, decay_epoch=30,
                 output_size=128, input_c_dim=1, output_c_dim=1, clip_b=0.025, lr=0.001, epoch=180,
                 ckpt_dir='./checkpoint', sample_dir='./sample', test_save_dir='./data/test/saved',
                 dataset='trainmodel_U+Inception-resnet_45-720_16b_headmse+L10.1', testset='test'):
        self.sess = sess
        self.is_gray = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_size = output_size
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.clip_b = clip_b
        self.lr = lr
        self.numEpoch = epoch
        self.ckpt_dir = ckpt_dir
        self.trainset = dataset
        self.testset = testset
        self.sample_dir = sample_dir
        self.test_save_dir = test_save_dir
        self.epoch = epoch
        self.decay_epoch = decay_epoch
        self.save_every_iter = 170000
        self.eval_every_iter = 100000
        # Adam setting (default setting)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.Balance = 0.00000005
        W1_array = np.array([1., 0., 0.])
        W1 = np.zeros([1, 3, 1, 1], np.single)
        W1[:, :, 0, 0] = W1_array
        W2_array = np.array([0., 0., 1.])
        W2 = np.zeros([1, 3, 1, 1], np.single)
        W2[:, :, 0, 0] = W2_array
        W3_array = np.array([[1.], [0.], [0.]])
        W3 = np.zeros([3, 1, 1, 1], np.single)
        W3[:, :, 0, 0] = W3_array
        W4_array = np.array([[0.], [0.], [1.]])
        W4 = np.zeros([3, 1, 1, 1], np.single)
        W4[:, :, 0, 0] = W4_array
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.W4 = W4
        self.build_model()

    def build_model(self):
        # input : [batchsize, image_size, image_size, channel]
        self.isTraining = tf.placeholder(tf.bool, name='phase_train')
        self.X = tf.placeholder(np.single, [None, self.image_size, self.image_size, self.input_c_dim],
                                name='noisy_image')
        self.X_ = tf.placeholder(np.single, [None, self.image_size, self.image_size, self.input_c_dim],
                                 name='clean_image')
        # self.a = tf.placeholder(np.single,[11,11,11,1],name='ssim_param')
        # layer 1
        with tf.variable_scope('layer1'):
            layer_1_output = self.conv_layer(self.X, [5, 5, self.input_c_dim, 64], b_init=0.0, stridemode=[1, 1, 1, 1])
            print('layer2 builded')
        with tf.variable_scope('layer2'):
            layer_2_output = self.bn_relu_conv_layer(layer_1_output, [3, 3, 64, 64])
        # layer 2 to 16
        #with tf.variable_scope('layer3'):
            #layer_3_output = self.bn_relu_conv_layer(layer_2_output, [3, 3, 64, 64])
        with tf.variable_scope('layer4'):
            layer_4_output = self.resblock_layer(layer_2_output, [3, 3, 64, 64], first_block=True)
        with tf.variable_scope('layer5'):
            layer_5_output = self.resblock_layer(layer_4_output, [3, 3, 64, 64])  # +layer_2_output_temp1
        with tf.variable_scope('layer6'):
            layer_6_output = self.max_pool(layer_5_output)
        with tf.variable_scope('layer7'):
            layer_7_output = self.bn_conv(layer_6_output,[3,3,64,128])
        with tf.variable_scope('layer8'):
            layer_8_output = self.Inception_resnetA_layer(layer_7_output, [3, 3, 128, 128])  # +layer_2_output_temp2
        with tf.variable_scope('layer9'):
            layer_9_output = self.max_pool(layer_8_output)
        with tf.variable_scope('layer10'):
            layer_10_output = self.bn_conv(layer_9_output,[3,3,128,256])
        with tf.variable_scope('layer11'):
            layer_11_output = self.Inception_resnetB_layer(layer_10_output, [3, 3, 256, 256])
        with tf.variable_scope('layer12'):
            layer_12_output_temp = self.unpool(layer_11_output)
            layer_12_output = tf.concat([layer_12_output_temp,layer_8_output],3,name = 'concat')
        with tf.variable_scope('layer13'):
            layer_13_output = self.bn_conv(layer_12_output,[3,3,384,128])
        with tf.variable_scope('layer14'):
            layer_14_output = self.Inception_resnetC_layer(layer_13_output, [3, 3, 128, 128])
        with tf.variable_scope('layer15'):
            layer_15_output_temp = self.unpool(layer_14_output)
            layer_15_output = tf.concat([layer_15_output_temp, layer_5_output], 3, name='concat')#mark
        with tf.variable_scope('layer16'):
            layer_16_output = self.bn_conv(layer_15_output, [3, 3, 192, 64])
        with tf.variable_scope('layer17'):
            layer_17_output = self.resblock_layer(layer_16_output, [3, 3, 64, 64])
        with tf.variable_scope('layer18'):
            layer_18_output = self.resblock_layer(layer_17_output, [3, 3, 64, 64])
        with tf.variable_scope('layer19'):
            self.Y = self.bn_relu_conv_layer(layer_18_output, [3, 3, 64, self.output_c_dim])
        # MSE loss
        self.Y_ = self.X - self.X_  # noisy = noisy image - clean image
        mse_loss = (1.0 / self.batch_size) * tf.reduce_mean(tf.square(self.Y - self.Y_))  # +(1-self.tf_ssim(self.Y,self.Y_))*0.001
        # self.op = tf.Print(mse_loss,[mse_loss],"mse_loss:")
        qmse_loss = (1.0 / self.batch_size) * tf.reduce_mean(tf.abs(self.Y - self.Y_))
        ssim_loss_image = self.tf_ssim(self.X - self.Y, self.X_) * (1.0 / self.batch_size)
        ssim_loss_noise = self.tf_ssim(self.Y, self.X - self.X_) * (1.0 / self.batch_size)
        # mse_loss_branch = (1.0/self.batch_size)*tf.reduce_mean(tf.square(layer_13_output-self.Y_))
        # grad_loss1 = (1.0/self.batch_size) * self.tf_gradloss(self.X-self.Y)
        # L1_loss = (1.0/self.batch_size)*self.L_loss()
        # L2_loss = (1.0/self.batch_size)*self.tf_L2_loss(self.X-self.Y)
        # self.loss = tf.py_func(self.grad_loss, [self.X-self.Y], tf.float32, stateful=False, name='my_func')#+mse_loss
        self.loss = mse_loss+0.000001 * 0.1 * self.tf_l1_loss(self.X - self.Y)
        # self.loss = 1-tf_ssim(self.Y,self.Y_)
        # params =  tf.trainable_variables()
        # grad_loss = (1.0 / self.batch_size) * ( self.Balance * tf.reduce_mean(self.gradients))
        #tf.add_to_collection('losses', mse_loss)
        #self.loss = tf.add_n(tf.get_collection('losses'))#+mse_loss
        print("results of each layer!!!!..................")
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_step = optimizer.minimize(self.loss)
        # create this init op after all variables specified
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        print("[*] Initialize model successfully...")

    def tf_gradloss(self, x_input):
        res = tf.reduce_sum(tf.sqrt(
            tf.square(self.con2d(x_input, self.W1) - self.con2d(x_input, self.W2)) + 0.00000001)) + tf.reduce_sum(
            tf.sqrt(tf.square( self.con2d(x_input, self.W3) - self.con2d(x_input, self.W4)) + 0.00000001))
        return res

    def tf_l1_loss(self, x_input):
        res = tf.reduce_sum(tf.abs(self.con2d(x_input, self.W1) - self.con2d(x_input, self.W2))) + tf.reduce_sum(tf.abs( \
            self.con2d(x_input, self.W3) - self.con2d(x_input, self.W4)))
        return res

    def tf_L2_loss(self, x_input):
        res = tf.nn.l2_loss(self.con2d(x_input, self.W1) - self.con2d(x_input, self.W2)) + \
              tf.nn.l2_loss(self.con2d(x_input, self.W3) - self.con2d(x_input, self.W4))
        res2 = tf.reduce_sum(tf.square(self.con2d(x_input, self.W1) - self.con2d(x_input, self.W2))) / 2.0 + \
               tf.reduce_sum(tf.square(self.con2d(x_input, self.W3) - self.con2d(x_input, self.W4))) / 2.0
        return res

    def grad_loss(self, x_test):
        res1 = np.gradient(x_test, axis=1)
        res2 = np.gradient(x_test, axis=2)
        return np.mean([np.square(res1), np.square(res2)])
        # return res1

    def L_loss(self, input):
        print("gradient:", input.shape)
        res1 = np.linalg.norm(input, ord=1, axis=1)
        res2 = np.linalg.norm(input, ord=1, axis=2)
        res = np.mean([np.mean(res1), np.mean(res2)])
        return res

    def unpool(self,inputs):
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])

    def con2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

    def max_pool(self, input):
        return tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    def bn_conv(self,input,weightshape):
        temp = self.bn_layer(input,weightshape[-2])
        output = self.conv_layer(tf.nn.relu(temp),weightshape,b_init=0,stridemode=[1,1,1,1])
        return output

    def _tf_fspecial_gauss(self, size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        # x = tf.constant(x_data,dtype=np.single)
        # y = tf.constant(y_data,dtype=np.single)

        g = np.exp(-((x_data ** 2 + y_data ** 2) / (2.0 * sigma ** 2)))
        return g / np.sum(g)

    def tf_ssim(self, img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
        window = self._tf_fspecial_gauss(size, sigma)  # window shape [size, size]
        K1 = 0.01
        K2 = 0.03
        L = 1  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
        if cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                  (sigma1_sq + sigma2_sq + C2)),
                     (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                 (sigma1_sq + sigma2_sq + C2))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value

    def conv_layer(self, inputdata, weightshape, b_init, stridemode):

        with tf.variable_scope(tf.get_variable_scope()):
            W = tf.get_variable('weights', weightshape, initializer= \
                # tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
            tf.contrib.layers.variance_scaling_initializer(), dtype=np.single)
            b = tf.get_variable('biases', [1, weightshape[-1]], initializer= \
                tf.constant_initializer(0.0), dtype=tf.float32)
            #tf.add_to_collection('W', W)
            tf.add_to_collection('losses', 0.0000001*tf.contrib.layers.l1_regularizer(0.1)(W))
            #self.weight = tf.Print(W, [W], "weight:")
        return tf.nn.conv2d(inputdata, W, strides=stridemode, padding="SAME") + b  # SAME with zero padding

    def bn_layer(self, logits, output_dim, b_init=0.0):

        alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer= \
            tf.constant_initializer(get_bn_weights([1, output_dim], self.clip_b, self.sess)), dtype=np.single,
                                trainable=True)
        beta = tf.get_variable('bn_beta', [1, output_dim], initializer= \
            tf.constant_initializer(b_init), dtype=np.single, trainable=True)
        return batch_normalization(logits, alpha, beta, isTraining=self.isTraining)

    def layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True):

        logits = self.conv_layer(inputdata, filter_shape, b_init, stridemode)
        if useBN:
            output = tf.nn.relu(self.bn_layer(logits, filter_shape[-1]))
        else:
            output = tf.nn.relu(logits)
        return output

    def bn_relu_conv_layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1]):

        input_data = tf.nn.relu(self.bn_layer(inputdata, filter_shape[-1]))
        output = self.conv_layer(input_data, filter_shape, b_init, stridemode)
        return output

    def resblock_layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True,
                       first_block=False):
        with tf.variable_scope('conv1_in_resblock'):
            if first_block:
                input1_data = self.conv_layer(inputdata, filter_shape, b_init, stridemode)
            else:
                input1_data = self.bn_relu_conv_layer(inputdata, filter_shape, b_init, stridemode)
            with tf.variable_scope('subconv_in_conv1'):
                input2_data = self.bn_relu_conv_layer(input1_data, filter_shape, b_init, stridemode)
        with tf.variable_scope('conv2_in_resblock'):
            if first_block:
                branchinput_data = self.conv_layer(inputdata, [1, 1, filter_shape[-2], filter_shape[-1]], b_init,stridemode)
            else:
                branchinput_data = inputdata
        output = input2_data + branchinput_data
        return output

    def Inception_resnetA_layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True):
        with tf.variable_scope('conv1_in_Inception_resnetA'):
            branchinput_data1 = self.Inception_in_resnetA_layer(inputdata, filter_shape, b_init, stridemode)
            with tf.variable_scope('subconv_in_RNA'):
                branchinput_data = self.bn_relu_conv_layer(branchinput_data1, filter_shape, b_init, stridemode)
        output = inputdata + branchinput_data
        return output

    def Inception_resnetB_layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True):
        with tf.variable_scope('conv1_in_Inception_resnetB'):
            branchinput_data1 = self.Inception_in_resnetB_layer(inputdata, filter_shape, b_init, stridemode)
            with tf.variable_scope('subconv_in_RNB'):
                branchinput_data = self.bn_relu_conv_layer(branchinput_data1, filter_shape, b_init, stridemode)
        output = inputdata + branchinput_data
        return output

    def Inception_resnetC_layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True):
        with tf.variable_scope('conv1_in_Inception_resnetC'):
            branchinput_data1 = self.Inception_in_resnetC_layer(inputdata, filter_shape, b_init, stridemode)
            with tf.variable_scope('subconv_in_RNC'):
                branchinput_data = self.bn_relu_conv_layer(branchinput_data1, filter_shape, b_init, stridemode)
        output = inputdata + branchinput_data
        return output

    def Inception_layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True):
        with tf.variable_scope('conv1_in_Inception'):
            input1_data1 = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 21], b_init, stridemode)
            with tf.variable_scope('subconv1_in_Inconv1'):
                input1_data = self.bn_relu_conv_layer(input1_data1, [3, 3, 21, 21], b_init, stridemode)
        with tf.variable_scope('conv2_in_Inception'):
            input2_data1 = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 21], b_init, stridemode)
            with tf.variable_scope('subconv2_in_Inconv2'):
                input2_data2 = self.bn_relu_conv_layer(input2_data1, [1, 3, 21, 21], b_init, stridemode)
                with tf.variable_scope('subconv3_in_Inconv2'):
                    input2_data = self.conv_layer(input2_data2, [3, 1, 21, 21], b_init, stridemode)
        with tf.variable_scope('conv3_in_Inception'):
            input3_data1 = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 22], b_init, stridemode)
            with tf.variable_scope('subconv2_in_Inconv3'):
                input3_data2 = self.bn_relu_conv_layer(input3_data1, [3, 3, 22, 22], b_init, stridemode)
                with tf.variable_scope('subconv3_in_Inconv3'):
                    input3_data = self.bn_relu_conv_layer(input3_data2, [3, 3, 22, 22], b_init, stridemode)
        output = tf.concat([input1_data, input2_data, input3_data], 3, name='concat')
        return output

    def Inception_in_resnetA_layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True):
        with tf.variable_scope('conv1_in_Inception_in_A'):
            input1_data = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 64], b_init, stridemode)
        with tf.variable_scope('conv2_in_Inception_in_A'):
            input2_data1 = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 32], b_init, stridemode)
            with tf.variable_scope('subconv1_in_Inconv2'):
                input2_data2 = self.bn_relu_conv_layer(input2_data1, [1, 3, 32, 32], b_init, stridemode)
                with tf.variable_scope('subconv2_in_Inconv2'):
                    input2_data = self.bn_relu_conv_layer(input2_data2, [3, 1, 32, 32], b_init, stridemode)
        with tf.variable_scope('conv3_in_Inception_in_A'):
            input3_data1 = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 32], b_init, stridemode)
            with tf.variable_scope('subconv1_in_Inconv3'):
                input3_data2 = self.bn_relu_conv_layer(input3_data1, [3, 3, 32, 32], b_init, stridemode)
                with tf.variable_scope('subconv2_in_Inconv3'):
                    input3_data = self.bn_relu_conv_layer(input3_data2, [3, 3, 32, 32], b_init, stridemode)
        output = tf.concat([input1_data, input2_data, input3_data], 3, name='concat')
        return output

    def Inception_in_resnetB_layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1]):
        with tf.variable_scope('conv1_in_Inception_in_B'):
            input1_data = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 128], b_init, stridemode)
        with tf.variable_scope('conv2_in_Inception_in_B'):
            input2_data1 = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 128], b_init, stridemode)
            with tf.variable_scope('subconv1_in_Inconv2'):
                input2_data2 = self.bn_relu_conv_layer(input2_data1, [1, 3, 128, 128], b_init, stridemode)
                with tf.variable_scope('subconv2_in_Inconv2'):
                    input2_data = self.bn_relu_conv_layer(input2_data2, [3, 1, 128, 128], b_init, stridemode)
        output = tf.concat([input2_data, input1_data], 3, name='concat')
        return output

    def Inception_in_resnetC_layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1]):
        with tf.variable_scope('conv1_in_Inception_in_C'):
            input1_data = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 64], b_init, stridemode)
        with tf.variable_scope('conv2_in_Inception_in_C'):
            input2_data1 = self.conv_layer(inputdata, [1, 1, filter_shape[-2], 64], b_init, stridemode)
            with tf.variable_scope('subconv1_in_Inconv2'):
                input2_data2 = self.bn_relu_conv_layer(input2_data1, [1, 7, 64, 64], b_init, stridemode)
                with tf.variable_scope('subconv2_in_Inconv2'):
                    input2_data = self.bn_relu_conv_layer(input2_data2, [7, 1, 64, 64], b_init, stridemode)
        output = tf.concat([input2_data, input1_data], 3, name='concat')
        return output

    def init3(size=11, sigma=1.5):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=0)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        # x = tf.constant(x_data, dtype=tf.float32)
        # y = tf.constant(y_data, dtype=tf.float32)

        g = np.exp(-((x_data ** 2 + y_data ** 2) / (2.0 * sigma ** 2)))
        return g / np.sum(g)

    def train(self):
        self.sess.run(self.init)
        epochsamplenum = 360
        eval_files = glob.glob('./data/eval/test_use/*.dcm')
        eval_data = load_images(eval_files)  # list of array of different size, 4-D
        batch_images = np.zeros((self.batch_size, 128, 128, 1), dtype=np.single)
        train_images = np.zeros((self.batch_size, 128, 128, 1), dtype=np.single)
        numBatch = int(epochsamplenum // self.batch_size)
        counter = 0
        print("[*] Start training : ")
        start_time = time.time()
        # a = self.init3()

        count = 0
        loss_record = []
        for epoch in xrange(self.epoch):
            lowpatch_data, comparepatch_data = load_train_data(filepath='./data')
            print("data shape = " + str(len(lowpatch_data)))
            for batch_id in xrange(numBatch):
                self.LR = tf.Print(self.lr, [self.lr], "Curren_lr:")
                count = count + 1
                index_arr1 = np.arange(epochsamplenum)
                np.random.shuffle(index_arr1)
                j = 0
                for i in range(batch_id * self.batch_size, np.min([(batch_id + 1) * self.batch_size, epochsamplenum])):
                    batch_images[j, :, :, 0] = comparepatch_data[index_arr1[i]]
                    train_images[j, :, :, 0] = lowpatch_data[index_arr1[i]]
                    j = j + 1
                print(str(train_images.shape))
                writer = tf.summary.FileWriter("/home/seu/xq/denoise_resFCN/data/graph")
                _, loss, _ = self.sess.run([self.train_step, self.loss, self.LR], feed_dict={self.X: train_images, self.X_: batch_images,
                                                      self.isTraining: True})
                # loss_record.append(loss)
                writer.close()
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" % (epoch + 1, batch_id + 1, int(numBatch),
                                                                          time.time() - start_time, loss))
                counter += 1
                if np.mod(counter, self.eval_every_iter) == 0:
                    self.evaluate(epoch, counter, eval_data)
                    print("eval ending....")
                # save the model
                if np.mod(counter, self.save_every_iter) == 0:
                    print("saving ...............")
                    self.save(counter)
                print("save end....")
                print("learningRate equals:%s", self.lr)
            # if np.mod(epoch+1, self.decay_epoch) == 0:
            print("checking epoch num!..........................................")
            print(epoch)
            # self.lr = self.lr*0.5
            self.lr = tf.train.exponential_decay(0.001, global_step=epoch, decay_steps=30, decay_rate=0.60, name='lr',
                                                 staircase=False)
        self.save(count)
        # plt.plot(range(len(loss_record)), loss_record, 'r')
        print("[*] Finish training.")

    def save(self, count):
        model_name = "DnCNN.model"
        model_dir = "%s_%s_%s" % (self.trainset, self.batch_size, self.image_size)
        checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=count)

    def sampler(self, image):
        self.isTraining = tf.placeholder(tf.bool, name='phase_train1')
        self.X_test = tf.placeholder(np.single, [None, 512, 512, 1], name='noisy_image_test')
        print(str(self.X_test.shape))
        # layer 1 (adpat to the input image)
        with tf.variable_scope('layer1',reuse=True):
            layer_1_output = self.conv_layer(self.X_test, [5, 5, self.input_c_dim, 64], b_init=0.0, stridemode=[1, 1, 1, 1])
        with tf.variable_scope('layer2',reuse=True):
            layer_2_output = self.bn_relu_conv_layer(layer_1_output, [3, 3, 64, 64])
        # layer 2 to 16
        #with tf.variable_scope('layer3',reuse=True):
            #layer_3_output = self.bn_relu_conv_layer(layer_2_output, [3, 3, 64, 64])
        with tf.variable_scope('layer4',reuse=True):
            layer_4_output = self.resblock_layer(layer_2_output, [3, 3, 64, 64], first_block=True)
        with tf.variable_scope('layer5',reuse=True):
            layer_5_output = self.resblock_layer(layer_4_output, [3, 3, 64, 64])  # +layer_2_output_temp1
        with tf.variable_scope('layer6',reuse=True):
            layer_6_output = self.max_pool(layer_5_output)
        with tf.variable_scope('layer7',reuse=True):
            layer_7_output = self.bn_conv(layer_6_output,[3,3,64,128])
        with tf.variable_scope('layer8',reuse=True):
            layer_8_output = self.Inception_resnetA_layer(layer_7_output, [3, 3, 128, 128])  # +layer_2_output_temp2
        with tf.variable_scope('layer9',reuse=True):
            layer_9_output = self.max_pool(layer_8_output)
        with tf.variable_scope('layer10',reuse=True):
            layer_10_output = self.bn_conv(layer_9_output,[3,3,128,256])
        with tf.variable_scope('layer11',reuse=True):
            layer_11_output = self.Inception_resnetB_layer(layer_10_output, [3, 3, 256, 256])
        with tf.variable_scope('layer12',reuse=True):
            layer_12_output_temp = self.unpool(layer_11_output)
            layer_12_output = tf.concat([layer_12_output_temp,layer_8_output],3,name = 'concat')
        with tf.variable_scope('layer13',reuse=True):
            layer_13_output = self.bn_conv(layer_12_output,[3,3,384,128])
        with tf.variable_scope('layer14',reuse=True):
            layer_14_output = self.Inception_resnetC_layer(layer_13_output, [3, 3, 128, 128])
        with tf.variable_scope('layer15',reuse=True):
            layer_15_output_temp = self.unpool(layer_14_output)
            layer_15_output = tf.concat([layer_15_output_temp, layer_5_output], 3, name='concat')#mark
        with tf.variable_scope('layer16',reuse=True):
            layer_16_output = self.bn_conv(layer_15_output, [3, 3, 192, 64])
        with tf.variable_scope('layer17',reuse=True):
            layer_17_output = self.resblock_layer(layer_16_output, [3, 3, 64, 64])
        with tf.variable_scope('layer18',reuse=True):
            layer_18_output = self.resblock_layer(layer_17_output, [3, 3, 64, 64])
        with tf.variable_scope('layer19',reuse=True):
            self.Y_test = self.bn_relu_conv_layer(layer_18_output, [3, 3, 64, self.output_c_dim])
            print("layer last........")

    def load(self, checkpoint_dir):
        '''Load checkpoint file'''
        print("[*] Reading checkpoint...")
        model_dir = "%s_%s_%s" % (self.trainset, self.batch_size, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def forward(self, noisy_image):
        print("sampler starting...")
        self.sampler(noisy_image)
        return self.sess.run(self.Y_test, feed_dict={self.X_test: noisy_image, self.isTraining: False})

    def test(self):
        self.sess.run(self.init)
        test_files = glob.glob('./data/test/test_use/*.dcm')
        test_save_filepath = glob.glob('./data/test/test_save')
        # load testing input
        test_input = []
        test_data = np.zeros((1, 512, 512, 1), dtype=np.single)
        print("[*] Loading test images ...")
        test_input = load_images(test_files)  # list of array of different size
        if self.load(self.ckpt_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for idx in xrange(len(test_files)):
            test_data = test_input[idx]
            # noisy_image = test_data[idx]
            print(test_data.shape)
            # test_data = self.sess.run(self.con2d(test_data1,self.W1))
            predicted_noise = self.forward(test_data)
            # predicted_noise = self.sess.run(self.con2d(test_data1,self.W2))
            print("forwarding successfully...")
            output_clean_image = test_data - predicted_noise
            test_save_images(test_data, predicted_noise, output_clean_image, test_save_filepath, idx)

    def evaluate(self, epoch, counter, test_data):
        psnr_sum = 0
        print(str(len(test_data)))
        for idx in xrange(len(test_data)):
            noisy_image = test_data[idx]
            predicted_noise = self.forward(noisy_image)
            output_clean_image = noisy_image - predicted_noise
            groundtruth = noisy_image
            outputimage = output_clean_image
            psnr = cal_psnr(groundtruth, outputimage)
            psnr_sum += psnr
            # eval_save_images(groundtruth, noisyimage, outputimage, \
            # os.path.join(self.sample_dir, 'test%d_%d_%d.dcm' % (idx, epoch, counter)), idx)
            print("number %d eval ........................" % idx)
        avg_psnr = psnr_sum / len(test_data)
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
