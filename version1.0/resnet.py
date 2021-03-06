# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:32:21 2018

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

class DnCNN(object):
    def __init__(self, sess, image_size=128, batch_size=32, decay_epoch=30,
                 output_size=128, input_c_dim=1, output_c_dim=1,clip_b=0.025, lr=0.001, epoch=120,
                 ckpt_dir='./checkpoint', sample_dir='./sample',test_save_dir='./data/test/saved',
                 dataset='trainmodel_resnet_90-360_16b_headmse', testset='test'):
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
        self.X = tf.placeholder(np.single, [None, self.image_size, self.image_size, self.input_c_dim], name='noisy_image')
        self.X_ = tf.placeholder(np.single, [None, self.image_size, self.image_size, self.input_c_dim], name='clean_image')
        # layer 1
        with tf.variable_scope('conv1'):
            layer_1_output = self.layer(self.X, [5, 5, self.input_c_dim, 64], useBN=True)
        # layer 2 to 16
        with tf.variable_scope('conv2'):
            layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv1'):
            layer_3_output = self.resblock_layer(layer_2_output, [3, 3, 64, 64], first_block=True)
        with tf.variable_scope('resconv2'):
            layer_4_output = self.resblock_layer(layer_3_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv3'):
            layer_5_output = self.resblock_layer(layer_4_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv4'):
            layer_6_output = self.resblock_layer(layer_5_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv5'):
            layer_7_output = self.resblock_layer(layer_6_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv6'):
            layer_8_output = self.resblock_layer(layer_7_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv7'):
            layer_9_output = self.resblock_layer(layer_8_output, [3, 3, 64, 64])

        with tf.variable_scope('conv3'):
            self.Y = self.bn_relu_conv_layer(layer_9_output, [3, 3, 64, self.output_c_dim])
        # MSE loss
        self.Y_ = self.X-self.X_  # noisy = noisy image - clean image
        self.loss = (1.0 / self.batch_size) * tf.reduce_mean(tf.square(self.Y - self.Y_))#+0.000001*0.012*self.tf_gradloss(self.X-self.Y)
        print("results of each layer!!!!..................")
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_step = optimizer.minimize(self.loss)
        # create this init op after all variables specified
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        print("[*] Initialize model successfully...")


    def con2d(self,x,W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

    def tf_gradloss(self,x_input):
        res = tf.reduce_sum(tf.sqrt(tf.square(self.con2d(x_input, self.W1) - self.con2d(x_input, self.W2))+0.00000001)) + tf.reduce_sum(tf.sqrt(tf.square(\
            self.con2d(x_input, self.W3) - self.con2d(x_input, self.W4))+0.00000001))
        return res

    def unpool(self,inputs):
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])

    def max_pool(self, input):
        return tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    def bn_conv(self,input,weightshape):
        temp = self.bn_layer(input,weightshape[-2])
        output = self.conv_layer(tf.nn.relu(temp),weightshape,b_init=0,stridemode=[1,1,1,1])
        return output

    def conv_layer(self, inputdata, weightshape, b_init, stridemode):

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            W = tf.get_variable('weights', weightshape, initializer= \
                #tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
                tf.contrib.layers.variance_scaling_initializer(),dtype=np.single)
            #b = tf.get_variable('biases', [1, weightshape[-1]], initializer= \
                #tf.constant_initializer(0.0),dtype=tf.float32)
            self.weight = tf.Print(W,[W],"weight:")
        return tf.nn.conv2d(inputdata, W, strides=stridemode, padding="SAME")#+b # SAME with zero padding

    def bn_layer(self, logits, output_dim, b_init=0.0):

        alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer= \
            tf.constant_initializer(get_bn_weights([1, output_dim], self.clip_b, self.sess)),dtype=np.single,trainable=True)
        beta = tf.get_variable('bn_beta', [1, output_dim], initializer= \
            tf.constant_initializer(b_init),dtype=np.single,trainable=True)
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
                input1_data = self.bn_relu_conv_layer( inputdata, filter_shape, b_init, stridemode)
            with tf.variable_scope('subconv_in_conv1'):
                input2_data = self.bn_relu_conv_layer(input1_data, filter_shape, b_init, stridemode)
        with tf.variable_scope('conv2_in_resblock'):
            if first_block:
                branchinput_data = self.conv_layer(inputdata, [1, 1, filter_shape[-2], filter_shape[-1]], b_init, stridemode)
            else:
                branchinput_data = inputdata
        output = input2_data + branchinput_data
        return output

    def train(self):
        self.sess.run(self.init)
        epochsamplenum = 360
        eval_files = glob.glob('./data/eval/test_use/*.dcm')
        eval_data = load_images(eval_files)  # list of array of different size, 4-D
        batch_images = np.zeros((self.batch_size, 128, 128, 1),dtype=np.single)
        train_images = np.zeros((self.batch_size, 128, 128, 1),dtype=np.single)
        numBatch = int(epochsamplenum//self.batch_size)
        counter = 0
        print("[*] Start training : ")
        start_time = time.time()
        for epoch in xrange(self.epoch):
            self.print_lr = tf.Print(self.lr, [self.lr], "Curren_lr:")
            lowpatch_data, comparepatch_data = load_train_data(filepath='./data')
            print("data shape = "+str(len(lowpatch_data)))
            for batch_id in xrange(numBatch):
                index_arr1 = np.arange(epochsamplenum)
                np.random.shuffle(index_arr1)
                j = 0
                for i in range(batch_id*self.batch_size,np.min([(batch_id+1)*self.batch_size,epochsamplenum])):
                    batch_images[j, :, :, 0] = comparepatch_data[index_arr1[i]]
                    train_images[j, :, :, 0] = lowpatch_data[index_arr1[i]]
                    j = j + 1
                print(str(train_images.shape))
                _, loss, weight,_= self.sess.run([self.train_step, self.loss, self.weight,self.print_lr], \
                                        feed_dict={self.X: train_images, self.X_: batch_images, self.isTraining:True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" % (epoch + 1, batch_id + 1, int(numBatch),
                         time.time() - start_time, loss))
                counter += 1
                if np.mod(counter, self.eval_every_iter) == 0:
                    self.evaluate(epoch, counter, eval_data)
                    print("eval ending....")
                # save the model
                if np.mod(counter, self.save_every_iter) == 0:
                    print("saving ...............")
                    self.save()
                print("save end....")
                print("learningRate equals:%s",self.lr)
            if np.mod(epoch+1, self.decay_epoch) == 0:
                print("checking epoch num!..........................................")
                print(epoch)
            self.lr = tf.train.exponential_decay(0.001,global_step=epoch,decay_steps=30,decay_rate=0.6,name='lr')
        self.save()
        print("[*] Finish training.")

    def save(self):
        model_name = "DnCNN.model"
        model_dir = "%s_%s_%s" % (self.trainset, \
                                  self.batch_size, self.image_size)
        checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name))

    def sampler(self, image):
        self.isTraining = tf.placeholder(tf.bool, name='phase_train1')
        self.X_test = tf.placeholder(np.single, [None,512,512,1], name='noisy_image_test')
        print(str(self.X_test.shape))
        # layer 1 (adpat to the input image)
        with tf.variable_scope('conv1',reuse=True):
            layer_1_output = self.layer(self.X_test, [5, 5, self.input_c_dim, 64], useBN=True)
        # layer 2 to 16
        with tf.variable_scope('conv2',reuse=True):
            layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv1',reuse=True):
            layer_3_output = self.resblock_layer(layer_2_output, [3, 3, 64, 64], first_block=True)
        with tf.variable_scope('resconv2',reuse=True):
            layer_4_output = self.resblock_layer(layer_3_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv3',reuse=True):
            layer_5_output = self.resblock_layer(layer_4_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv4',reuse=True):
            layer_6_output = self.resblock_layer(layer_5_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv5',reuse=True):
            layer_7_output = self.resblock_layer(layer_6_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv6',reuse=True):
            layer_8_output = self.resblock_layer(layer_7_output, [3, 3, 64, 64])
        with tf.variable_scope('resconv7',reuse=True):
            layer_9_output = self.resblock_layer(layer_8_output, [3, 3, 64, 64])

        with tf.variable_scope('conv3',reuse=True):
            self.Y_test = self.bn_relu_conv_layer(layer_9_output, [3, 3, 64, self.output_c_dim])
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
        return self.sess.run(self.Y_test, feed_dict={self.X_test: noisy_image,self.isTraining:False})

    def test(self):
        self.sess.run(self.init)
        test_files = glob.glob('./data/test/test_use/*.dcm')
        test_save_filepath = glob.glob('./data/test/test_save')
        print('resnet!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # load testing input
        test_input = []
        test_data = np.zeros((1, 512, 512, 1), dtype=np.single)
        print("[*] Loading test images ...")
        test_input = load_images(test_files)  # list of array of different size
        #test_data = test_input
        if self.load(self.ckpt_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        psnr_sum = 0
        for idx in xrange(len(test_files)):
            test_data = test_input[idx]
            #noisy_image = test_data[idx]
            predicted_noise = self.forward(test_input[idx])
            print("forwarding successfully...")
            output_clean_image = test_data - predicted_noise
            #print(output_clean_image)
            psnr = cal_psnr(test_data, output_clean_image)
            print("cal_psnr")
            psnr_sum += psnr
            test_save_images(test_data, predicted_noise, output_clean_image, test_save_filepath, idx)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)

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
            #eval_save_images(groundtruth, noisyimage, outputimage, \
                        #os.path.join(self.sample_dir, 'test%d_%d_%d.dcm' % (idx, epoch, counter)), idx)
            print("number %d eval ........................" %idx)
        avg_psnr = psnr_sum / len(test_data)
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
