# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:23:58 2018

@author: Joseph
"""
#import hyper_parameters
from net_tools import *
import numpy as np
import tensorflow as tf
batch_size = 32
lr_init = 0.001
epoch_num = 80
epoch_decay = 20
image_size = 128
train_dir = ''
test_dir = ''
valid_dir = ''
checkpoint_filepath = ''
class U_net(object):
    def __init__(self,sess):
        self.batch_size = batch_size
        self.lr = lr_init
        self.epoch_num = epoch_num
        self.epoch_decay = epoch_decay
        self.image_size = image_size
        self.train_dir = train_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.valid_dir = valid_dir
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)))
        self.saver = tf.train.Saver()

    def model(self):
        self.init = tf.global_variables_initializer()
        self.X_input = tf.placeholder(np.single,[None,32,32,1],'input')
        self.X_label = tf.placeholder(np.single,[],'labels')
        X_output = self.net_U(self,self.X_input,reuse = False)
        self.loss = tf.reduce_mean(tf.square(X_output-self.X_label))*(1/self.batch_size)
        optimizer = tf.train.AdamOptimizer(self.lr,name = 'optimizer')
        self.train_step = optimizer.minimize(self.loss)
       
    
    def net_U(self,X_input,reuse):
        
        with tf.variable_scope('layer1',reuse = reuse):
            layer1_output = conv_layer(X_input,[5,5,1,16],[1,1,1,1])
        with tf.variable_scope('layer2',reuse = reuse):
            layer2_output = conv_layer(layer1_output,[3,3,16,32],[1,1,1,1])
        with tf.variable_scope('layer3',reuse = reuse):
            layer3_output = max_pool(layer2_output)
        with tf.variable_scope('layer4',reuse = reuse):
            layer4_output = conv_layer(layer3_output,[3,3,32,64],[1,1,1,1])
        with tf.variable_scope('layer5',reuse = reuse):
            layer5_output = conv_layer(layer4_output,[3,3,64,64],[1,1,1,1])
        with tf.variable_scope('layer6',reuse = reuse):
            layer6_output = max_pool(layer5_output)
        with tf.variable_scope('layer7',reuse = reuse):
            layer7_output1 = up_conv_layer(layer6_output,[2,2,64,64],[None,64,64,64],[1,2,2,1])
            layer7_output2 = layer5_output
            layer7_output = tf.concat([layer7_output1,layer7_output2],3,name = 'concat')
        with tf.variable_scope('layer8',reuse = reuse):
            layer8_output = conv_layer(layer7_output,[3,3,128,64],[1,1,1,1])
        with tf.variable_scope('layer9',reuse = reuse):
            layer9_output = conv_layer(layer8_output,[3,3,64,32],[1,1,1,1])
        with tf.variable_scope('layer10',reuse = reuse):
            layer10_output1 = up_conv_layer(layer9_output,[2,2,32,32],[None,128,128,32],[1,2,2,1])
            layer10_output2 = layer2_output
            layer10_output = tf.concat([layer10_output1,layer10_output2],3,name = 'concat')
        with tf.variable_scope('layer11',reuse = reuse):
            layer11_output = conv_layer(layer10_output,[3,3,64,32],[1,1,1,1])
        with tf.variable_scope('layer12',reuse = reuse):
            layer12_output = conv_layer(layer11_output,[3,3,32,1],[1,1,1,1])
        return layer12_output
        
    def net_Inception_resnet(self,X_input,reuse):
        pass
    
    def net_resnet(self,X_input,reuse):
        pass
    
    def net_plain(self,X_input,reuse):
        pass
    
    def net_desnet(self,X_input,reuse):
        with tf.variable_scope('layer1',reuse = reuse):
            layer1 = conv_layer(X_input,[7,7,1,16],[1,1,1,1])
        with tf.variable_scope('layer2',reuse = reuse):
            for i in range(4):
                layer1 = dense_block(layer1)
            
            
        with tf.variable_scope('layer3',reuse = reuse):
            shape = layer1.get_shape().as_list()
            layer2 = conv_layer(layer1,[1,1,shape[3],64],[1,1,1,1])
            for i in range(4):
                layer2 = dense_block(layer2,name = 'dense_layer.{}'.format(i))
                
        with tf.variable_scope('layer4',reuse = reuse):
            shape = layer2.get_shape().as_list()
            layer3 = conv_layer(layer2,[1,1,shape[3],64],[1,1,1,1])
            for i in range(4):
                layer3 = dense_block(layer3,name = 'dense_layer.{}'.format(i))
        
        with tf.variable_scope('layer5',reuse = reuse):
            shape = layer3.get_shape().as_list()
            layer4 = conv_layer(layer3,[1,1,shape[3],64],[1,1,1,1])
            for i in range(4):
                layer4 = dense_block(layer4,name='dense_layer.{}'.format(i))
        
        with tf.variable_scope('layer6',reuse = reuse):
            shape = layer4.get_shape().as_list()
            layer5 = res_block(layer4,[3,3,shape[3],1],[1,1,1,1])
        return layer5
    
    def train(self):
        self.sess.run(self.init)
        train_images = np.zeros([self.batch_size,128,128,1])
        label_images = np.zeros([self.batch_size,128,128,1])
        train_images,label_images = load_images(train_dir)
        self.sess.run(self.train_step,feed_dict={self.X_input:train_images,self.X_label:label_images})
        self.saver.save(self.sess,checkpoint_filepath)
        
    def test(self):
        test_images = np.zeros([None,512,512,1])
        test_images = load_test_images(test_dir)
        X_input = tf.placeholder(np.single,[None,32,32,1],'input')
        self.saver.restore(self.sess, checkpoint_filepath)
        self.test_final = self.net_U(self,X_input,reuse = True)
        predicted_noise = self.sess.run(self.test_final,feed_dict={X_input:test_images})
        result = test_images-predicted_noise
        result.reshape((512,512))