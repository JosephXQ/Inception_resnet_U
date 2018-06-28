# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:26:23 2018

@author: Joseph
"""

import tensorflow as tf 
import math
from six.moves import xrange
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
def batch_normalization(logits, scale, offset, isTraining = False, name="bn"):
	mean, var = tf.nn.moments(logits, [0, 1, 2])
	moving_mean_shape = mean.get_shape()
	moving_var_shape = var.get_shape()
	moving_mean = tf.get_variable('moving_mean',moving_mean_shape,initializer=tf.zeros_initializer,trainable=False)
	moving_var = tf.get_variable('moving_var', moving_var_shape,initializer=tf.ones_initializer,trainable=False)
	moving_mean = moving_averages.assign_moving_average(moving_mean,mean,0.95)
	moving_var = moving_averages.assign_moving_average(moving_var,var,0.95)
	mean, var = control_flow_ops.cond(isTraining, lambda:(mean,var), lambda:(moving_mean, moving_var))
	output = tf.nn.batch_normalization(logits, mean, var, offset, scale, variance_epsilon=1e-5)
	return output


def get_conv_weights(weight_shape, sess, name="get_conv_weights"):
	#return math.sqrt(2 / (9.0 * 64)) * sess.run(tf.truncated_normal(weight_shape))
	return sess.run(tf.truncated_normal(weight_shape,stddev=0.1))

def get_bn_weights(weight_shape, clip_b, sess, name="get_bn_weights"):
	weights = get_conv_weights(weight_shape, sess)
	#return clipping(weights, clip_b)
	return  weights

def clipping(A, clip_b, name="clipping"):
	h, w = A.shape 
	for i in xrange(h):
		for j in xrange(w):
			if A[i,j] >= 0 and A[i,j] < clip_b:
				A[i,j] = clip_b
			elif A[i,j] > -clip_b and A[i,j] < 0:
				A[i,j] = -clip_b
	return A

	









