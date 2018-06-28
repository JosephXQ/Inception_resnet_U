# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:22:11 2018

@author: Joseph
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:24:45 2017

@author: Joseph
"""

import argparse
import os
#import scipy.misc
import numpy as np
#from resFCN_init3 import DnCNN
#from resnet_U_init import DnCNN
#from Inception_resnet_U_init import DnCNN
from Inception_resnet_U4 import DnCNN
#from Inception_resnet_U3init import DnCNN
#from Inception_resnet_U_prob import DnCNN
#from resFCN_Inception_init import DnCNN
#from plain_init import  DnCNN
#from U_net import DnCNN
#from dense_net_init import DnCNN
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--trainset', dest='trainset', default='trainmodel_U4+Inception-resnet_60-1200_16b_nmumse', help='name of the training dataset')
parser.add_argument('--testset', dest='testset', default='train', help='name of the training dataset')

parser.add_argument('--epoch', dest='epoch', type=int, default=150, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=80, help='# images in batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch size/input size')

parser.add_argument('--input_c', dest='input_c', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_c', dest='output_c', type=int, default=1, help='# of output image channels')

parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999, help='momentum term of adam')

parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')

parser.add_argument('--use_gpu', dest='use_gpu', type=bool, default=True, help='gpu flag')
parser.add_argument('--phase', dest='phase', default='test', help='train or test')
args = parser.parse_args()

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    if args.use_gpu:
        # added to controll the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = DnCNN(sess,  lr=args.lr, dataset=args.trainset)
            if args.phase == 'train':
                print("Train successfully!..............")
                model.train()
            else:
                print("test successfully!..............")
                model.test()

    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = DnCNN(sess, lr=args.lr, dataset=args.trainset)
            if args.phase == 'train':
                model.train()
            else:
                model.test()

if __name__ == '__main__':
    tf.app.run()
