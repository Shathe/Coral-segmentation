import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import random
import math
import os
import argparse
import time
from Loader import Loader
from imgaug import augmenters as iaa
import imgaug as ia
from augmenters import get_augmenter
import Network
import cv2

random.seed(os.urandom(9))

parser = argparse.ArgumentParser()
#parser.add_argument("--dataset", help="Dataset to train", default='/media/msrobot/discoGordo/Corales/patch_data')  # 'Datasets/MNIST-Big/'
parser.add_argument("--dataset", help="Dataset to train", default='dataset_classif')  # 'Datasets/MNIST-Big/'
parser.add_argument("--init_lr", help="Initial learning rate", default=1e-3)
parser.add_argument("--min_lr", help="Initial learning rate", default=1e-5)
parser.add_argument("--init_batch_size", help="batch_size", default=16)
parser.add_argument("--max_batch_size", help="batch_size", default=16)
parser.add_argument("--epochs", help="Number of epochs to train", default=40)
parser.add_argument("--width", help="width", default=224)
parser.add_argument("--height", help="height", default=224)
parser.add_argument("--save_model", help="save_model", default=1)
args = parser.parse_args()


# Hyperparameter
init_learning_rate = float(args.init_lr)
min_learning_rate = float(args.min_lr)
save_model = bool(int(args.save_model ))
init_batch_size = int(args.init_batch_size)
max_batch_size = int(args.max_batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
height = int(args.height)
channels = 3
change_lr_epoch = math.pow(min_learning_rate/init_learning_rate, 1.0/total_epochs)
change_batch_size = (max_batch_size - init_batch_size) / float(total_epochs - 1)


# Class Loader for lading the data
loader = Loader(dataFolderPath=args.dataset,  problemType = 'classification', width=width, height=height)
testing_samples = len(loader.test_list)
training_samples = len(loader.train_list)

# For Batch_norm or dropout operations: training or testing
training_flag = tf.placeholder(tf.bool)

# Placeholder para las imagenes.
# x = tf.placeholder(tf.float32, shape=[None, None, None, channels], name='input')
x = tf.placeholder(tf.float32, shape=[None, width, height, channels], name='input') # PUT NONE TO BE DYNAMIC
label = tf.placeholder(tf.float32, shape=[None, loader.n_classes], name='output')
# Placeholders para las clases (vector de salida que seran valores de 0-1 por cada clase)

# Network
# output = Network.encoder_nasnet(n_classes=loader.n_classes)
output = Network.encoder_nasnet(input_x=x, n_classes=loader.n_classes)


predictions  = tf.argmax(output, 1)
labels  = tf.argmax(label, 1)


acc, acc_op  = tf.metrics.accuracy(labels, predictions)
mean_acc, mean_acc_op = tf.metrics.mean_per_class_accuracy(labels, predictions, loader.n_classes)
conf_mat = tf.confusion_matrix( labels, predictions, num_classes=loader.n_classes)

saver = tf.train.Saver(tf.global_variables())

confusion_matrix_total = None
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	ckpt = tf.train.get_checkpoint_state('./models/nasnet_encoder/best')  # './model/best'
	ckpt_best = tf.train.get_checkpoint_state('./models/nasnet_encoder/best')  # './model/best'
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		saver.restore(sess, ckpt.model_checkpoint_path)

	# TEST
	count = 0
	suma_acc = 0
	for i in xrange(0, testing_samples, max_batch_size):
		if i + max_batch_size > testing_samples:
			max_batch_size = testing_samples - i
		x_test, y_test = loader.get_batch(size=max_batch_size, train=False)
		count = count + 1
		test_feed_dict = {
			x: x_test,
			label: y_test,
			#mask_label: mask_test,
			training_flag: False
		}
		matrix ,acc_update, acc_total, mean_acc_total, mean_acc_update = sess.run([conf_mat, acc_op, acc, mean_acc, mean_acc_op], feed_dict=test_feed_dict)

		if  i == 0:
			confusion_matrix_total = matrix
		else:
			confusion_matrix_total = confusion_matrix_total + matrix


	print("Accuracy: " + str(acc_update))
	print("mean accuracy: " + str(mean_acc_total))
	print('Confusion matrix')
	print(confusion_matrix_total)

