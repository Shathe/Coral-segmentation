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
#tensorboard --logdir=train:./logs/train,test:./logs/test/

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='./camvid')  # 'Datasets/MNIST-Big/'
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample", default=3)
parser.add_argument("--tensorboard", help="Monitor with Tensorboard", default=0)
parser.add_argument("--augmentation", help="Image augmentation", default=1)
parser.add_argument("--init_lr", help="Initial learning rate", default=5e-3)
parser.add_argument("--min_lr", help="Initial learning rate", default=3e-7)
parser.add_argument("--init_batch_size", help="batch_size", default=2)
parser.add_argument("--max_batch_size", help="batch_size", default=2)
parser.add_argument("--n_classes", help="number of classes to classify", default=11)
parser.add_argument("--ignore_label", help="class to ignore", default=11)
parser.add_argument("--epochs", help="Number of epochs to train", default=2)
parser.add_argument("--width", help="width", default=224)
parser.add_argument("--height", help="height", default=224)
parser.add_argument("--save_model", help="save_model", default=1)
args = parser.parse_args()



# Hyperparameter
init_learning_rate = float(args.init_lr)
min_learning_rate = float(args.min_lr)
augmentation = bool(int(args.augmentation))
save_model = bool(int(args.save_model ))
tensorboard = bool(int(args.tensorboard))
init_batch_size = int(args.init_batch_size)
max_batch_size = int(args.max_batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
n_classes = int(args.n_classes)
ignore_label = int(args.ignore_label)
height = int(args.height)
channels = int(args.dimensions)
change_lr_epoch = math.pow(min_learning_rate/init_learning_rate, 1.0/total_epochs)
change_batch_size = (max_batch_size - init_batch_size) / float(total_epochs - 1)

loader = Loader(dataFolderPath=args.dataset, n_classes=n_classes, problemType = 'segmentation', width=width, height=height, ignore_label = ignore_label)
testing_samples = len(loader.image_test_list)
training_samples = len(loader.image_train_list)


# For Batch_norm or dropout operations: training or testing
training_flag = tf.placeholder(tf.bool)

# Placeholder para las imagenes.
x = tf.placeholder(tf.float32, shape=[None, None, None, channels], name='input')
label = tf.placeholder(tf.float32, shape=[None, None, None, n_classes+1], name='output') # +1 for ignore class
mask_label = tf.placeholder(tf.float32, shape=[None, None, None, n_classes], name='mask')
# Placeholders para las clases (vector de salida que seran valores de 0-1 por cada clase)

# Network
output = Network.complex(input_x=x, n_classes=n_classes, width=width, height=height, channels=channels, training=training_flag)
shape_output = tf.shape(output)
label_shape = tf.shape(label)

predictions = tf.reshape(output, [-1, shape_output[1]* shape_output[2] , shape_output[3]]) # tf.reshape(output, [-1])
labels = tf.reshape(label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])
output_image = tf.expand_dims(tf.cast(tf.argmax(output, 3), tf.float32), -1)
# mask_labels = tf.reshape(mask_label, [-1, label_shape[1]* label_shape[2] , label_shape[3] - 1]) # tf.reshape(output, [-1])


 # Metrics

labels = tf.argmax(labels, 2)
predictions = tf.argmax(predictions, 2)
labels = tf.reshape(labels,[tf.shape(labels)[0] * tf.shape(labels)[1]])
predictions = tf.reshape(predictions,[tf.shape(predictions)[0] * tf.shape(predictions)[1]])

print(tf.where(tf.less_equal(labels, n_classes - 1)).get_shape())

indices = tf.squeeze(tf.where(tf.less_equal(labels, n_classes - 1))) # ignore all labels >= num_classes 
labels = tf.cast(tf.gather(labels, indices), tf.int32)
predictions = tf.gather(predictions, indices)


acc, acc_op  = tf.metrics.accuracy(labels, predictions)
mean_acc, mean_acc_op = tf.metrics.mean_per_class_accuracy(labels, predictions, n_classes)
miou, miou_op = tf.metrics.mean_iou(labels, predictions, n_classes)
conf_mat = tf.confusion_matrix( labels, predictions, num_classes=n_classes)

saver = tf.train.Saver(tf.global_variables())

confusion_matrix_total = None
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	ckpt = tf.train.get_checkpoint_state('./model_complex')  # './model/best'
	ckpt_best = tf.train.get_checkpoint_state('./model_complex/best')  # './model/best'
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		saver.restore(sess, ckpt.model_checkpoint_path)

	# TEST
	count = 0
	suma_acc = 0
	for i in xrange(0, testing_samples, max_batch_size):
		if i + max_batch_size > testing_samples:
			max_batch_size = testing_samples - i
		x_test, y_test, mask_test = loader.get_batch(size=max_batch_size, train=False)
		count = count + 1
		test_feed_dict = {
			x: x_test,
			label: y_test,
			#mask_label: mask_test,
			training_flag: False
		}
		image_salida, matrix ,acc_update, acc_total, miou_update, miou_total,mean_acc_total, mean_acc_update = sess.run([output, conf_mat, acc_op, acc, miou_op, miou, mean_acc, mean_acc_op], feed_dict=test_feed_dict)

		if  i == 0:
			confusion_matrix_total = matrix
		else:
			confusion_matrix_total = confusion_matrix_total + matrix

	if not os.path.exists('output/'):
		os.makedirs('output/')
		
	image_salida = np.argmax(image_salida, 3)
	for index_output in xrange(max_batch_size):
		name_split = loader.image_test_list[index_output + i].split('/')
		name = name_split[len(name_split)-1].replace('.jpg','.png').replace('.jpeg','.png')
		cv2.imwrite('output/'+name, image_salida[index_output])

	print("Accuracy: " + str(acc_update))
	print("miou: " + str(miou_total))
	print("mean accuracy: " + str(mean_acc_total))
	print('Confusion matrix')
	print(confusion_matrix_total)

	x_test, y_test, mask_test = loader.get_batch(size=1, train=False)

	import time
	first = time.time()
	predictions = sess.run(output_image, feed_dict={x: x_test, training_flag : False})
	second = time.time()
	print(str(second - first) + " seconds to load")


	first = time.time()
	output_image.eval(feed_dict={x: x_test, training_flag : False})
	second = time.time()
	print(str(second - first) + " seconds to load")

# mejor complex sin regularizer y droput: 0.80/0.77, 0.44, 0.66
