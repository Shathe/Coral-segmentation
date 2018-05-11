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
import tensorflow.contrib.slim as slim
import Network
import cv2
import math
import sys



random.seed(os.urandom(9))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='/media/msrobot/discoGordo/Corales/patch_data') 
#parser.add_argument("--dataset", help="Dataset to train", default='dataset_classif')  # 'Datasets/MNIST-Big/'
parser.add_argument("--init_lr", help="Initial learning rate", default=1e-4)
parser.add_argument("--min_lr", help="Initial learning rate", default=1e-7)
parser.add_argument("--init_batch_size", help="batch_size", default=32)
parser.add_argument("--max_batch_size", help="batch_size", default=32)
parser.add_argument("--epochs", help="Number of epochs to train", default=3)
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
x = tf.placeholder(tf.float32, shape=[None, None, None, channels], name='input') # PUT NONE TO BE DYNAMIC

label = tf.placeholder(tf.float32, shape=[None, loader.n_classes], name='output')
# Placeholders para las clases (vector de salida que seran valores de 0-1 por cada clase)

# Network
# output = Network.encoder_nasnet(n_classes=loader.n_classes)
output = Network.encoder_resnet101(input_x=x, n_classes=loader.n_classes,  is_training=training_flag)



learning_rate = tf.placeholder(tf.float32, name='learning_rate')

# Loss function
cross_entropy_cnn = - label * tf.nn.log_softmax(output)
cross_entropy_cnn_unbatched = tf.reduce_mean(cross_entropy_cnn , axis = 0) 
cost = tf.reduce_sum(cross_entropy_cnn_unbatched * loader.median_frequency_exp()) 



# Accuracy function
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# This is needed because of using batch normalization
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):

	# Uso el optimizador de Adam y se quiere minimizar la funcion de coste
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train = optimizer.minimize(cost) # VARIABLES TO OPTIMIZE 


tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.profiler.ProfileOptionBuilder.float_operation())
 


# Count parameters
total_parameters = 0
for variable in tf.trainable_variables():
	# shape is an array of tf.Dimension
	shape = variable.get_shape()
	variable_parameters = 1
	for dim in shape:
		variable_parameters *= dim.value
	total_parameters += variable_parameters
print("Total parameters of the net: " + str(total_parameters)+ " == " + str(total_parameters/1000000.0) + "M")



 
# Times to show information of batch traiingn and test
times_show_per_epoch = 30
saver = tf.train.Saver(tf.global_variables())

if not os.path.exists('./models/resnet_encoder/best'):
    os.makedirs('./models/resnet_encoder/best')

with tf.Session() as sess:
	# Create the session and  load weigths if it is possible
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	ckpt_best = tf.train.get_checkpoint_state('./models/resnet_encoder/best')  # Loader model if exists
	ckpt = tf.train.get_checkpoint_state('./models/resnet_encoder')  # Loader model if exists
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		saver.restore(sess, ckpt.model_checkpoint_path)




	# Start variables
	epoch_learning_rate = init_learning_rate
	batch_size_decimal = float(init_batch_size)
	best_val_loss = float('Inf')

	# EPOCH  loop
	for epoch in range(total_epochs):
		# Calculate tvariables for the batch and inizialize others
		time_first=time.time()
		batch_size = int(batch_size_decimal)
		print ("epoch " + str(epoch+ 1) + ", lr: " + str(epoch_learning_rate) + ", batch_size: " + str(batch_size) )

		total_batch = int(training_samples / batch_size)
		show_each_steps = int(total_batch / times_show_per_epoch)

		val_loss_acum = 0
		accuracy_rates_acum = 0
		accuracy_training_acum = 0
		train_loss_acum = 0
		times_test=0

		# steps in every epoch
		for step in range(total_batch):
			# get training data
			batch_x, batch_y = loader.get_batch(size=batch_size, train=True, augmenter='coral')
			train_feed_dict = {
				x: batch_x,
				label: batch_y,
				learning_rate: epoch_learning_rate,
				training_flag: 1
			}


			# TRAIN WITH A BATCH
			_, loss, acc_train = sess.run([train, cost, accuracy ], feed_dict=train_feed_dict)
			accuracy_training_acum = accuracy_training_acum + acc_train
			train_loss_acum = train_loss_acum + loss

			# show info
			if step % show_each_steps == 0:
				# SHOW TRAIN LOSS AND ACCURACY
				if save_model:
					saver.save(sess=sess, save_path='./models/resnet_encoder/weigths.ckpt')
				print("Step:", (step+1), "Loss:", train_loss_acum/(step+1), "Training accuracy:", accuracy_training_acum/(step+1))

		print("Evaluating epoch " + str(epoch) + "...")

		# ACUMULATE TEST LOSS AND ACCURACY
		for test_index in xrange(testing_samples // batch_size):
			batch_x_test, batch_y_test = loader.get_batch(size=batch_size, train=False)

			test_feed_dict = {
				x: batch_x_test,
				label: batch_y_test,
				learning_rate: 0,
				training_flag: 0
			}

			accuracy_rates,  val_loss= sess.run([accuracy, cost], feed_dict=test_feed_dict)
		
			times_test=times_test+1
			val_loss_acum = val_loss_acum + val_loss
			accuracy_rates_acum = accuracy_rates + accuracy_rates_acum


		print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy=', accuracy_rates_acum/times_test,  '/ val_loss =', val_loss_acum/times_test)

		# save model
		if save_model and best_val_loss > val_loss_acum:
			print(save_model)
			best_val_loss = val_loss_acum
			saver.save(sess=sess, save_path='./models/resnet_encoder/best/weigths.ckpt')

		# show tiem to finish training
		time_second=time.time()
		epochs_left = total_epochs - epoch - 1
		segundos_per_epoch=time_second-time_first
		print(str(segundos_per_epoch * epochs_left)+' seconds to end the training. Hours: ' + str(segundos_per_epoch * epochs_left/3600.0))
	
		#agument batch_size per epoch and decrease the learning rate
		epoch_learning_rate = init_learning_rate * math.pow(change_lr_epoch, epoch)
		batch_size_decimal = batch_size_decimal + change_batch_size
	


