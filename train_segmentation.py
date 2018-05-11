import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
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
import math
import sys
sys.path.append("models")
from DeepLabV3_plus import build_deeplabv3_plus


random.seed(os.urandom(9))
#tensorboard --logdir=train:./logs/train,test:./logs/test/

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='./dataset_segmentation')  # 'Datasets/MNIST-Big/'
#/media/msrobot/discoGordo/Download_april/machine_printed_legible
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample", default=3)
parser.add_argument("--augmentation", help="Image augmentation", default=1)
parser.add_argument("--init_lr", help="Initial learning rate", default=1e-3)
parser.add_argument("--lr_decay", help="1 for lr decay, 0 for not", default=0)
parser.add_argument("--min_lr", help="Initial learning rate", default=1e-8)
parser.add_argument("--init_batch_size", help="batch_size", default=2)
parser.add_argument("--max_batch_size", help="batch_size", default=2)
parser.add_argument("--n_classes", help="number of classes to classify", default=11)
parser.add_argument("--ignore_label", help="class to ignore", default=11)
parser.add_argument("--epochs", help="Number of epochs to train", default=350)
parser.add_argument("--width", help="width", default=512)
parser.add_argument("--height", help="height", default=512)
parser.add_argument("--save_model", help="save_model", default=1)
parser.add_argument("--finetune_encoder", help="whether to finetune_encoder", default=1)
args = parser.parse_args()



# Hyperparameter
init_learning_rate = float(args.init_lr)
min_learning_rate = float(args.min_lr)
lr_decay = bool(int(args.lr_decay))
augmentation = bool(int(args.augmentation))
save_model = bool(int(args.save_model ))
init_batch_size = int(args.init_batch_size)
max_batch_size = int(args.max_batch_size)
total_epochs = int(args.epochs)
finetune_encoder = int(args.finetune_encoder)
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
input_x = tf.placeholder(tf.float32, shape=[None, None, None, channels], name='input')
batch_images = tf.reverse(input_x, axis=[-1]) #opencv rgb -bgr
label = tf.placeholder(tf.float32, shape=[None, None, None, n_classes + 1], name='output') # the n_classes + 1 is for the ignore classes
mask_label = tf.placeholder(tf.float32, shape=[None, None, None, n_classes], name='mask')
# Placeholders para las clases (vector de salida que seran valores de 0-1 por cada clase)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

# Network
#output = Network.encoder_decoder(input_x=x, n_classes=n_classes, width=width, height=height, channels=channels, training=training_flag)
output = Network.encoder_decoder_resnet101(input_x=input_x, n_classes=n_classes, width=width, height=height, channels=channels, training=training_flag)
shape_output = tf.shape(output)
label_shape = tf.shape(label)

predictions = tf.reshape(output, [-1, shape_output[1]* shape_output[2] , shape_output[3]]) # tf.reshape(output, [-1])
labels = tf.reshape(label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])
# mask_labels = tf.reshape(mask_label, [-1, label_shape[1]* label_shape[2] , label_shape[3] - 1]) # - 1 because of the ignoreclass

# calculate the loss [cross entropy]
#Clip output (softmax) for -inf values and calculate the log
#clipped_output =  tf.log(tf.clip_by_value(tf.nn.softmax(predictions), 1e-20, 1e+20))
log_softmax =  tf.nn.log_softmax(predictions)
labels_ignore = labels[:,:,n_classes]
labels = labels[:,:,:n_classes]

# Compare to the label (loss)
softmax_loss = labels*log_softmax
# mask the loss
cost_masked = tf.reduce_mean(softmax_loss, axis=1)
#cost_masked = tf.reduce_mean(softmax_loss*mask_labels, axis=1)
# Get hte median frequency weights of the labels
weights = loader.median_frequency_exp()
# Apply the tweights to the loss and multiply for the number of classes (you are applying the mean)
cost_with_weights = tf.reduce_mean(cost_masked*weights*n_classes, axis=1) 
# For normalizing the loss accoding to the number of pixels calculated, multiply for the percentage of  non mask pixels (valuable pixels)
mean_masking = tf.reduce_mean(labels_ignore)
#cost = -tf.reduce_mean(cost_with_weights, axis=0) / mean_masking
cost = -tf.reduce_mean(cost_with_weights, axis=0) / (1 - mean_masking)



'''

# Accuracy per class:
predictions_one_hot = tf.one_hot(tf.argmax(predictions, 2) , n_classes)
correct_prediction_per_class = tf.cast(tf.equal(predictions_one_hot, labels), tf.float32) 
accuracy_per_class = tf.multiply(labels, correct_prediction_per_class )
accuracy_per_class_sum = tf.reduce_sum(accuracy_per_class, axis=0)
accuracy_per_class_sum = tf.reduce_sum(accuracy_per_class_sum, axis=0)
labels_sum = tf.reduce_sum(labels, axis=0)
labels_sum = tf.reduce_sum(labels_sum, axis=0)
mean_accuracy = tf.reduce_mean(accuracy_per_class_sum/labels_sum)
'''

# Calculate the accuracy
labels = tf.argmax(labels, 2)
predictions = tf.argmax(predictions, 2)

labels = tf.reshape(labels,[tf.shape(labels)[0] * tf.shape(labels)[1]])
predictions = tf.reshape(predictions,[tf.shape(predictions)[0] * tf.shape(predictions)[1]])


print(tf.where(tf.less_equal(labels, n_classes - 1)).get_shape())

indices = tf.squeeze(tf.where(tf.less_equal(labels, n_classes - 1))) # ignore all labels >= num_classes 
labels = tf.cast(tf.gather(labels, indices), tf.int64)
predictions = tf.gather(predictions, indices)


correct_prediction = tf.cast(tf.equal(labels, predictions), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)



#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#hacer media por clase acc
 

# SUMMARY
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('learning_rate', learning_rate)
if int(args.dimensions) == 3:

	tf.summary.image('input', batch_images, max_outputs=10)
else:
	tf.summary.image('input_0-3', batch_images[:, :, :, 0:3], max_outputs=10)

output_image = tf.expand_dims(tf.cast(tf.argmax(output, 3), tf.float32), -1)
tf.summary.image('output', output_image, max_outputs=10)
label_image = tf.expand_dims(tf.cast(tf.argmax(label, 3), tf.float32), -1)
tf.summary.image('label', label_image, max_outputs=10)



restore = True
restore_variables = [var for var in tf.trainable_variables() if 'resnet_v2_101' in var.name]
decoder_variables =  [var for var in tf.trainable_variables() if 'resnet_v2_101' not in var.name ]
all_variables =  [var for var in tf.trainable_variables() ]


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




# For batch norm
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):

	# Uso el optimizador de Adam y se quiere minimizar la funcion de coste
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) #adamOptimizer does not need lr decay
	train = optimizer.minimize(cost, var_list=all_variables) # VARIABLES TO PTIMIZE 





 
# Times to show information of batch traiingn and test
times_show_per_epoch = 3

saver = tf.train.Saver(tf.global_variables())

if finetune_encoder:
	saver = tf.train.Saver(var_list = restore_variables)



if not os.path.exists('./models/model_decoder/best'):
    os.makedirs('./models/model_decoder/best')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())


	ckpt = tf.train.get_checkpoint_state('./models/model_decoder/best')
	if finetune_encoder:
		ckpt = tf.train.get_checkpoint_state('./models/resnet_encoder/')
	
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Loading model...')
		saver.restore(sess, ckpt.model_checkpoint_path)
		print('Model loaded')


	merged = tf.summary.merge_all()
	writer_train = tf.summary.FileWriter('./logs/train', sess.graph)
	writer_test = tf.summary.FileWriter('./logs/test', sess.graph)



	# Start variables
	global_step = 0
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
		times_test=0

		# steps in every epoch
		for step in range(total_batch):
			# get training data
			batch_x, batch_y, batch_mask = loader.get_batch(size=batch_size, train=True, augmenter='segmentation')#, augmenter='segmentation'

			train_feed_dict = {
				input_x: batch_x,
				label: batch_y,
				learning_rate: epoch_learning_rate,
				mask_label: batch_mask,
				training_flag: 1
			}
			_, loss = sess.run([train, cost], feed_dict=train_feed_dict)

			# show info
			if step % show_each_steps == 0:
				global_step += show_each_steps

				train_summary, train_accuracy= sess.run([merged, accuracy], feed_dict=train_feed_dict)
				print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
				writer_train.add_summary(train_summary, global_step=global_step/show_each_steps)

				batch_x_test, batch_y_test, batch_mask = loader.get_batch(size=batch_size, train=False)

				test_feed_dict = {
					input_x: batch_x_test,
					label: batch_y_test,
					learning_rate: 0,
					mask_label: batch_mask,
					training_flag: 0
				}
				test_summary, accuracy_rates,  val_loss= sess.run([merged, accuracy, cost], feed_dict=test_feed_dict)
			
				writer_test.add_summary(test_summary, global_step=global_step/show_each_steps)
				# in case there is a nan value
				times_test=times_test+1
				if math.isnan(val_loss):
					val_loss = np.inf

				val_loss_acum = val_loss_acum + val_loss
				accuracy_rates_acum = accuracy_rates + accuracy_rates_acum


		print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy=', accuracy_rates_acum/times_test,  '/ val_loss =', val_loss_acum/times_test)

		# save models
		if save_model:
			saver.save(sess=sess, save_path='./models/model_decoder/dense.ckpt')
		if save_model and best_val_loss > val_loss_acum:
			print(save_model)
			best_val_loss = val_loss_acum
			saver.save(sess=sess, save_path='./models/model_decoder/best/dense.ckpt')

		# show tiem to finish training
		time_second=time.time()
		epochs_left = total_epochs - epoch - 1
		segundos_per_epoch=time_second-time_first
		print(str(segundos_per_epoch * epochs_left)+' seconds to end the training. Hours: ' + str(segundos_per_epoch * epochs_left/3600.0))
	
		#agument batch_size per epoch and decrease the learning rate
		if lr_decay:
			epoch_learning_rate = init_learning_rate * math.pow(change_lr_epoch, epoch) # adamOptimizer does not need lr decay
		batch_size_decimal = batch_size_decimal + change_batch_size
	


