from __future__ import print_function
import os
import numpy as np
from os import walk
from keras.utils.np_utils import to_categorical
import glob
import ntpath
import cv2
import random
from imgaug import augmenters as iaa
import imgaug as ia
from augmenters import get_augmenter
import tensorflow as tf
import sys

random.seed(os.urandom(9))


problemTypes=['classification', 'segmentation']

class Loader:
	#poder aumentar datos.. poder batch.. opcion sampleado aleatorio
	#guardar lista de imagenes test y train de una carpeta
	# opcion de que devuelva la mascara de lo que se aumenta
	# opcion del tipo de entrenamiento. Clafiicacion, semantica, gan.. eventos
	def __init__(self, dataFolderPath, width=224, height=224, dim=3, n_classes=21,  problemType='classification', ignore_label=None):
		self.dataFolderPath=dataFolderPath
		self.height = height
		self.width = width
		self.dim = dim 
		self.ignore_label = ignore_label # label to ignore
		self.freq = np.zeros(n_classes) # vector for calculating the class frequency
		self.index_train = 0 #indexes for iterating while training
		self.index_test = 0#indexes for iterating while testing
		if ignore_label and ignore_label < n_classes:

			raise Exception( 'please, change the labeling in order to put the ignore label value to the last value > nunm_classes')

		# Load filepaths
		files = []
		print('Reading files...')
		'''
		possible structures:
		classification: dataset/train/class/image.png
		segmentation: dataset/train/images/image.png
		'''
		files = glob.glob(os.path.join(dataFolderPath,'*','*','*'))
		'''
		for (dirpath, dirnames, filenames) in walk(dataFolderPath):
			filenames = [os.path.join(dirpath, filename) for filename in filenames]
			if filenames != None and filenames != []:
				files = files + filenames
				print ('Read '+ str(len(files)) + ' files...', end='\r')
				sys.stdout.flush()
		'''
		print('Structuring test and train files...')
		self.test_list = [file for file in files if '/test/' in file]
		self.train_list = [file for file in files if '/train/' in file]

		# Check problem type
		if problemType in problemTypes:
			self.problemType=problemType
		else:
			raise Exception('Not valid problemType')


		if problemType == 'classification':
			# The structure has to be dataset/train/class/image.png
			#Extract dictionary to map class -> label

			# Shuffle train
			s = np.arange(len(self.train_list))
			np.random.shuffle(s)
			self.train_list=np.array(self.train_list)[s]

			# Shuffle test
			s = np.arange(len(self.test_list))
			np.random.shuffle(s)
			self.test_list=np.array(self.test_list)[s]

			print('Loaded '+ str(len(self.train_list)) +' training samples')
			print('Loaded '+ str(len(self.test_list)) +' testing samples')
			classes_train = [file.split('/train/')[1].split('/')[0] for file in self.train_list]
			classes_test = [file.split('/test/')[1].split('/')[0] for file in self.test_list]
			classes = np.unique(np.concatenate((classes_train, classes_test)))
			self.classes = {}
			for label in range(len(classes)):
				self.classes[classes[label]] = label
			self.n_classes=len(classes)
			self.freq = np.zeros(self.n_classes)


		elif problemType == 'segmentation':
			# The structure has to be dataset/train/images/image.png
			# The structure has to be dataset/train/labels/label.png
			# Separate image and label lists
			# Sort them to align labels and images


			self.image_train_list = [file for file in self.train_list if '/images/' in file]
			self.image_test_list = [file for file in self.test_list if '/images/' in file]
			self.label_train_list = [file for file in self.train_list if '/labels/' in file]
			self.label_test_list = [file for file in self.test_list if '/labels/' in file]
			self.label_test_list.sort()
			self.image_test_list.sort()
			self.label_train_list.sort()
			self.image_train_list.sort()

			# Shuffle train
			s = np.arange(len(self.image_train_list))
			np.random.shuffle(s)
			self.image_train_list=np.array(self.image_train_list)[s]
			self.label_train_list=np.array(self.label_train_list)[s]
			'''
			# Shuffle test
			s = np.arange(len(self.image_test_list))
			np.random.shuffle(s)
			self.image_test_list=np.array(self.image_test_list)[s]
			self.label_test_list=np.array(self.label_test_list)[s]
			'''
			print('Loaded '+ str(len(self.image_train_list)) +' training samples')
			print('Loaded '+ str(len(self.image_test_list)) +' testing samples')
			self.n_classes = n_classes

		
		print('Dataset contains '+ str(self.n_classes) +' classes')


	# Returns a random batch of segmentation images: X, Y, mask
	def _get_batch_segmentation(self, size=32, train=True, augmenter=None):

		x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
		y = np.zeros([size, self.height, self.width], dtype=np.uint8)
		mask_expanded = np.ones([size, self.height, self.width, self.n_classes], dtype=np.uint8)



		if train:
			image_list = self.image_train_list
			label_list = self.label_train_list
			index_files = self.index_train
			folder = '/train/'
			# Get [size] random numbers			
			indexes = [i%len(image_list) for i in range(self.index_train, self.index_train+size)]
			self.index_train=indexes[-1] + 1

		else:
			image_list = self.image_test_list
			label_list = self.label_test_list
			folder = '/test/'
		 # Get [size] random numbers
			indexes = [i%len(image_list) for i in range(self.index_test, self.index_test+size)]
			self.index_test=indexes[-1] + 1


		random_images = [image_list[number] for number in indexes]
		random_labels = [label_list[number] for number in indexes]

		# for every random image, get the image, label and mask.
		# the augmentation has to be done separately due to augmentation
		for index in range(size):
			if self.dim == 1:
				img = cv2.imread(random_images[index], 0)
			else:
				img = cv2.imread(random_images[index])

			label = cv2.imread(random_labels[index],0)

			if img is None or label is  None:
				print(random_images[index])
				print(random_labels[index])
				print(indexes[index])

			if img.shape[1] != self.width or img.shape[0] != self.height:
				img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_AREA)
			if label.shape[1] != self.width or label.shape[0] != self.height:
				label = cv2.resize(label, (self.width, self.height), interpolation = cv2.INTER_NEAREST)

			macara = mask_expanded[index, :, :, 0] 
			if train and augmenter:

				seq_image_contrast, seq_image_translation, seq_label, seq_mask = get_augmenter(name=augmenter, c_val=self.ignore_label)

				#apply some contrast  to de rgb image
				img=img.reshape(sum(((1,),img.shape),()))
				img = seq_image_contrast.augment_images(img)  
				img=img.reshape(img.shape[1:])


				#Apply shifts and rotations to the mask, labels and image
				
				# Reshapes for the AUGMENTER framework
				# the loops are due to the external library failures
				
				img=img.reshape(sum(((1,),img.shape),()))
				img = seq_image_translation.augment_images(img)  
				img=img.reshape(img.shape[1:])

				label=label.reshape(sum(((1,),label.shape),()))
				label = seq_label.augment_images(label)
				label=label.reshape(label.shape[1:])

				macara=macara.reshape(sum(((1,),macara.shape),()))
				macara = seq_mask.augment_images(macara)
				macara=macara.reshape(macara.shape[1:])

			# modify the mask and the labels

			mask_ignore = label == self.ignore_label
			macara[mask_ignore] = 0
			label[mask_ignore] = self.n_classes

			if self.dim == 1:
				img = np.reshape(img, (img.shape[0], img.shape[1], self.dim))

			x[index, :, :, :] = img
			y[index, :, :] = label
			for i in xrange(mask_expanded.shape[3]):
				mask_expanded[index, :, :, i] = macara

		# the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
		a, b, c =y.shape
		y = y.reshape((a*b*c))


		# Convert to categorical. Add one class for ignored pixels
		y = to_categorical(y, num_classes=self.n_classes+1)
		y = y.reshape((a,b,c,self.n_classes+1)).astype(np.uint8)


¡		#tf.keras.applications.imagenet_utils.preprocess_input(x, mode='tf')
		#x = tf.keras.applications.xception.preprocess_input(x)
		x = x.astype(np.float32) / 255.0 - 0.5

		return x, y, mask_expanded 


	# Returns a random batch
	def _get_batch_rgb(self, size=32, train=True, augmenter=None):

		x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
		y = np.zeros([size], dtype=np.uint8)


		if train:
			file_list = self.train_list
			folder = '/train/'
			# Get [size] random numbers
			indexes = [i%len(file_list) for i in range(self.index_train, self.index_train+size)]
			self.index_train=indexes[-1] + 1

		else:
			file_list = self.test_list
			folder = '/test/'
			# Get [size] random numbers
			indexes = [i%len(file_list) for i in range(self.index_test, self.index_test+size)]
			self.index_test=indexes[-1] + 1


		random_files = [file_list[number] for number in indexes]
		classes = [self.classes[file.split(folder)[1].split('/')[0]] for file in random_files]

		for index in range(size):
			img = cv2.imread(random_files[index])
			if img is None :
				print(random_files[index])
				print(indexes[index])

			if img.shape[1] != self.width or img.shape[0] != self.height:
				img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_AREA)
				


			x[index, :, :, :] = img
			y[index] = classes[index]
		# the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
		y = to_categorical(y, num_classes=self.n_classes)
		# augmentation
		if augmenter:
			augmenter_seq = get_augmenter(name=augmenter)
			x = augmenter_seq.augment_images(x)


		#x = x.astype(np.float32) 
		#tf.keras.applications.imagenet_utils.preprocess_input(x, mode='tf')
		#x = tf.keras.applications.xception.preprocess_input(x)
		x = x.astype(np.float32) / 255.0 - 0.5

		return x, y


	def _get_key_by_value(self, dictionary, value_searching):
		for key, value in dictionary.iteritems():  
			if value == value_searching:
				return key

		return None

	# Returns a random batch
	def get_batch(self, size=32, train=True, augmenter=None):
		'''
		Gets a batch of size [size]. If [train] the data will be training data, if not, test data.
		if augmenter is no None, image augmentation will be perform (see file augmenters.py)
		if images are bigger than max_size of smaller than min_size, images will be resized (forced)
		'''
		if self.problemType == 'classification':
			return self._get_batch_rgb(size=size, train=train, augmenter=augmenter)
		elif self.problemType == 'segmentation':
			return self._get_batch_segmentation(size=size, train=train, augmenter=augmenter)

	# Returns the median frequency for class imbalance. It can be soften with the soft value (<=1)
	def median_frequency_exp(self, soft=0.30):
		if self.problemType == 'classification':
			quantity = []
			for class_name in self.classes:
				path = os.path.join(self.dataFolderPath, 'train', class_name)
				class_freq = len(glob.glob(os.path.join(path,'*')))
				self.freq[self.classes[class_name]] = class_freq

		elif self.problemType == 'segmentation' :
			for image_label_train in self.label_train_list:
				image = cv2.imread(image_label_train,0)
				for label in xrange(self.n_classes):
					self.freq[label] = self.freq[label] + sum(sum(image == label))
		
		# Common code
		zeros = self.freq == 0
		if sum(zeros) > 0:
			print('There are some classes which are not contained in the training samples')
			'''
			indexes = [i for i, x in enumerate(zeros) if x]
			for index_zero in indexes:
			
				print(len(glob.glob(os.path.join(self.dataFolderPath, 'train', self._get_key_by_value(self.classes, index_zero),'*'))))
				list_train = glob.glob(os.path.join(self.dataFolderPath, 'train', self._get_key_by_value(self.classes, index_zero),'*'))
				for i in xrange(1):
					try:
						file = random.choice(list_train)
						os.rename(file, file.replace('/test/' ,'/train/'))
					except Exception:
						pass
			'''

		results = np.median(self.freq)/self.freq
		results[zeros]=0 # for not inf values.
		results = np.power(results,soft)
		return results


		
if __name__ == "__main__":
	'''
	loader = Loader('./dataset_rgb')
	print(loader.classes)
	x, y =loader.get_batch(size=2)
	print(y)
	'''
	#	

	loader = Loader('./dataset_segmentation', problemType = 'segmentation', ignore_label=11, n_classes=11)
	# print(loader.median_frequency_exp())
	x, y, mask =loader.get_batch(size=3, augmenter='segmentation')#
	'''
	for i in xrange(50):

		cv2.imshow('x',x[i,:,:,:])
		cv2.imshow('y',(np.argmax(y,3)[i,:,:]*255).astype(np.uint8))
		print(mask.shape)
		cv2.imshow('mask',mask[i,:,:,0]*255)
		cv2.waitKey(0)
	cv2.destroyAllWindows()
	x, y, mask =loader.get_batch(size=3, train=False)
	'''

 
