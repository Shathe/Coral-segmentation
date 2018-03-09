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

problemTypes=['classification', 'GAN', 'segmentation', 'DVS']

class Loader:
	#poder aumentar datos.. poder batch.. opcion sampleado aleatorio
	#guardar lista de imagenes test y train de una carpeta
	# opcion de que devuelva la mascara de lo que se aumenta
	# opcion del tipo de entrenamiento. Clafiicacion, semantica, gan.. eventos
	def __init__(self, dataFolderPath, width=224, height=224, dim=3, n_classes=21,  problemType='classification', ignore_label=None):
		self.height = height
		self.width = width
		self.dim = dim 
		self.ignore_label = ignore_label
		self.freq = np.zeros(n_classes)

		if ignore_label and ignore_label < n_classes:
			raise Exception( 'please, change the labeling in order to put the ignore label value to the last value > nunm_classes')

		# Load filepaths
		files = []
		for (dirpath, dirnames, filenames) in walk(dataFolderPath):
			filenames = [os.path.join(dirpath, filename) for filename in filenames]
			files.extend(filenames)

		self.test_list = [file for file in files if '/test/' in file]
		self.train_list = [file for file in files if '/train/' in file]
		self.train_list.sort()
		self.test_list.sort()
		# Check problem type
		if problemType in problemTypes:
			self.problemType=problemType
		else:
			raise Exception('Not valid problemType')


		if problemType == 'classification' or problemType == 'GAN':
			#Extract dictionary to map class -> label
			print('Loaded '+ str(len(self.train_list)) +' training samples')
			print('Loaded '+ str(len(self.test_list)) +' testing samples')
			classes_train = [file.split('/train/')[1].split('/')[0] for file in self.train_list]
			classes_test = [file.split('/test/')[1].split('/')[0] for file in self.test_list]
			classes = np.unique(np.concatenate((classes_train, classes_test)))
			self.classes = {}
			for label in range(len(classes)):
				self.classes[classes[label]] = label
			self.n_classes=len(classes)

		elif problemType == 'segmentation':
			# The structure has to be dataset/train/images/image.png
			# The structure has to be dataset/train/labels/label.png
			# Separate image and label lists


			self.image_train_list = [file for file in self.train_list if '/images/' in file]
			self.image_test_list = [file for file in self.test_list if '/images/' in file]
			self.label_train_list = [file for file in self.train_list if '/labels/' in file]
			self.label_test_list = [file for file in self.test_list if '/labels/' in file]
			self.label_test_list.sort()
			self.image_test_list.sort()
			self.label_train_list.sort()
			self.image_train_list.sort()
			print('Loaded '+ str(len(self.image_train_list)) +' training samples')
			print('Loaded '+ str(len(self.image_test_list)) +' testing samples')
			self.n_classes = n_classes

		elif problemType == 'DVS':
			# Yet to know how to manage this data
			pass
		print('Dataset contains '+ str(self.n_classes) +' classes')


	# Returns a random batch of segmentation images: X, Y, mask
	def _get_batch_segmentation(self, size=32, train=True, augmenter=None, index=None, validation=False):

		x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
		y = np.zeros([size, self.height, self.width], dtype=np.uint8)
		mask_expanded = np.ones([size, self.height, self.width, self.n_classes], dtype=np.uint8)

		image_list = self.image_test_list
		label_list = self.label_test_list
		folder = '/test/'
		if train:
			image_list = self.image_train_list
			label_list = self.label_train_list
			folder = '/train/'

		# Get [size] random numbers
		indexes = [random.randint(0,len(image_list) - 1) for file in range(size)]
		if index:
			indexes = [i for i in range(index, index+size)]

		random_images = [image_list[number] for number in indexes]
		random_labels = [label_list[number] for number in indexes]

		# for every random image, get the image, label and mask.
		# the augmentation has to be done separately due to augmentation
		for index in range(size):
			img = cv2.imread(random_images[index])
			label = cv2.imread(random_labels[index],0)
			if img.shape[1] != self.width and img.shape[0] != self.height:
				img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_AREA)
			if label.shape[1] != self.width and label.shape[0] != self.height:
				label = cv2.resize(label, (self.width, self.height), interpolation = cv2.INTER_NEAREST)
			macara = mask_expanded[index, :, :, 0] 

			if train and augmenter and random.random()<0.90:
				seq_image2, seq_image, seq_label, seq_mask = get_augmenter(name=augmenter, c_val=self.ignore_label)

				#apply some contrast  to de rgb image
				img=img.reshape(sum(((1,),img.shape),()))
				img = seq_image2.augment_images(img)  
				img=img.reshape(img.shape[1:])

				if random.random()<0.90:
					#Apply shifts and rotations to the mask, labels and image
					
					# Reshapes for the AUGMENTER framework
					# the loops are due to the external library failures
					
					cuenta_ignore=sum(sum(sum(img==self.ignore_label)))
					cuenta_ignore2=cuenta_ignore
					i=0
					while abs(cuenta_ignore2-cuenta_ignore)<5 and i<15:
						img=img.reshape(sum(((1,),img.shape),()))
						img = seq_image.augment_images(img)  
						img=img.reshape(img.shape[1:])
						cuenta_ignore2=sum(sum(sum(img==self.ignore_label)))
						i = i+ 1

					cuenta_ignore=sum(sum(label==self.ignore_label))
					cuenta_ignore2=cuenta_ignore
					i=0
					while cuenta_ignore2==cuenta_ignore and i<15:
						label=label.reshape(sum(((1,),label.shape),()))
						label = seq_label.augment_images(label)
						label=label.reshape(label.shape[1:])
						cuenta_ignore2=sum(sum(label==self.ignore_label))
						i = i+ 1

					cuenta_ignore=sum(sum(macara==self.ignore_label))
					cuenta_ignore2=cuenta_ignore
					i=0
					while cuenta_ignore2==cuenta_ignore and i<15:
						macara=macara.reshape(sum(((1,),macara.shape),()))
						macara = seq_mask.augment_images(macara)
						macara=macara.reshape(macara.shape[1:])
						cuenta_ignore2=sum(sum(macara==self.ignore_label))
						i = i+ 1


			if self.ignore_label and not validation:
				#ignore_label to value 0-n_classes and add it to mask
				mask_ignore = label == self.ignore_label
				macara[mask_ignore] = 0
				label[mask_ignore] = 0


			x[index, :, :, :] = img
			y[index, :, :] = label
			for i in xrange(mask_expanded.shape[3]):
				mask_expanded[index, :, :, i] = macara

		# the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
		a, b, c =y.shape
		y = y.reshape((a*b*c))
		if self.ignore_label and validation:
			y = to_categorical(y, num_classes=self.n_classes+1)
		else:
			y = to_categorical(y, num_classes=self.n_classes)
		y = y.reshape((a,b,c,self.n_classes)).astype(np.uint8)
		x = x.astype(np.float32) / 255.0 - 0.5
		return x, y, mask_expanded


	# Returns a random batch
	def _get_batch_rgb(self, size=32, train=True, augmenter=None):

		x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
		y = np.zeros([size], dtype=np.uint8)

		file_list = self.test_list
		folder = '/test/'
		if train:
			file_list = self.train_list
			folder = '/train/'

		# Get [size] random numbers
		random_files = [file_list[random.randint(0,len(file_list) - 1)] for file in range(size)]
		classes = [self.classes[file.split(folder)[1].split('/')[0]] for file in random_files]


		for index in range(size):
			img = cv2.imread(random_files[index])
			if img.shape[1] != self.width and img.shape[0] != self.height:
				img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_AREA)

			x[index, :, :, :] = img
			y[index] = classes[index]
		# the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
		y = to_categorical(y, num_classes=len(self.classes))
		# augmentation
		if augmenter:
			augmenter_seq = get_augmenter(name=augmenter)
			x = augmenter_seq.augment_images(x)
		x = x.astype(np.float32) / 255.0 - 0.5

		return x, y

	# Returns a random batch
	def _get_batch_GAN(self, size=32, train=True, augmenter=None):
		return self._get_batch_rgb(size=size, train=train, augmenter=augmenter)


	# Returns a random batch
	def _get_batch_DVS(self, size=32, train=True):
		# Yet to know how to manage this data
		pass


	# Returns a random batch
	def get_batch(self, size=32, train=True, index=None, augmenter=None, validation=False):
		if self.problemType == 'classification':
			return self._get_batch_rgb(size=size, train=train, augmenter=augmenter)
		elif self.problemType == 'GAN':
			return self._get_batch_GAN(size=size, train=train, augmenter=augmenter)
		elif self.problemType == 'segmentation':
			return self._get_batch_segmentation(size=size, train=train, augmenter=augmenter, index=index, validation=False)
		elif self.problemType == 'DVS':
			return self._get_batch_DVS(size=size, train=train)

	def median_frequency_exp(self):

		for image_label_train in self.label_train_list:
			image = cv2.imread(image_label_train,0)

			for label in xrange(self.n_classes):
				self.freq[label] = self.freq[label] + sum(sum(image == label))
		return np.median(self.freq)/self.freq

		
if __name__ == "__main__":
	'''
	loader = Loader('./dataset_rgb')
	print(loader.classes)
	x, y =loader.get_batch(size=2)
	print(y)
	'''
	loader = Loader('./camvid', problemType = 'segmentation', ignore_label=12, n_classes=12)
	x, y, mask =loader.get_batch(size=50)#, augmenter='segmentation'
	print(x.shape)
	print(np.argmax(y,3).shape)
	print(mask.shape)
	for i in xrange(50):
		print(np.unique(np.argmax(y,3)*12))
		print(np.unique(mask[i,:,:]*255))
		print((np.argmax(y,3)*12).dtype)
		print(mask.dtype)
		cv2.imshow('x',x[i,:,:,:])

		for label in xrange(12):

			imagen_label=np.argmax(y,3)[i,:,:]
			imagen_label[imagen_label==label]=255
			imagen_label[imagen_label!=255]=0
			cv2.imshow(str(label),(imagen_label).astype(np.uint8))
			cv2.waitKey(0)
		cv2.imshow('x',x[i,:,:,:])
		cv2.imshow('y',(np.argmax(y,3)[i,:,:]*12).astype(np.uint8))
		cv2.imshow('mask',mask[i,:,:]*255)
		cv2.waitKey(0)
	cv2.destroyAllWindows()
	x, y, mask =loader.get_batch(size=3, train=False)


 