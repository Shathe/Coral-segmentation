# Semantic Segmentation

## Dependencies
* python (2.7)

Install with pip:

* opencv
* imgaug
* numpy
* tensorflow-gpu
* keras

## Training the classifier
```
python train_classifier.py
```

Parameters to take into account:

**--dataset", help="Dataset to train", default='./dataset_classif'**

--init_lr", help="Initial learning rate", default=1e-3

--min_lr", help="Initial learning rate", default=1e-5

--init_batch_size", help="batch_size", default=16

--max_batch_size", help="batch_size", default=16

--epochs", help="Number of epochs to train", default=40

--width", help="width", default=224

--height", help="height", default=224

--save_model,  help="whether to save the model weights while training",  default=1



## Datasets


* Classification datasets structure:
	* train
	    * class_1
	    	* image_1
	    * calss_2
	    	* image_1
	    * class_N
	    	* image_1
	* test
	    * class_1
	    	* image_1
	    * class_2
	    	* image_1
	    * class_N
	    	* image_1

* Segmentation dataset structure:
	* images
	    * train
	    	* image_1
	    * test
	    	* image_1
	* labels
	    * train
	    	* image_1
	    * test
	    	* image_1
