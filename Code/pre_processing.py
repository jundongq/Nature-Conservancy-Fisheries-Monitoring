import numpy as np
import os
import cv2
import h5py
import time
import glob
from random import randint
from sklearn.preprocessing import LabelEncoder

# data directory, which stores the augmented dataset
data_dir = 'Data/train_aug_256_144/'

# data in 8 folders with names of classes, return names of classes without considering
# '.DS_Store' file in the subdirectory
classes  = [s for s in os.listdir(data_dir) if not s.startswith('.')]

# encode the classes into numbers, from 0 to 7.
LE = LabelEncoder()
LE.fit(classes)
labels      = LE.transform(classes)
fish_labels = dict(zip(classes, labels))
# print fish_labels

new_img_size = (256, 144) # or (64, 36)

def preprocessing(fish_class):

	"""preprocess images by resize the images to smaller ones;
	and encode the labels into integers.

	parameters:
	fish_class  : a string, name of one class
	
	it returns stacked imgs in np.ndarray format and associated encoded labels
	"""

	# encode fish_class into integer
	fish_label  = fish_labels[fish_class]
	
	# return a list of image directories for each image
	img_handles = glob.glob(data_dir + fish_class + '/' + '*.jpg')
	
	# build an empty list to store each img as np.ndarray
	imgs   = []
	
	# build an empty list to store the encoded label for each image
	labels = []
	
	# iterate through all images in the fish_class folder
	for img_handle in img_handles:
	
		# read img as np.ndarray
		img = cv2.imread(img_handle)
		
		# resize it 
		cv2.resize( img, (new_img_width, new_img_height)
		img = cv2.resize(img, new_img_size, interpolation=cv2.INTER_CUBIC)
		store the img in format of np.ndarray into the imgs 
		imgs.append(img)
		
		# store a label in labels
		labels.append(fish_label)
	
	return imgs, labels

# time the preprocessing
t0 = time.time()

# build an empty list to store preprocessed data
preprocessed_data = []

# build an empty list to store labels
encoded_labels    = []

for num, fish in enumerate(classes):
	print num
	print 'Preprocessing imgs of fish %s' %(fish)
	print '----------------------------------------------------------------------'
	preprocessed_imgs, labels = preprocessing(fish)
	preprocessed_data.append(preprocessed_imgs)
	encoded_labels.append(labels)

t1 = time.time()
t  = t1 - t0

# make a list containing each individual img from a list of lists
preprocessed_data_list = [data for sublist in preprocessed_data for data in sublist]
label_list             = [data for sublist in encoded_labels for data in sublist]

print 'Preprocessing Completed!'
print 'The preprocessing process took %.2f mins.' %(round(t/60., 2))
print '----------------------------------------------------------------------'

print 'Saving Preprocessed imgs and corresponding labels into HDF5 format...'
# Save preprocessed data as hdf5 format
filename = 'Preprocessed_Aug_Dataset_%s_%s.h5' %new_img_size
with h5py.File(filename, 'w') as w_hf:
	w_hf.create_dataset("preprocessed_data",  data=preprocessed_data_list)
	w_hf.create_dataset('label', data=label_list)
print 'Saving Completed!'
print '----------------------------------------------------------------------' 
