import tensorflow as tf
import numpy as np
import os
import cv2
import glob
import time
from keras.preprocessing.image import ImageDataGenerator


# data directory
data_dir = 'Data/train/'


# data in 8 folders with names of classes, return names of classes without considering
# '.DS_Store' file in the subdirectory
classes  = [s for s in os.listdir(data_dir) if not s.startswith('.')]

img_nb = []
for fish_class in classes:
	img_nb_fish_class  = len(glob.glob(data_dir + fish_class + '/' + '*.jpg'))
	img_nb.append(img_nb_fish_class)

img_aug_fold = np.round(np.max(img_nb)/[float(i) for i in img_nb])

print classes
print 'Image number for each class:'
print img_nb
print 'Fold of image augmentation for each class:'
print img_aug_fold - 1
print 'Image number for each class after augmentation:'
print [np.int(reduce(lambda x, y: x*y, item)) for item in zip(img_nb, img_aug_fold)]


fish_class_aug_fold = zip(classes, img_aug_fold)

# resize the original images to smaller ones, in shape of cols (width) and rows (height)
new_img_size = (256, 144)

def resize(img_handle, new_img_size, data_aug_dir):
	# read an image through its handle
    img = cv2.imread(img_handle)
    
    # resize the image to smaller size
    img = cv2.resize(img, new_img_size, interpolation=cv2.cv.CV_INTER_LINEAR)
    
    # write the resized img into new directory
    cv2.imwrite(os.path.join(data_aug_dir, '%s_resized.jpg' %(img_handle.split('/')[-1][:-4])), img)
    
    return img


def data_augmentation(img_handle, fish_class, nb_fold):
    """
    This function is to generate synthetic pics for each class
    
    parameters:
    img_handle: a path for each input img
    fish_class: name of each class in this problem, such as 'ALB', 'BET' and so on
    nb_fold:    an integer which indicates the number of folds that should run for each class
    for generating the same number of images for each class.
    
    It returns resized one original image, and its corresponding distorted images that are
    processed through tf.image, and keras.preprocessing.ImageDataGenerator
    """
    
    # data directory to store augmented data, including resized imgs
    data_aug_dir = 'Data/train_aug_256_144/' + fish_class

    # check if the directory exists, otherwise, make a new directory to store resized and augmented images
    if not os.path.exists(data_aug_dir):
        os.makedirs(data_aug_dir)
    
    # if nb_fold = 1, the original images of corresponding fish_class only need to be resized
    if nb_fold == 1:
    	resize(img_handle, new_img_size, data_aug_dir)
    
    # if nb_fold > 1, new augmented data are needed
    elif nb_fold > 1:
    	
    	# the resized img as input for tf.image() makes the code run faster.
    	img = resize(img_handle, new_img_size, data_aug_dir)

    	# randomly adjust the hue of the img
    	img = tf.image.random_hue(img, max_delta=0.3)
    
    	# randomly adjust the contrust
    	img = tf.image.random_contrast(img,lower=0.3, upper=1.0)
    
    	# randomly adjust the brightness
    	img = tf.image.random_brightness(img, max_delta=0.2)
    
   		# randomly adjust the saturation
    	img = tf.image.random_saturation(img, lower=0.0, upper=2.0)
    
    	with tf.Session() as session, tf.Graph().as_default():
        	# this output is np.ndarray
        	img = session.run(img)
    
    	datagen = ImageDataGenerator(
        	zca_whitening = False,
        	rotation_range=45,
        	width_shift_range=0.2,
        	height_shift_range=0.2,
        	rescale = 1./255,
        	shear_range=0.2,
        	zoom_range=0.2,
        	horizontal_flip=True,
        	fill_mode='nearest')

    	x = img.reshape((1,) + img.shape)  # this is a Numpy array with shape (1, 3, height, width)

    	# the .flow() command below generates batches of randomly transformed images
    	# and saves the results to the `save_to_dir/` directory
    	i = 0
    	for batch in datagen.flow(x, batch_size=1, save_to_dir = data_aug_dir, \
    								save_prefix=fish_class, save_format='jpg'):
    		i += 1
    		if i > nb_fold-1:
    			break


def run(X):
    """
    This function run the data_augmentation, to produce sythetic data for training.
    
    parameter:
    X: a list of tuples. Each tuple contain two elements (fish_class, augmentation_fold)
    """
    
    fish_class = X[0]
    aug_fold   = X[1]
    
    print fish_class, aug_fold
    
    # grab all image handles in the corresponding fish class folder
    img_handles = glob.glob(data_dir + fish_class + '/' + '*.jpg')
	
	# run data_augmentation for each image of each fish class
    for img_handle in img_handles:
        data_augmentation(img_handle, fish_class, aug_fold)

###
print 'Start Generating Synthetic Data...'
t0 = time.time()

for i in range(len(fish_class_aug_fold)):
    run(fish_class_aug_fold[i])

t1 = time.time()
t = t1 - t0
print 'Synthetic Data Generation Completed!'
print 'The preprocessing process took %.2f mins.' %(round(t/60., 2))
