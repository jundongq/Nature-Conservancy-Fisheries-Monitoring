import cv2
import h5py
import tables
import time
import os
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

from sklearn.metrics import f1_score, classification_report, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import keras.backend as K

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras import optimizers
from keras import regularizers

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.utils.np_utils import to_categorical

# Import dataset
# the h5 file contains preprocessed imgs (dataset_sizedataset_size, height, width, channel)
# and corresponding labels
hdf5_path_img = "Preprocessed_Aug_Dataset_256_144.h5"

# open the h5 file to peek the number of rows of the dataset
with tables.openFile(hdf5_path_img, 'r') as data_set:
    N = len(data_set.root.preprocessed_data)
    y = data_set.root.label[:]

# Randomly shuffle the data
random_seed = 678
np.random.seed(random_seed)

# Create a randonly shuffled list of indices
perm = np.random.permutation(N)

# set both test data size and validation data size as 2500
test_size  = 2500
val_size   = 2500
train_size = N - test_size - val_size

test_idx  = perm[:test_size]
val_idx   = perm[test_size:test_size+val_size]
train_idx = perm[-train_size:]


def pop(self):
    '''Removes a layer instance on top of the layer stack.
    Credit: joelthchao, https://github.com/fchollet/keras/issues/2371
    '''
    if not self.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        self.layers.pop()
        if not self.layers:
            self.outputs = []
            self.inbound_nodes = []
            self.outbound_nodes = []
        else:
            self.layers[-1].outbound_nodes = []
            self.outputs = [self.layers[-1].output]
        self.built = False

# use VGG16 to compute the features
def save_bottleneck_features(data, data_size, data_idx):

    if data:
        print 'Loading %s dataset......' %(data)
        with tables.openFile(hdf5_path_img, 'r') as data_set:
            # Here we slice [:] all the data back into memory, then operate on it
            X_selected = data_set.root.preprocessed_data[data_idx, :, :, :]
            y_selected = data_set.root.label[data_idx]
        print 'Loading %s dataset completed!' %(data)
        print '%s dataset is %.2f MB ' %(data, round((X_selected.nbytes+y_selected.nbytes)*1e-6, 2))
    
    # build a generator, which generates batches of tensor image data with real-time
    # data augmentation. The data will be looped over (in batches) indefinitely..
	datagen = ImageDataGenerator(rescale=1. / 255)
	
	print 'Loading VGG16.....'
    # build the VGG16 network
	model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(144,256,3))
	print 'Loading VGG16 complete!'
	# get rid of the last MaxPooling layer of VGG16
	pop(model)
	print 'Removed the last MaxPooling layer of VGG16!'
	
	# use the flow method of generator
	generator = datagen.flow(X_selected, batch_size=1, shuffle=False)
    
	t0 = time.time()
	print 'Computing bottleneck features for %s dataset......' %(data)
    # use model (functional API) to access the predict_generator method
    # This method generates predictions for input samples from a data generator
	bottleneck_features = model.predict_generator(generator, data_size)
    
    # save generated features 
	filename = 'bottleneck_features_256_144_%s.npy' %(data)
	np.save(open(filename, 'w'), bottleneck_features)
	t1 = time.time()
	print 'Computing %s bottleneck features took %.2f mins.' %(data, round((t1-t0)/60., 2))

# define a f1 score metric
def fmeasure(y_true, y_pred):
    """According to the keras version 1.2.0 metrics source code, a custom f1 score metric
    is built here. The link for source code:
    https://github.com/fchollet/keras/blob/53e541f7bf55de036f4f5641bd2947b96dd8c4c3/keras/metrics.py
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives  = K.sum(K.round(K.clip(y_true, 0, 1)))
    # K.epsilon = 1e-7
    p = true_positives / (predicted_positives + K.epsilon())
    r = true_positives / (possible_positives + K.epsilon())
    
    beta = 1 # f1 measure
    bb   = beta**2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    
    return fbeta_score
    

# define a function to train the layers on top of the bottleneck features
def Top_FCN_Model(optimizer='sgd', init='glorot_uniform'):

	model = Sequential()
	
	model.add(BatchNormalization(input_shape = (9, 16, 512)))
	
	model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())

	model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((1,2)))
	
	model.add(Conv2D(8, (4,4), activation='relu', padding='same', kernel_initializer=init))
	model.add(GlobalAveragePooling2D())
	model.add(Activation('softmax'))
	
	# sgd = optimizers.SGD(lr=0.08, decay=1e-5, momentum=0.9, nesterov=True)
	# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[fmeasure])
	return model

# define a grid search function that take model object as input
def Grid_Search_Training(model):
    
    f1_scorer = make_scorer(f1_score, average='weighted')
    optimizers = ['sgd', 'rmsprop', 'adam']
    init = ['glorot_uniform', 'glorot_normal', 'uniform']
    epochs = [1, 20, 30]
    batches = [32, 64, 128]
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size = batches, init = init)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, scoring=f1_scorer)
    
    return grid

# assign a name for the h5 file that stores model weights
top_model_weights_path = 'Top_FCN_Model_weights.h5'

def load_train_val_data():
    """Load train and validation bottleneck features
    """
    train_data   = np.load(open('bottleneck_features_256_144_train.npy'))
    train_labels = to_categorical(y[train_idx])
    # train_labels = y[train_idx]

    validation_data   = np.load(open('bottleneck_features_256_144_validation.npy'))
    validation_labels = to_categorical(y[val_idx])
    # validation_labels = y[val_idx]
    return train_data, train_labels, validation_data, validation_labels

def load_test_data():
    """Load test bottleneck features
    """
    test_data = np.load(open('bottleneck_features_256_144_test.npy'))
    return test_data

def run():
    """This funtion loads train and validation data, and starts training process.
    """
	train_data, train_labels, validation_data, validation_labels = load_train_val_data()
	X = np.concatenate((train_data, validation_data), axis=0)
	y = np.concatenate((train_labels, validation_labels), axis=0)
	# print np.shape(X)
	# print np.shape(y)
	print 'Loaded train and validation data!'

	# create a model
	model = Top_FCN_Model()
	# serialize model to JSON
	model_json = model.to_json()
	with open("Top_FCN_Model.json", "w") as json_file:
	    json_file.write(model_json)
	print model.summary()
	
	model = KerasClassifier(build_fn = Top_FCN_Model, verbose=1)
	
	grid = Grid_Search_Training(model)
	
	print 'Start Training the model......'
	# checkpointer = ModelCheckpoint(top_model_weights_path, verbose=1, save_best_only=True)
	grid_result = grid.fit(X, y)
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

	
if os.path.isfile('bottleneck_features_256_144_train.npy') and os.path.isfile('bottleneck_features_256_144_validation.npy'):
	print 'Bottleneck features for training and validation datasets exist!'
	run()
	
else:
	print 'Computing bottleneck features for train and validation dataset...'
	save_bottleneck_features('train', train_size, train_idx)
	save_bottleneck_features('validation', val_size, val_idx)
	run()
	

print 'Saved top model weights!'

if os.path.isfile('bottleneck_features_256_144_test.npy') :
	print 'Bottleneck features for test dataset exist!'
	print 'Loaded test data!'
	test_data = load_test_data()
	print 'Computing predictions on test dataset...'
	model.load_weights(top_model_weights_path)
	pred_prop = model.predict(test_data)
	y_pred = model.predict_classes(test_data)
	
else:
	print 'Computing bottleneck features test dataset...'
	save_bottleneck_features('test', test_size, test_idx)
	print 'Computing predictions on test dataset...'
	model.load_weights(top_model_weights_path)
	pred_prop = model.predict(test_data)
	y_pred = model.predict_classes(test_data)	
	
y_test = y[test_idx]
# print pred_prop[:2]
print "The first 20 ground truth labels:"
print y_test[:20]
print "The first 20 predicted labelS:"
print y_pred[:20]
print "The F1 score is: ", f1_score(y_test, y_pred, average='weighted')
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# confusion matrix
labels = ['ALB(0)', 'BET(1)', 'DOL(2)', 'LAG(3)', 'NoF(4)', 'OTHER(5)', 'SHARK(6)', 'YFT(7)']
print classification_report(y_test, y_pred, target_names=labels)
print confusion_matrix(y_test, y_pred)

