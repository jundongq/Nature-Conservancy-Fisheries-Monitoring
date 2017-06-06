import cv2
import h5py
import time
import tables
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras import optimizers
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical


# Import dataset
# the h5 file contains preprocessed imgs (dataset_sizedataset_size, height, width, channel)
# and corresponding labels
hdf5_path_img = "Preprocessed_Aug_Dataset_64_36.h5"

print "Loading data...."
# open the h5 file to peek the number of rows of the dataset
with tables.openFile(hdf5_path_img, 'r') as data_set:
    X = data_set.root.preprocessed_data[:]
    N = len(data_set.root.preprocessed_data)
    y = data_set.root.label[:]
print "Data Loaded!"

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

X_train = X[train_idx]/255.
X_val   = X[val_idx]/255.
X_test  = X[test_idx]/255.

# one hot encode train and validation set labels
y_label = to_categorical(y)
y_train = y_label[train_idx]
y_val   = y_label[val_idx]

y_test  = y[test_idx]


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


# parameter settings
nb_classes = 8

nb_filters_conv_1  = 24
kernel_size_conv_1 = (5,5)

nb_filters_conv_2  = 64
kernel_size_conv_2 = (5,5)

nb_filters_conv_3  = 96
kernel_size_conv_3 = (5,5)

pool_size = (2,2)
input_shape = (36, 64, 3)

def Benchmark_LeNet():

	model = Sequential()
	model.add(Conv2D(nb_filters_conv_1, (kernel_size_conv_1[0], kernel_size_conv_1[1]),
	                        padding='valid',
	                        input_shape=input_shape,
	                        activation = 'relu',	                        
	                        kernel_initializer = 'glorot_normal',
	                        name = 'conv2d_1'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=pool_size,
							name = 'maxpooling_1'))

	model.add(Conv2D(nb_filters_conv_2, (kernel_size_conv_2[0], kernel_size_conv_2[1]),
							activation = 'relu',
							kernel_initializer = 'glorot_normal',
							name = 'conv2d_2'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=pool_size,
							name = 'maxpooling_2'))
	
	model.add(Conv2D(nb_filters_conv_3, (kernel_size_conv_3[0], kernel_size_conv_3[1]),
							activation = 'relu',
							kernel_initializer = 'glorot_normal',
							name = 'conv2d_3'))
	
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(84, activation = 'relu', kernel_initializer = 'glorot_normal', name = 'fully_connected_1'))

	model.add(Dense(nb_classes, activation = 'softmax'))
	sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
	# lr: learning rate
	# decay: learning rate decay over each update (iteration?)
	# momentum: parameter updates momentum
	# nesterov: boolean, whether to apply Nesterove momentum

	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[fmeasure])
	
	return model

model = Benchmark_LeNet()

print model.summary()

Benchmark_model_weights_path = 'Benchmark_Model_weights.h5'
t0 = time.time()
# train the model
checkpointer = ModelCheckpoint(Benchmark_model_weights_path, verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=128,
					validation_data = (X_val, y_val), callbacks=[checkpointer])
t1 = time.time()
t = t1-t0
print 'The Benchmark_LeNet took %.2f mins.' %(round(t/60., 2))
pred_prop = model.predict(X_test)
y_pred = model.predict_classes(X_test)
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

# summarize history for accuracy 
plt.plot(history.history['fmeasure'])
plt.plot(history.history['val_fmeasure'])
plt.title('model fmeasure')
plt.ylabel('fmeasure')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
