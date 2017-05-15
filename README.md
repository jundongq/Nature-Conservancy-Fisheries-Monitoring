# Nature-Conservancy-Fisheries-Monitoring
I use the kaggle competition 'Nature Conservancy Fisheries Monitoring' as Capstone Project of Udacity ML Nanodegree

Only train data in the [competition](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) is used here, since it is the only dataset with labels.

### 1. Data imbalance

There are 3777 images in total in the train dataset. Almost half of the images belong to one class 'ALB', there is a serious data imbalance problem as shown in the figure below ![Train_data_sample_size](https://github.com/jundongq/Nature-Conservancy-Fisheries-Monitoring/blob/master/Train_data_sample_size.png)

### 2. Data augmentation

Store the [train dataset](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data) in directory 'Data/train'.

Run 'data_augmentation.py' to resize the original image to 256(width) x 144(height), and for every resized image (except for the ones in ALB subfolder), randomly generated severy augmented images by changing its hue, contrast, saturation, brightness and so on. Save all resized and augmented images into directory 'Data/train_aug_256_144/'.

### 3. Data preprocessing

Run 'pre_processing.py', to store the [preprocessed data](https://drive.google.com/open?id=0B2ifRtIZ8FKkOXN4aHZ6MkpGRGM) in h5 format. Every preprocessed image is in format of 144(height) x 256(width) x 3(channel).

### 4. Bottleneck features

Use VGG16's convolutional layers to precompute the features of the images. Use [preprocessed data](https://drive.google.com/open?id=0B2ifRtIZ8FKkOXN4aHZ6MkpGRGM) as input, firstly, random permute the preprocessed data and split the it into train, validation, and test dataset. For the three sub datasets, the computed bottleneck features are available as: (1) [bottleneck_features_256_144_train.npy](https://drive.google.com/open?id=0B2ifRtIZ8FKkRlpZWFh5akhwSDQ); (2) [bottleneck_features_256_144_validation.npy](https://drive.google.com/open?id=0B2ifRtIZ8FKkakZReDBhU2JNMGM) and (3) [bottleneck_features_256_144_test.npy](https://drive.google.com/open?id=0B2ifRtIZ8FKkVEVhaGd5VVU1M3M).
