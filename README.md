# Nature-Conservancy-Fisheries-Monitoring
I use the kaggle competition 'Nature Conservancy Fisheries Monitoring' as Capstone Project of Udacity ML Nanodegree

Only train data in the [competition](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) is used here, since it is the only dataset with labels.

1. Data imbalance
There are 3777 images in total in the train dataset. Almost half of the images belong to one class 'ALB', there is a serious data imbalance problem as shown in the figure below ![Train_data_sample_size](https://github.com/jundongq/Nature-Conservancy-Fisheries-Monitoring/blob/master/Train_data_sample_size.png)

2. Data augmentation
Store the [train dataset](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data) in directory 'Data/train'.

Run 'data_augmentation.py' to resize the original image to 256(width) x 144(height), and for every resized image (except for the ones in ALB subfolder), randomly generated severy augmented images by changing its hue, contrast, saturation, brightness and so on. Save all resized and augmented images into directory 'Data/train_aug_256_144/'.

3. Data preprocessing
Run 'pre_processing.py'
