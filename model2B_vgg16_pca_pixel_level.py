#!/usr/bin/env python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, f1_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from image_slicer import slice
from glob import glob
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def prepare_image(path):
    # load an image from file
    image = load_img(path) # 224, 224

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the VGG model
    image = preprocess_input(image)
    
    return image


# load model
model = VGG16(input_shape=(100,100,3), include_top=False)


def feature_convergence(folder_path, model):
    data = defaultdict(list)
    for i in range(len(os.listdir(folder_path))):
        image = folder_path+f'{i:03d}.png'
        slice(image, 81)
        for j in range(1, 10):
            for k in range(1, 10):
                image = prepare_image(folder_path+f'{i:03d}_{j:02d}_{k:02d}.png')
                output = model.predict(image)
                output = output.flatten()
                data[f'{j:02d}_{k:02d}'].append(output)
        for file in glob(folder_path+f'{i:03d}_*.png'):
            os.remove(file)
    return data


# folder location of dataset
train_path = 'bottle/train/good/'
test_bs = 'bottle/test/broken_small/'
test_bl = 'bottle/test/broken_large/'
test_g = 'bottle/test/good/'
test_con = 'bottle/test/contamination/'


# test small broken
test_bs_data = feature_convergence(test_bs, model)


# broken large
test_bl_data = feature_convergence(test_bl, model)


# test contamination
test_con_data = feature_convergence(test_con, model)


# test good image
test_g_data = feature_convergence(test_g, model)


# train good image
train_data = feature_convergence(train_path, model)


# apply PCA on each slide of good image
train_loss = []
test_bs_loss = []
test_bl_loss = []
test_con_loss = []
test_good_loss = []

for j in range(1, 10):
    for k in range(1, 10):
        
        pca = PCA(n_components=100)
        
        # train loss
        lat = pca.fit_transform(train_data[f'{j:02d}_{k:02d}'])
        inv = pca.inverse_transform(lat)
        train_loss.append(np.sum((np.array(train_data[f'{j:02d}_{k:02d}'])-inv)**2, axis=1))
        
        # test small broken
        lat = pca.transform(test_bs_data[f'{j:02d}_{k:02d}'])
        inv = pca.inverse_transform(lat)
        test_bs_loss.append(np.sum((np.array(test_bs_data[f'{j:02d}_{k:02d}'])-inv)**2, axis=1))
        
        # test large broken
        lat = pca.transform(test_bl_data[f'{j:02d}_{k:02d}'])
        inv = pca.inverse_transform(lat)
        test_bl_loss.append(np.sum((np.array(test_bl_data[f'{j:02d}_{k:02d}'])-inv)**2, axis=1))
        
        # test large broken
        lat = pca.transform(test_con_data[f'{j:02d}_{k:02d}'])
        inv = pca.inverse_transform(lat)
        test_con_loss.append(np.sum((np.array(test_con_data[f'{j:02d}_{k:02d}'])-inv)**2, axis=1))
        
        # test large broken
        lat = pca.transform(test_g_data[f'{j:02d}_{k:02d}'])
        inv = pca.inverse_transform(lat)
        test_good_loss.append(np.sum((np.array(test_g_data[f'{j:02d}_{k:02d}'])-inv)**2, axis=1))


# store reconstruction loss
pd.DataFrame(train_loss).T.to_csv('train_loss.csv', index=False)
pd.DataFrame(test_bs_loss).T.to_csv('test_bs_loss.csv', index=False)
pd.DataFrame(test_bl_loss).T.to_csv('test_bl_loss.csv', index=False)
pd.DataFrame(test_con_loss).T.to_csv('test_con_loss.csv', index=False)
pd.DataFrame(test_good_loss).T.to_csv('test_good_loss.csv', index=False)


# load the reconstruction loss of pixel-level
train = pd.read_csv('train_loss.csv')
test_bs = pd.read_csv('test_bs_loss.csv')
test_bl = pd.read_csv('test_bl_loss.csv')
test_con = pd.read_csv('test_con_loss.csv')
test_good = pd.read_csv('test_good_loss.csv')


# calculate the mean loss of features
test_good_mean = np.mean(test_good, axis=1)
test_bs_mean = np.mean(test_bs, axis=1)
test_bl_mean = np.mean(test_bl, axis=1)
test_con_mean = np.mean(test_con, axis=1)


# Let choose highest value of test good image as threshold
threshold = np.max(test_good_mean)


# test good image
predit = [0 if val<= threshold else 1 for val in test_good_mean]
real = [0]*len(test_good)

# test bs small
predit.extend([0 if val<= threshold else 1 for val in test_bs_mean])
real.extend([1]*len(test_bs_mean))

# test bl
predit.extend([0 if val<= threshold else 1 for val in test_bl_mean])
real.extend([1]*len(test_bl_mean))

# test bl
predit.extend([0 if val<= threshold else 1 for val in test_con_mean])
real.extend([1]*len(test_con_mean))


def print_metrics(real, predict):
    print('Accuracy: ', accuracy_score(real, predict))
    print('\nPrecision: ', precision_score(real, predict))
    print('\nrecall: ', recall_score(real, predict))
    print('\nf1_score: ', f1_score(real, predict))
    print('\nconfusion_matrix:\n ', pd.DataFrame(confusion_matrix(real, predict), index=[0, 1], columns=[0, 1]))
    print('\nclassification_report:\n ', classification_report(real, predict, digits=4))


# print metrics

print_metrics(real, predit)


# roc_curve
fpr1, tpr1, thresh1 = roc_curve(real, test_good_mean.append(test_bs_mean).append(test_bl_mean).append(test_con_mean))


# auc scores
auc_score = roc_auc_score(real, test_good_mean.append(test_bs_mean).append(test_bl_mean).append(test_con_mean))

print('AUC score: ', auc_score)


# plot the aur curve
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();

