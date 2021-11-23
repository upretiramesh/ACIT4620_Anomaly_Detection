#!/usr/bin/env python
from PIL import Image
import numpy as np
import pandas as pd
from glob import glob
import os

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, f1_score, classification_report


# function to load image 
def read_and_save_to_df(path):
    im = Image.open(path).convert('L')
    im = im.resize((50,50))
    im = np.array(im)/255
    data = im.flatten()
    return data


# load the train iamge
df_train = []
for img in glob('./bottle/train/good/*'):
    df_train.append(read_and_save_to_df(img))


# load the test image
df_test = []
real = []
for img in glob('./bottle/test/**/*'):
    df_test.append(read_and_save_to_df(img))
    real.append( 0 if 'good' in img else 1)


# train the PCA model on good image
pca = PCA(n_components=100)
df_train_reduced = pca.fit_transform(df_train)
df_train_inverse = pca.inverse_transform(df_train_reduced)
train_loss = np.sum((df_train - df_train_inverse) ** 2, axis=1)
threshold = np.max(train_loss)*5
print('Threshold: ', threshold)


# used the train pca model to transfrom and inserve transfrom the test dataset
df_test_reduced = pca.transform(df_test)
df_test_inverse =pca.inverse_transform(df_test_reduced)


# calculate the reconstruction loss
test_loss = np.sum((df_test - df_test_inverse) ** 2, axis=1)
predict = [1 if val>threshold else 0 for val in test_loss]


def print_metrics(real, predict):
    print('Accuracy: ', accuracy_score(real, predict))
    print('\nPrecision: ', precision_score(real, predict))
    print('\nrecall: ', recall_score(real, predict))
    print('\nf1_score: ', f1_score(real, predict))
    print('\nconfusion_matrix:\n ', pd.DataFrame(confusion_matrix(real, predict), index=[0, 1], columns=[0, 1]))
    print('\nclassification_report:\n ', classification_report(real, predict, digits=4))


# print different metrics
print_metrics(real, predict)
