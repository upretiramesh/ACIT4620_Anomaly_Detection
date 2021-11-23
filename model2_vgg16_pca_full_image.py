#!/usr/bin/env python

# example of using the vgg16 model as a feature extraction model 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

from tensorflow.keras.models import Model

from pickle import dump
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, f1_score, classification_report


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


# train good images
train_good = []
for p in glob('./bottle/train/good/*'):
    train_good.append(prepare_image(p))
    
# test small broken
test_small = []
for p in glob('./bottle/test/broken_small/*'):
    test_small.append(prepare_image(p))

# test large broken
test_large = []
for p in glob('./bottle/test/broken_large/*'):
    test_large.append(prepare_image(p))
    
# test contamination
test_con = []
for p in glob('./bottle/test/contamination/*'):
    test_con.append(prepare_image(p))
    
# test good 
test_good = []
for p in glob('./bottle/test/good/*'):
    test_good.append(prepare_image(p))


from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# load model
model = VGG16(input_shape=(900,900,3), include_top=False)

out = GlobalAveragePooling2D()(model.output)
model = Model(inputs=model.input, outputs=out)


def extract_features(data):
    result = []
    for i in range(len(data)): 
        features = model.predict(data[i])
        result.append(features.flatten())
    return pd.DataFrame(result)


# extract features using VGG16 model
df_train = extract_features(train_good)
df_test_small = extract_features(test_small)
df_test_large = extract_features(test_large)
df_test_con = extract_features(test_con)
df_test_good = extract_features(test_good)


# dimensional_reduction_using_PCA(train and reduced)
pca = PCA(n_components=100)
df_train_reduced = pca.fit_transform(df_train)


# reduced the dimension of test set as well using trained pca
df_test_small_reduced = pca.transform(df_test_small)
df_test_large_reduced = pca.transform(df_test_large)
df_test_con_reduced = pca.transform(df_test_con)
df_test_good_reduced = pca.transform(df_test_good)


# reconstruct the features using inverse transform of trained pca model
df_train_inverse = pca.inverse_transform(df_train_reduced)
df_test_small_inverse = pca.inverse_transform(df_test_small_reduced)
df_test_large_inverse = pca.inverse_transform(df_test_large_reduced)
df_test_con_inverse = pca.inverse_transform(df_test_con_reduced)
df_test_good_inverse = pca.inverse_transform(df_test_good_reduced)


# define threshold value based on training loss
train_loss = np.sum((df_train - df_train_inverse) ** 2, axis=1)
threshold = np.max(train_loss)*5
print('Threshold value: ', threshold)


# calculate loss of small broken
small_broken_loss = np.sum((df_test_small - df_test_small_inverse) ** 2, axis=1)
# prediction based on threshold
predict = [1 if val>threshold else 0 for val in small_broken_loss]
real = [1]*len(small_broken_loss)


# calculate loss of large broken
large_broken_loss = np.sum((df_test_large - df_test_large_inverse) ** 2, axis=1)
# prediction based on threshold
predict.extend([1 if val>threshold else 0 for val in large_broken_loss])
real.extend([1]*len(large_broken_loss))


# calculate loss of contamination
cont_loss = np.sum((df_test_con - df_test_con_inverse) ** 2, axis=1)
# prediction based on threshold
predict.extend([1 if val>threshold else 0 for val in cont_loss])
real.extend([1]*len(cont_loss))


# calculate loss of good test image
good_test_loss = np.sum((df_test_good - df_test_good_inverse) ** 2, axis=1)
# prediction based on threshold
predict.extend([1 if val>threshold else 0 for val in good_test_loss])
real.extend([0]*len(good_test_loss))

# define function to print different metrics
def print_metrics(real, predict):
    print('Accuracy: ', accuracy_score(real, predict))
    print('\nPrecision: ', precision_score(real, predict))
    print('\nrecall: ', recall_score(real, predict))
    print('\nf1_score: ', f1_score(real, predict))
    print('\nconfusion_matrix:\n ', pd.DataFrame(confusion_matrix(real, predict), index=[0, 1], columns=[0, 1]))
    print('\nclassification_report:\n ', classification_report(real, predict, digits=4))


# print different metrics
print_metrics(predict, real)

