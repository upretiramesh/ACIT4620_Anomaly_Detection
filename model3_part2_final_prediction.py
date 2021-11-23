#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, f1_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


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
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();
