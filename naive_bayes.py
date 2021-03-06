#!/usr/bin/python

    
import sys
from time import time
import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, labels_train, features_test, labels_test = preprocess.load_data()


dataset_size = len(features_train)
features_train = features_train.reshape(dataset_size,-1)

dataset_size_test = len(features_test)
features_test = features_test.reshape(dataset_size_test,-1)

#########################################################

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
predict = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy score: ", accuracy_score(labels_test, predict)

#########################################################