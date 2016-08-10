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
from sklearn.svm import SVC
clf = SVC(C=1.0, kernel='rbf')

#features_train = features_train[:len(features_train)/10]
#labels_train = labels_train[:len(labels_train)/10]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
from sklearn.metrics import accuracy_score
print "Accuracy Score: ", accuracy_score(labels_test,pred)

#print "Values: ", sum(pred == 1)


###best for C=1, and full dataset_size ===== 35%
'''
training time: 1464.42 s
prediction time: 101.125 s
Accuracy Score:  0.3537
'''


#########################################################


