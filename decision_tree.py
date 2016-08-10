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


from sklearn import tree

print len(features_train), len(labels_train)

clf = tree.DecisionTreeClassifier(min_samples_split=50)
clf = clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print "Accuracy: ", accuracy_score(labels_test, y_pred)



###best accuracy = 35%, for split= 50

#########################################################


