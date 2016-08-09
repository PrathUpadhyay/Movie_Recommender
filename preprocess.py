#!/usr/bin/python

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import user_features, movie_genre
User_features = {} 
Movie_genre = {}
features_combined = []
ratings_combined = []

features_combined_train = []
ratings_combined_train = []

def load_data(path = "ml-100k"):
    User_features = user_features.load_data()
    Movie_genre = movie_genre.load_data()
    '''
    i = 1
    while 1:
        print User_features[i]
        print "\n"
        print Movie_genre[i]
        print "\n"
        i = i+1
        y = raw_input()
    '''

    counter = 0
    prefs={}
    for line in open(path+'/u1.base'):
        (user) = map(int, line.split('\t')[0:1])
        (movieid) = map(int, line.split('\t')[1:2])
        (rating) = map(int, line.split('\t')[2:3])
        prefs.setdefault(user[0],{})
        prefs[user[0]][movieid[0]] = float(rating[0])
        features_combined.append([])
        a = numpy.array(User_features[user[0]])
        b = numpy.array(Movie_genre[movieid[0]])
        a.resize(b.shape)
        features_combined[counter] = a + b
        ratings_combined.append(rating[0])
        counter+=1



        '''
        print features_combined
        print ratings_combined
        x = raw_input()
        '''
        
    '''
    for line in open(path+'/u.user'): 
        features = {}
        (u_id,age, gender, occupation)=line.split('|')[0:4]
        features[u_id] = [age, gender, occupation]
    '''

    counter_train = 0
    prefs={}
    for line in open(path+'/u1.test'):
        (user1) = map(int, line.split('\t')[0:1])
        (movieid1) = map(int, line.split('\t')[1:2])
        (rating1) = map(int, line.split('\t')[2:3])
        prefs.setdefault(user1[0],{})
        prefs[user1[0]][movieid1[0]] = float(rating1[0])
        features_combined_train.append([])
        a = numpy.array(User_features[user1[0]])
        b = numpy.array(Movie_genre[movieid1[0]])
        a.resize(b.shape)
        features_combined_train[counter_train] = a + b
        ratings_combined_train.append(rating1[0])
        counter_train+=1



    return numpy.array(features_combined), ratings_combined, numpy.array(features_combined_train), ratings_combined_train
    




def preprocess(words_file = "../tools/word_data.pkl", authors_file="../tools/email_authors.pkl"):
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "r")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "r")
    word_data = cPickle.load(words_file_handler)
    words_file_handler.close()

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)



    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)



    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=1)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### info on the data
    print "no. of Chris training emails:", sum(labels_train)
    print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test


if __name__ == "__main__":
    load_data()