#!/usr/bin/python

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

movie_genres = ["unknown","Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

def load_data(path = "ml-100k"):
    
    prefs={}
    for line in open(path+'/u.data'):
        (user)=map(int, line.split('\t')[0:1])
        (movieid)=map(int, line.split('\t')[1:2])
        (rating)=map(int, line.split('\t')[2:3])
        prefs.setdefault(user[0],{})
        prefs[user[0]][movieid[0]]=float(rating[0])
        print prefs
        x = raw_input()
    
    for line in open(path+'/u.user'): 
        features = {}
        (u_id,age, gender, occupation)=line.split('|')[0:4]
        features[u_id] = [age, gender, occupation]

    return prefs


if __name__ == "__main__":
    load_data()