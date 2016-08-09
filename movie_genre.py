#!/usr/bin/python

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

movie_genres = ["unknown","Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

def load_data(path = "ml-100k"):
    
    movies_gen = {}
    movies={}
    genres= {}
    #m_id = {}
    
    for line in open(path+'/u.item'):
        (m_id)=map(int, line.split('|')[0:1])
        (genres) = map(int, line.split('|')[5:24])
        movies_gen[m_id[0]] = []
        '''
        for i in range(19):
            if genres[i]==1:
                #if len(movies_gen) == 0:
                    #movies_gen[m_id[0]] = []
                    #movies_gen[0].append(movie_genres[i])
        '''
        movies_gen[m_id[0]].append(genres)
        '''
        print movies_gen
        x = raw_input()
        '''

    return movies_gen


if __name__ == "__main__":
    load_data()

    
