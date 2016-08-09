#!/usr/bin/python

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

movie_genres = ["unknown","Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
occupation_check = ["administrator","artist","doctor","educator","engineer","entertainment","executive","healthcare","homemaker","lawyer","librarian","marketing","none","other","programmer","retired","salesman","scientist","student","technician","writer"]

# Male = 0 and Female = 1

def load_data(path = "ml-100k"):
    user_info = {}
    
    
    for line in open(path+'/u.user'):
        (u_id)=map(int, line.split('|')[0:1])
        (age) = map(int, line.split('|')[1:2])
        (gender)=map(str, line.split('|')[2:3])
        (occupation)=map(str, line.split('|')[3:4])
        if gender[0] == 'M':
            gender[0] = 0
        else:
            gender[0] = 1
        for i in range(21):
            if occupation[0] == occupation_check[i]:
                #if len(movies_gen) == 0:
                    #movies_gen[m_id[0]] = []
                    #movies_gen[0].append(movie_genres[i])
                occupation[0] = i

        user_info[u_id[0]] = []
        user_info[u_id[0]].append(age[0])
        user_info[u_id[0]].append(gender[0])
        user_info[u_id[0]].append(occupation[0])

    return user_info



    

if __name__ == "__main__":
    load_data()
