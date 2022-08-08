#!/usr/bin/env python
# coding: utf-8

# Movie Recommedation System

#importing Dependencies
import pandas as pd #reads CSV files
import numpy as np #linear Algebra 
import matplotlib.pyplot as plt #helps in  data visualization and graphical plotting
import seaborn as sns #Visualize Distributions
import os

# Project Flow
# Collecting Data --> Loading Data --> Preprocessing/Visualisation --> Building a Model --> Create a Website and Deploy

#loading datasets 
moviefile = pd.read_csv('~/Desktop/Datasets/movies.csv')
creditfile = pd.read_csv('~/Desktop/Datasets/credits.csv')
moviefile
creditfile
moviefile.head(3)
creditfile.head(3)

# We can merge both the dataframes on 'title' basis 
#as both the dataframes 
#contains almost same data
moviefile.shape
creditfile.shape

moviefile = moviefile.merge(creditfile,on='title')
moviefile.shape
moviefile.info()
moviefile.head(2)
moviefile['original_language'].value_counts()
moviefile['production_companies'].value_counts()
moviefile.head(1)


# Both the DataFrames are merged now
# Removing columns which are not used in our analysis 

moviefile.head()
moviefile.info()
moviefile = moviefile[['id','title','overview','genres','keywords','cast','crew']]
#Removing unwanted columns  
moviefile.head()
# Check for Missing/Null values and also check for duplicate values
moviefile.head()
moviefile.isnull().sum()
moviefile.dropna(inplace=True)
moviefile.isnull().sum()
moviefile.duplicated().sum()
moviefile.iloc[0].keywords
#iloc - helps us to select a specific row or column from the data set
moviefile.iloc[0].genres
moviefile.iloc[0].cast


# This is in format '[{"id": 28, "name": "Action"}, 
#{"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, 
#{"id": 878, "name": "Science Fiction"}]'

# We need to change the format to 
#'['Action','Adventure','Fantasy','Science Fiction']'

import ast
def convert(objects):
    ABC=[]
    for i in ast.literal_eval(objects):
        ABC.append(i['name'])
    return ABC

moviefile['keywords'] = moviefile['keywords'].apply(convert)
moviefile['genres'] = moviefile['genres'].apply(convert)
moviefile.head()

def convert1(objects):
    ABC=[]
    cotr=0
    for i in ast.literal_eval(objects):
        if cotr!=3:
            ABC.append(i['name'])
            cotr+=1
        else:
            break
    return ABC

moviefile['cast'] = moviefile['cast'].apply(convert)
moviefile.head()

#For now we have converted 'keywords','genres','cast' into a proper format
#Now We need to get CREW into a proper format
moviefile.iloc[0].crew
#We need to check for "DIRECTOR" only

import ast
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 
    
moviefile['crew'] = moviefile['crew'].apply(fetch_director)
moviefile.head()
moviefile.iloc[0].overview

moviefile.head()


# Now the data is in proper format
# Now remove the space between words like
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

moviefile['cast'] = moviefile['cast'].apply(collapse)
moviefile['crew'] = moviefile['crew'].apply(collapse)
moviefile['genres'] = moviefile['genres'].apply(collapse)
moviefile['keywords'] = moviefile['keywords'].apply(collapse)

#or you can use this method
#moviefile['genres'] = moviefile['genres'].apply(lambda a:[b.replace(" ","")for b in a])
#moviefile['keywords'] = moviefile['keywords'].apply(lambda a:[b.replace(" ","")for b in a])
#moviefile['cast'] = moviefile['cast'].apply(lambda a:[b.replace(" ","")for b in a])
#moviefile['crew'] = moviefile['crew'].apply(lambda a:[b.replace(" ","")for b in a])

moviefile['overview'] = moviefile['overview'].apply(lambda x:x.split())
# Now create a new column which will be concatenation of 
#overview column, genres column, keywords column, cast column, crew column
# Then create a new dataframe using the new column, title and id 

moviefile['tags'] = moviefile['overview'] + moviefile['genres'] + moviefile['keywords'] + moviefile['cast'] + moviefile['crew']
moviefile.head()

new_data_frame = moviefile[['id','title','tags']]
new_data_frame.head()
new_data_frame['tags'].apply(lambda x:" ".join(x))
new_data_frame['tags'] = new_data_frame['tags'].apply(lambda x:" ".join(x))
new_data_frame.head(6)
new_data_frame['tags'] = new_data_frame['tags'].apply(lambda x:x.lower())
new_data_frame.head()

mport nltk
from nltk .stem.porter import PorterStemmer
pos = PorterStemmer()

def stem(texts):
    x=[]
    for a in texts.split():
        x.append(pos.stem(a))
    return " ".join(x)

new_data_frame['tags'] = new_data_frame['tags'].apply(stem)


# We import "nltk" it is a NLP Library 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
from sklearn.metrics.pairwise import cosine_similarity

vectors = cv.fit_transform(new_data_frame['tags']).toarray()
vectors.shape
vectors
vectors[0]
len(cv.get_feature_names())
similar = cosine_similarity(vectors)
similar.shape
similar
similar[0]

len(sorted(list(enumerate(similar[0])),reverse=True,key=lambda x:x[1]))
new_data_frame[new_data_frame['title'] == 'Batman Begins'].index[0]
new_data_frame[new_data_frame['title'] == 'Avatar'].index[0]
new_data_frame[new_data_frame['title'] == 'The Dark Knight Rises'].index[0]

def recommend(movie):
    index = new_data_frame[new_data_frame['title'] == movie].index[0]
    distances = sorted(list(enumerate(similar[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_data_frame.iloc[i[0]].title)

# Type 'recommend(movie name)' 
# And the model will recommend you movies which are similar to that movie 

recommend('Avatar')
recommend('Iron Man 3')
recommend('The Dark Knight Rises')
recommend('John Carter')
recommend('Batman Begins')
recommend('Avengers: Age of Ultron')
recommend('Thor')

# Model Building is Done
# Next step is to create an app and deploy this model

import pickle
pickle.dump(new_data_frame,open('movie_list.pkl','wb'))
pickle.dump(similar,open('similarity.pkl','wb'))