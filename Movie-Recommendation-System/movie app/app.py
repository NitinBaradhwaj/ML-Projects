import pandas as pd
import streamlit as st
#Run 'pip install streamlit'
import pickle
import requests

movielist = pickle.load(open('movielist_dict.pkl','rb'))
movietowatch = pd.DataFrame(movielist)

st.title('Movie Recommendation System')
st.subheader('Welcome to MR-Movie Recommender')

selected_option = st.selectbox(
     'Enter the Movie Name',
movietowatch['title'].values)
st.write('You selected:', selected_option)

similar = pickle.load(open('silimarlist.pkl','rb'))
def recommendation(movie):
     index = movietowatch[movietowatch['title'] == movie].index[0]
     distances = similar[index]
     movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
     recommend = []
     for i in movies:
          recommend.append(movietowatch.iloc[i[0]].title)
     return recommend

if st.button('Recommend Movies to Watch'):
     recommedning = recommendation(selected_option)
     for i in recommedning:
          st.write('• Movie similiar to',selected_option,'are',i)


st.header('Hope you liked our recommendations')
st.subheader('Thanks for using our MR')