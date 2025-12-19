import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.title('News Classification')

#load stopwords and stemmer
nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()

#load model
dbfile = open('LogisticRegression.pickle', 'rb')
model = pickle.load(dbfile)

#taking data from user and convert to dataframe
news = st.text_area("Enter News For Classification")
if st.button("Submit"):
    d = {'news':[news]}
    df = pd.DataFrame(d)
    df['news'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), df['news']))
    df['news'] = df['news'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    #predict the news
    result = model.predict(df['news'])[0]
    st.dataframe(df)
    st.write(result)