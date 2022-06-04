import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn

ps = PorterStemmer()


def text_transformer(a):
    sen=a.lower()
    token_word=nltk.word_tokenize(sen)
    new_token=[]
    
    for i in token_word:
        if i not in stopwords.words('English'):
            if i not in string.punctuation:
                if i.isalnum():### We will only keed Alpha numeric tokens..
                    new_token.append(ps.stem(i))
    return ' '.join(new_token)## Convert tokens into strings..


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.header('SMS Spam Classifier')

input_sms=st.text_area('Enter the message')

if st.button('Check Now :'):    

    #Pipeline to follow

    #1. Text Preprocessing
    transformed_sms=text_transformer(input_sms)
    #2. Vectorization
    vector_input=tfidf.transform([transformed_sms]).toarray()
    #3. Prediction
    prediction=model.predict(vector_input)[0]
    #4. Display

    if prediction==1:
        st.header('Spam')
    else:
        st.header('Not Spam')
