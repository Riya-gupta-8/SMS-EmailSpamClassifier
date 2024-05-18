import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('EMAIL/SMS CLASSIFIER')

#input_sms = st.text_input("Enter the message ")  #short  sms
input_sms = st.text_area("Enter the message ")  # long para
if st.button('Predict output'):
    #1. Preprocess
    transform_sms = preprocess_text(input_sms)
    #2. Vectorize
    vector_input = tfidf.transform([transform_sms])
    #3. Predict
    output = model.predict(vector_input)[0]
    #4. Display
    if output == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
