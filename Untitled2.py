#!/usr/bin/env python
# coding: utf-8




# In[3]:


import pandas as pd
df_mrg = pd.read_csv("assurances.csv")


# In[4]:


def label_sentiment(rating):
    if rating > 10:
        return 2  # Positive
    elif rating < 6:
        return 0  # Negative
    else:
        return 1  # Neutral

# Apply the function to the dataframe to create the 'Sentiment' column
df_mrg['Sentiment'] = df_mrg['Rating'].apply(label_sentiment)


# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Selecting the 'Comment Content cleaned' and 'Rating' columns
X = df_mrg['Comment Content cleaned']
y = df_mrg['Sentiment']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the French stopwords from NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords
french_stop_words = stopwords.words('french')

# Creating a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=french_stop_words, max_df=0.7)

# Fitting and transforming the training data, and transforming the test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Preparing the models for training (Naive Bayes and SVM for now)
nb_model = MultinomialNB()
svm_model = SVC(kernel='linear')

# Train the Naive Bayes model
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)

# Train the SVM model
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)





# In[26]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Prédiction de Sentiment")

# Zone de saisie du commentaire
commentaire = st.text_area("Saisissez votre commentaire ici:")
commentaire_tfidf = tfidf_vectorizer.transform([commentaire])


# Bouton pour effectuer la prédiction
if st.button("Prédire le Sentiment"):
    # Utiliser le modèle pour la prédiction
    prediction = svm_model.predict(commentaire_tfidf)

    # Afficher le résultat
    st.write(f"Sentiment prédit : {prediction[0]}")
    

