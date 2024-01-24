#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install streamlit


# In[1]:


import joblib

joblib.__version__


# In[18]:


pip install --upgrade joblib


# In[2]:





# In[21]:


pip install numpy==1.22.1


# In[1]:


import numpy

numpy.__version__


# In[2]:


import streamlit as st
import pandas as pd
import sklearn.externals
import joblib


model = joblib.load("modele_apprentissage_supervise_USE.pkl")


# In[5]:


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the French stopwords from NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords
french_stop_words = stopwords.words('french')


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
st.title("Prédiction de Sentiment")

# Zone de saisie du commentaire
#commentaire = st.text_area("Saisissez votre commentaire ici:")
commentaire="Je suis extrêmement déçu."


# Bouton pour effectuer la prédiction
if st.button("Prédire le Sentiment"):
    # Utiliser le modèle pour la prédiction
    prediction = model.predict(commentaire_transforme)

    # Afficher le résultat
    st.write(f"Sentiment prédit : {prediction[0]}")


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
# Creating a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fitting and transforming the training data, and transforming the test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[6]:


pip install tensorflow-hub


# In[7]:


import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the Universal Sentence Encoder model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)


# Selecting the 'Comment Content cleaned' and 'Rating' columns
X = df_mrg['Comment Content cleaned']
y = df_mrg['Sentiment']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate embeddings for training and testing sentences
X_train_embeddings = embed(X_train)
X_test_embeddings = embed(X_test)

# Define a simple neural network model for classification
model_USE = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(512,)),  # Adjust input shape based on USE output size
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Adjust output units based on the number of classes
])

# Compile the model
model_USE.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_USE.fit(X_train_embeddings, y_train, epochs=5, batch_size=32, validation_split=0.1)


# In[18]:


import streamlit as st
import pandas as pd
import sklearn.externals
st.title("Prédiction de Sentiment")

# Zone de saisie du commentaire
#commentaire = st.text_area("Saisissez votre commentaire ici:")
commentaire="Je suis extrêmement déçu."


# Bouton pour effectuer la prédiction
#if st.button("Prédire le Sentiment"):
    # Utiliser le modèle pour la prédiction

prediction = model_USE.predict(commentaire)

    # Afficher le résultat
print(prediction)
    #.write(f"Sentiment prédit : {prediction[0]}")


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


# In[24]:


# Nouveau commentaire à prédire
nouveau_commentaire = "Je suis extrêmement déçu, l'assurance coûte trop cheret j'ai eu des sinistres."

# Prétraitement du nouveau commentaire (assurez-vous que le prétraitement est cohérent avec celui de l'entraînement)
#nouveau_commentaire_traite = preprocess_function(nouveau_commentaire)

# Transformation avec le vecteur TF-IDF
nouveau_commentaire_tfidf = tfidf_vectorizer.transform([nouveau_commentaire])

# Faire la prédiction avec le modèle Naive Bayes
nb_prediction = nb_model.predict(nouveau_commentaire_tfidf)

# Faire la prédiction avec le modèle SVM
svm_prediction = svm_model.predict(nouveau_commentaire_tfidf)

# Afficher les prédictions
print("Prédiction Naive Bayes :", nb_prediction[0])
print("Prédiction SVM :", svm_prediction[0])


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
    

