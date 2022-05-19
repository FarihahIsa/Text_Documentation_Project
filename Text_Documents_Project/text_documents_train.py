#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:26:08 2022

@author: farihahisa
"""

import pandas as pd
import re
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
import datetime

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#%% path
URL ='https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKENIZER_JSON_PATH= os.path.join(os.getcwd(),'saved_models','tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(), 'logs')
OHE_SCALER_PATH = os.path.join(os.getcwd(),'saved_models', 'one_hot_encoder.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models','model.h5')
#%% EDA

# Step 1) Loading data

df = pd.read_csv(URL)

text = df['text']
text_dummy = text.copy() # X_train

category = df['category']
category_dummy = category.copy() # Y_train

# Step 2) Data Inspection

text_dummy[5] # ---> found some signs (',.-,£)
category_dummy[5] 

# Step 3) Data Cleaning
# to remove any signs
for index, text in enumerate(text_dummy):
    text_dummy[index] = re.sub('<.*?>', '', text)
    #text_dummy[index] = text.replace('<br />', '')


# to convert to lowercase and split it
for index, text in enumerate(text_dummy):
    text_dummy[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    
# Step 4) Feature selection --> no features need to be selected

# Step 5) Data Preprocessing

# Data vectorization for reviews

num_words = 10000  # total number of index in this dataset is 2224
oov_token = '<OOV>' 

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(text_dummy)

# to observe the number of words
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# to vectorize the sequence of text
text_dummy = tokenizer.texts_to_sequences(text_dummy)

# to save the tokenizer for deployment purpose

token_json = tokenizer.to_json()

with open(TOKENIZER_JSON_PATH,'w') as json_file:
    json.dump(token_json, json_file)


# pad sequences

pad_sequences(text_dummy, maxlen=200)

[np.shape(i)for i in text_dummy] # to check the number words inside the list

text_dummy = pad_sequences(text_dummy, 
                             maxlen=200,
                             padding='post',
                             truncating='post')

# one-hot encoder for categories

one_hot_encoder = OneHotEncoder(sparse=False)

category_dummy.unique()    

category_encoded = one_hot_encoder.fit_transform(np.expand_dims(category_dummy, 
                                                                 axis=-1))
# save one hot encoder

pickle.dump(one_hot_encoder, open(OHE_SCALER_PATH,'wb'))

# train-test-split
    
X_train, X_test, y_train, y_test =  train_test_split(text_dummy, 
                 category_encoded,
                 test_size=0.3, 
                 random_state=123)

  

#%% model creation

model = Sequential()
model.add(Embedding(num_words,64))
model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.summary()

#%% callbacks

log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%% compile and model fitting

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['acc'])

hist =model.fit(X_train, y_train, 
                epochs=5, 
                validation_data=(X_test, y_test),
                callbacks=[tensorboard_callback])

#%% model evaluation

# append approach

predicted=[]

for test in X_test:
  predicted.append(model.predict(np.expand_dims(test, axis=0)))
    
# preallocation of memory approach

predicted_advanced = np.empty([len(X_test),5])

for index, test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test, axis=0))

#%% model analysis

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1) 

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))      

#%% model deployment

model.save('model.h5')

#%% Discussion

# This model achieves about 0.92 of F1 accuracy score after Embedding and 
# Bidirectional layers are applied.


