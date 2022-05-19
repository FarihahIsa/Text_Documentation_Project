#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:15:58 2022

@author: adik
"""

#%%% Import

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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#%%% path

URL ='https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKENIZER_JSON_PATH= os.path.join(os.getcwd(),'saved_models','tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(), 'logs')
OHE_SCALER_PATH = os.path.join(os.getcwd(),'saved_models', 'one_hot_encoder.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models','model.h5')

#%% 

df = pd.read_csv(URL)
text = df['text']
category = df['category']

#%% 

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def remove_signs(self,data):

        for index, text in enumerate(data):
            text[index] = re.sub('<.*?>', '', text)
            
            return data

    def lower_split(data):
        
        for index, text in enumerate(data):
            text[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()
            
            return data
        
    def sentiment_tokenizer( self, data,token_save_path,num_words=10000,
                            oov_token='<OOV>', prt=False):
        
        # tokenizer to vectorize the words
            tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
            tokenizer.fit_on_texts(data)
        
        # to save the tokenizer for deployment purpose
            token_json = tokenizer.to_json()
        
            with open(TOKENIZER_JSON_PATH,'w') as json_file:
                json.dump(token_json, json_file)
            
        # to observe the number of words
            word_index = tokenizer.word_index
        
            if prt == True:
            # to view the tokenized words
            # print (word_index)
                print(dict(list(word_index.items())[0:10]))
            
        # to vectorize sequences of text
            data = tokenizer.texts_to_sequences(data)
            
            return data
    
    def sentiment_pad_sequences(self, data):
        
        return pad_sequences(data,maxlen=200, padding='post',
                                     truncating='post')
    
    def one_hot_encoder(data):
    
        one_hot_encoder = OneHotEncoder(sparse=False)

        category.unique()    

        category_encoded = one_hot_encoder.fit_transform(np.expand_dims(category, 
                                                                         axis=-1))
        # save one_hot_encoder pickle

        pickle.dump(one_hot_encoder, open(OHE_SCALER_PATH,'wb'))
        
        return data
        
    def train_test_split(data):
        X_train, X_test, y_train, y_test =  train_test_split(text, 
                         category_encoded,
                         test_size=0.3, 
                         random_state=123)
        
        return data
        
class ModelCreation():
     
    def lstm_layer(self, num_words, nb_categories,embedding_output=64, nodes=32, 
                    dropout=0.2):
                        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output)) # add the enbedding layer
        model.add(Bidirectional(LSTM(nodes,return_sequences=True))) # added bidirectonal
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model

class ModelComfit():
    
    def model_compile(data):
    
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['acc'])
        
        return data
     
    def model_fit (self,data):

        hist =model.fit(X_train, y_train, 
                        epochs=5, 
                        validation_data=(X_test, y_test),
                        callbacks=[tensorboard_callback])
        
        return data
            
class ModelEvaluation():
    
    def append_approach(data):
        
        predicted=[]

        for test in X_test:
          predicted.append(model.predict(np.expand_dims(test, axis=0)))

    def preallocation_memory(data):  
        
        predicted_advanced = np.empty([len(X_test),5])

        for index, test in enumerate(X_test):
            predicted_advanced[index,:] = model.predict(np.expand_dims(test, axis=0))
            
        return data
        
class ModelAnalysis():
    
    def model_analysis_report (data):
        
        y_pred = np.argmax(predicted_advanced, axis=1)
        y_true = np.argmax(y_test, axis=1) 

        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))  
        
        return data
    
    

