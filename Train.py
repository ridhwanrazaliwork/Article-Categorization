#%% Import libraries
import os
import pandas as pd
import re
import numpy as np
import string
import statistics as st
import datetime
import json
import pickle

from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input,Sequential
from tensorflow.keras.layers import  Embedding, Bidirectional
from tensorflow.keras.utils import plot_model
#%% Define function
# Cleaning text
def clean_text(text):
    '''Make text lowercase, remove text in square bracket,
     remove punctuation, remove words containing numbers,
     remove HTML tags, remove non-alphabet characters'''
    for index, word in enumerate(text):
        text[index] = re.sub('<.*?>','',word)
        text[index] = re.sub('[^a-zA-Z]',' ',word).lower().split()
        text[index] = re.sub(r'\[.*?\]', '', word)
        text[index] = re.sub(r'[%s]' % re.escape(string.punctuation), '', word)
        text[index] = re.sub(r'\w*\d\w*', '', word)
        text[index] = re.sub('\\$|\\£','',word)
        text[index] = re.sub('[()]|[(.)]','',word)

# constant
LOGS_PATH = os.path.join(os.getcwd(),'Logs',datetime.datetime.now().
                    strftime('%Y%m%d-%H%M%S'))
TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','tokenizer.json')
OHE_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
early_callback = EarlyStopping(monitor='val_loss',patience=3)
#%% Data loading
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')

# backup df
df_copy = df

# Data Inspection
df.info()
df.describe().T
df.head()
# 5 categories, highest freq category is from sport

#checking counts of category
df['category'].value_counts()
#%% Data cleaning
# Check duplicates and NaNs
df.duplicated().sum()
df.isna().sum()
# No NaNs but 99 duplicated

# Drop duplicates
df= df.drop_duplicates()
df.duplicated().sum()

# check text column
print(df['text'][4])
print(df['text'][10])
# mostly clean from HTML tags, all lowercase etc

clean_text(df['text'])

# Define df features and target
category = df['category']
text = df['text']
# backup
text_backup = text.copy()
category_backup = category.copy()


# %% Data preprocessing

# create tokenizer
vocab_size = 10000
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index

# Printing word_index example
print(dict(list(word_index.items())[0:100]))

# tokenize
text_int = tokenizer.texts_to_sequences(text)

# Checking Median, mode, mean, max, min
median = np.median([len(text_int[i]) for i in range(len(text_int))])
mean = np.mean([len(text_int[i]) for i in range(len(text_int))])
mode = st.mode([len(text_int[i]) for i in range(len(text_int))])
max_words = np.max([len(text_int[i]) for i in range(len(text_int))])
min_words = np.min([len(text_int[i]) for i in range(len(text_int))])

print(f'Median: {median}\nMean: {mean}\nMode: {mode}\nMax: {max_words}\nMin: {min_words}')

# Median is chosen in this case, because it is less affected by outliers
# Which in the above data shown an outlier
max_len = median

# Padding the tokenized data
padded_text = pad_sequences(text_int,
                            maxlen=int(max_len),
                            padding='post',
                            truncating='post')
padded_text.shape
# Encode the target using OneHotEncoder
ohe = OneHotEncoder(sparse=False)
category_encode = ohe.fit_transform(np.expand_dims(category,axis=-1))
category_encode.shape

# Train test split
X_train,X_test,y_train,y_test = train_test_split(padded_text,category_encode,
                                                test_size=0.3,
                                                random_state=123)
# %% model development
input_shape = np.shape(X_train)[-1]
out_shape = np.shape(y_train)[-1]
embed_shape = 128

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(Embedding(vocab_size,embed_shape)) #need two dimension
model.add(Bidirectional(LSTM(128,return_sequences=True)))#LSTM will adjust be 3d
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(out_shape,activation='relu'))
model.summary()
plot_model(model,show_shapes=True,show_layer_names=True)

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['acc'])

#%%
hist = model.fit(X_train,y_train,epochs=160,
                    validation_data=(X_test,y_test),   
                    callbacks=[tensorboard_callback, early_callback])

# %%
y_pred = np.argmax(model.predict(X_test),axis=1)
y_actual = np.argmax(y_test,axis=1)

# %% Saving OHE, Tokenizer, and model
# Save
token_json = tokenizer.to_json()
with open(TOKENIZER_SAVE_PATH, 'w') as file:
    json.dump(token_json,file)
# OHE
with open(OHE_SAVE_PATH,'wb') as file:
    pickle.dump(ohe,file)
# Model
model.save(MODEL_SAVE_PATH)
# %%
# plot performance graph
print(hist.history.keys())
from DeepLearnModule import ModelHist_plot,Model_Analysis

ModelHist_plot(hist,'loss','val_loss','loss','val_loss')
ModelHist_plot(hist,'acc','val_acc','acc','val_acc')
