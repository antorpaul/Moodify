# Testing out LSTM through medium article
# Much of this comes from the following medium article: https://medium.com/appening-io/emotion-classification-2d4ed93bf4e2
# -*- encoding: utf-8 -*-
import numpy as np

# dont really know which ones we'll need but might as well have them all
# This is just a slightly more organized version of the imports from the medium article
import sys
import os
import emoji

# Math stuff
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd

# Sklearn
import sklearn
from sklearn.model_selection import train_test_split

# Tensorflow
import tensorflow as tf

# Keras imports
import keras
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import LSTM, Bidirectional
from keras.models import Model, load_model, save_model
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,Flatten,Conv1D,MaxPooling1D
from keras.utils import np_utils
import time
import datetime

# Download stopwords nltk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# The lyric fields are very large so we have to increase field_size_limit
# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
import csv
csv.field_size_limit(sys.maxsize)

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices = my_devices, device_type='CPU')

# Constants
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 50

# Read in song information
dataFrame = pd.read_csv('data_moodsUPDATED.csv',
                        header=0, engine='python')

print("Column names:")
print(dataFrame.columns.tolist())

x = dataFrame['lyrics']  # Lyrics
y = dataFrame['mood']  # Mood

print(x[1:10], y[1:10])


# Preprocessing Text Data
# Removing punctuation, words that start with '@' and stop words
    # In our own system we would only look at lyrics so some of these ideas might not be as necessary for us
stop_words = set(stopwords.words('english'))
new_stop_words = set(stop_words)

# adding wouldlnt (wouldn't) type of words into stopwords list
for s in stop_words:
    new_stop_words.add(s.replace('\'',''))

stop_words = new_stop_words
print("Excluding stopwords ...")

# Removing @ from default base filter to remove that whole word, which might be considered as user or page name
    # Again in our own system we would only look at lyrics so this would not be as important
base_filters='\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '

word_sequences=[]

# Go through the content text
for i in x:
    i=str(i)
    i=i.replace('\'', '')
    newlist = [x for x in text_to_word_sequence(i,filters=base_filters, lower=True) if not x.startswith("@")]
    filtered_sentence = [w for w in newlist if not w in stop_words] 
    word_sequences.append(filtered_sentence)

# Tokenising words / Converting words to indices
tokenizer = Tokenizer()
tokenizer.fit_on_texts(word_sequences)
word_indices = tokenizer.texts_to_sequences(word_sequences)
word_index = tokenizer.word_index
print("Tokenized to Word indices as ")
print(np.array(word_indices).shape)

# Padding words
    # Pad out each input with 20 words each
    # Pad last empty input entry with unknown words if we run out of words
x_data = pad_sequences(word_indices, maxlen=MAX_SEQUENCE_LENGTH)
print("After padding data")
print(x_data.shape)

# Build Word Embeddings
    # Using pretrained glove vector
print("Loading Glove Vectors ...")

embeddings_index = {}
f = open(os.path.join('GloVe-master', 'vectors.txt'),'r',encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded GloVe Vectors Successfully')

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print("Embedding Matrix Generated : ",embedding_matrix.shape)

embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM, weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False)

# One Hot Encoding
    # 13 labels = 13 bits
label_encoder = sklearn.preprocessing.LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
print("Label Encoding Classes as ")
print(le_name_mapping)

y_data=np_utils.to_categorical(integer_encoded)
print("One Hot Encoded class shape ")
print(y_data.shape)

# Embedding Layers
    # Given word index, returns embedded word vector
    # With GloVe 50 D, embedding layer converts in the following way:
    # input (None, 20) => (Embedding Layer) => (None, 20, 50)

# Comment
# Building Model
model = keras.models.Sequential()
model.add(embedding_layer)
model.add(Conv1D(30, 1, activation="relu"))
model.add(MaxPooling1D(4))
model.add(LSTM(100, return_sequences=True))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(500, activation="relu"))
model.add(Dense(y_data.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])

print(model.summary)

# Training model
print("Split training and testing data with 0.25/0.75 split")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

batch_size = 64
num_epochs = 100
x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]


st=datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
# Define the checkpoint
    # As accuracy got better, save into memory that checkpoint
filepath="model_weights-improvement-{epoch:02d}-{val_acc:.6f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history=model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
# history=model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list)

scores = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', scores[1])

pyplot.plot(history.history['acc'],label='Training Accuracy')
# pyplot.plot(history.history['val_acc'],label='Validation Accuracy')

pyplot.legend()
pyplot.show()