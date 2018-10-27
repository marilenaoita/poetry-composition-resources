from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy

import numpy as np
import random

import sys
import os
from os import listdir
from os.path import isfile, join

import time
import codecs
import collections
from six.moves import cPickle

import re

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import nltk
from nltk.corpus import wordnet as wn

import codecs
import spacy
nlp = spacy.load('en')
nlp.max_length=1000000000

import json_lines

#set logger
import logging
file_name="gutenberg-embeddings"
myapp=file_name
logger = logging.getLogger(myapp)
hdlr = logging.FileHandler(myapp+'.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

import json_lines

def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    
    logging.info("reading file {0}...this may take a while".format(input_file))
    max=20000
    idx=0
    with json_lines.open(input_file, 'rb') as f:
        for item in f:
            if max>0 and idx%100==0:
                #max=max-1
                yield item['s']
            #idx+=1
                
data_file='./gutenberg-poetry-v001.ndjson.gz'

lines_of_alignment = list(read_input(data_file))

logging.info("Done reading #versuri=" + str(len(lines_of_alignment)))   

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text

# source text
data = "\n".join(lines_of_alignment)

# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# create line-based sequences
sequences = list()
for line in data.split('\n'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+2]
        sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
#y = to_categorical(y, num_classes=vocab_size)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dropout(0.5))   
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)
# evaluate model
print(generate_seq(model, tokenizer, max_length-1, 'fire', 4))
print(generate_seq(model, tokenizer, max_length-1, 'love', 4))



#hidden_units=512
#learning_rate = 0.001#learning rate
#model = bidirectional_lstm_model(seq_length, vocab_size)
#batch_size = 256 # minibatch size
#num_epochs = 20 # number of epochs

modelName=vocab_file_name+ "_hunits"+str(50)+"_numEpochs"+str(500)


from keras.models import load_model

# Save the weights
model.save_weights(save_dir + "/" + modelName+'_model_weights.h5')

# Save the model architecture
with open(save_dir + "/" + modelName+'_model_architecture.json', 'w') as f:
    f.write(model.to_json())
    
#save the model
model_path=save_dir + "/" + modelName+'_model.h5'
model.save(model_path)

same_model= load_model(model_path)


#load the model
from keras.models import model_from_json

# Model reconstruction from JSON file
with open(save_dir + "/" + modelName+'_model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(save_dir + "/" + modelName+'_model_weights.h5')

print("Loading model ok...")
print(model.summary())
















