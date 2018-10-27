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
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    
    logging.info("reading file {0}...this may take a while".format(input_file))
    max=20000
    with json_lines.open(input_file, 'rb') as f:
        for item in f:
            if max>0:
                #max=max-1
                yield item['s']
          
base_dir="./"      
data_file=base_dir +'gutenberg-poetry-v001.ndjson.gz'

#lines_of_alignment = list(read_input(data_file))
#logging.info("Done reading #versuri=" + str(len(lines_of_alignment)))

min_token_length=2

def create_headlist(doc):
    wl = []
    for token in doc:
        #print("token="+token.text, token.dep_, token.head.text, token.pos_,[child for child in token.children])
        children=[child for child in token.children if child.pos_ not in ["SPACE","PUNCT"]]
        childrenJoined=" ".join([child.text.lower() for child in token.children if child.pos_ not in ["SPACE","PUNCT"]])
        
        if token.dep_=="relcl": 
            wl.append(token.head.text.lower() + " " + childrenJoined + " " + token.text.lower())
        
        if len(token.text)>min_token_length and  token.dep_=="compound":#token.tag_ not in : 
            wl.append(token.text.lower() + " " + token.head.text.lower())
        
        #if token.dep_=="pobj" or token.dep_=="dobj"  or token.dep_=="ROOT":
        if(len(children) >=1):
            if token.pos_=="NOUN":
                expr= childrenJoined + " " + token.text.lower() 
            else: 
                expr= token.text.lower() + " " + childrenJoined
            if(len(expr.strip().split(" "))>1):
                wl.append(expr.strip())
        #if(len(children) >=1):
         #   wl.append(childrenJoined.strip())
    return wl


nlp.max_length=1000000000
wordlist_computed = []
write_path="./headwordlist.tsv"
path=write_path
with open(path, "w") as wf:
    count=0
    for line in lines_of_alignment:
        processed = nlp(line)
        for sentence in processed.sents:
            #print(">" +str(sentence))
            #wl = create_headlist(sentence)
            wordlist_computed = wordlist_computed +[]
        wf.write(line+"\t"+"**".join(wl)+"\n")
        count+=1
print("Done tokenizing-lex #lines=" + str(count))


metaphors_annotated=[]
wordlist=[]
path_metaphors="./metaphor_annotated.txt"
with open(path_metaphors) as f:
    for line in f:
        if("@y" in line):
            metaphor=line.split("@")[0]
            #print(metaphor)
            metaphors_annotated.append(metaphor)
            processed = nlp(metaphor)
            for sentence in processed.sents:
                #print(">" +str(sentence))
                wordlist = wordlist +create_headlist(sentence)


processed_file="./processed.tsv"
wordlist_g=[]
nblines=0
with open(processed_file, "r") as f:
    for line in f:
        nblines+=1
        comp=line.split("\t")[1]
        expr_list=comp.split("**")
        wordlist_g+=expr_list
print("read processed#lines=" + str(nblines) + "=>" + str(len(wordlist_g)))

#learning expressions instead of words for poetry generation
sequences_step = 1
seq_length=8

print(str(len(wordlist)))

import collections
word_counts = collections.Counter(wordlist)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

#size of the vocabulary
vocab_size = len(words)
print("vocab size: ", vocab_size)

vocab_file_name=file_name+"_seq_length"+str(seq_length)
save_dir=basePath+"models/"
vocab_file = os.path.join(save_dir, vocab_file_name +"_words_vocab.pkl")
#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)

sequences = []
next_words = []
for i in range(0, len(wordlist) - seq_length, sequences_step):
    sequences.append(wordlist[i: i + seq_length])
    next_words.append(wordlist[i + seq_length])
    #print(wordlist[i: i + seq_length])
    #print(wordlist[i + seq_length])

print('nb sequences:', len(sequences))

#one-hot encoding
X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1


hidden_units=1024
learning_rate = 0.0001#learning rate
#model = bidirectional_lstm_model(seq_length, vocab_size)
batch_size = 512 # minibatch size
num_epochs = 60 # number of epochs

modelName=vocab_file_name+ "_hunits"+str(hidden_units)+"_numEpochs"+str(num_epochs)

from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True,input_shape=(seq_length, vocab_size))))
model.add(Dropout(0.4))  
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True,input_shape=(seq_length, vocab_size))))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True,input_shape=(seq_length, vocab_size))))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(hidden_units)))
model.add(Dropout(0.4))         
model.add(Dense(vocab_size, activation='softmax'))

optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
print("model built!")

from keras.callbacks import ModelCheckpoint

callbacks=[EarlyStopping(patience=50, monitor='val_loss'),
           ModelCheckpoint(filepath=save_dir + "/" + modelName+'text_gen_model.{epoch:02d}-{val_loss:.2f}.hdf5',\
                           monitor='val_loss', verbose=0, mode='auto', period=5)]
print("Now Train...")
history = model.fit(X, y,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=num_epochs,
                 callbacks=callbacks,
                 validation_split=0.2)

print("test loading vocabulary...")
with open(os.path.join(save_dir, vocab_file_name+'_words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)
vocab_size = len(words)
print("ok")

#save the model
full_path=save_dir + "/" + modelName+'_model.h5'
full_path_weights=save_dir + "/" + modelName+'_mode_weights.h5'
model.save_weights(full_path_weights)
model.save(full_path)
from keras.models import load_model
print("loading model..." + modelName)
mmodel=load_model(full_path)
print(mmodel.summary())

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_from_imagination(seed_sentences, gen_number_words, temperature):
    
    generated =""
    sentence = []
    previous_word=""
    
    count_seq_idx=seq_length
    seed = nlp(seed_sentences)
    for token in seed:
        if(token.pos_ not in ["SPACE","PUNCT"] and count_seq_idx>0):
            sentence.append(token.text)
            count_seq_idx-=1
    original=sentence
            
    for i in range(words_number):
        #create the vector
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(sentence):
            #print(str(t) + " :: " + word)
            if word in vocab:
                x[0, t, vocab[word]] = 1.

        #calculate next word
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_word = vocabulary_inv[next_index]

        
        if(previous_word != next_word):
            #add the next word to the text
            generated += " " + next_word
        else:
            i=i-1 #generate some other word to compensate the lack of value at this step
            
        # shift the sentence by one, and and the next word at its end    
        sentence = sentence[1:] + [next_word]
        previous_word=next_word
    
    print(" ".join(original) + " >>> " + generated)

    return generated


words_number =seq_length-5
temperature=0.99









