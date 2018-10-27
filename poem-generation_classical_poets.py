# coding: utf-8
import logging
logger=logging.getLogger("supp-material-classicalpoets")
hdlr=logging.FileHandler("./supp-material-classicalpoets.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
# In[38]:


#LOAD#
from annoy import AnnoyIndex
D=512
NUM_TREES=20
batch_size=20000
index = AnnoyIndex(D)
base="./"
#annoy_20Trees_Batch15000_all_gutenberg
annoyIndexName=base+"annoy_"+str(NUM_TREES)+"Trees_Batch"+str(batch_size)+"_all_gutenberg.ann"
print(annoyIndexName)
index.load(annoyIndexName)
print('Loaded annoy Index: {}'.format(index.get_n_items()))#should be instead #versuri=3085117


# In[68]:


meta_index = AnnoyIndex(D)
#annoy_20Trees_Batch15000_all_gutenberg
meta_annoyIndexName=base+"annoy_"+str(10)+"Trees_Batch"+str(512)+"_metaphors_annotated.ann"
print(meta_annoyIndexName)
meta_index.load(meta_annoyIndexName)
print('Loaded annoy Index: {}'.format(meta_index.get_n_items()))#should be instead #versuri=3085117


# In[2]:


import nltk
from nltk.corpus import wordnet as wn

import codecs
import spacy
nlp = spacy.load('en')
nlp.max_length=1000000000

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


# In[3]:


#load the model
from keras.models import model_from_json

model_name="metaphors_seq_length8_hunits1024_numEpochs60_lrate001__model"
save_dir=base+"models/"
with open(save_dir + "/" + model_name+'_architecture.json', 'r') as f:
    model = model_from_json(f.read())
logger.info("Model reconstruction from JSON file, DONE")

model.load_weights(save_dir + "/" + model_name+'_weights.h5')
logger.info("# Load weights into the new model, DONE")

logger.info(model.summary())


# In[14]:


#load vocabulary
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[59]:


def generate_from_imagination(seed_sentence, gen_number_words, temperature):
    
    generated =""
    seed_expres = []
    previous_word=""
    
    count_seq_idx=seq_length
    seed_expressions = create_headlist(nlp(seed_sentence))
    for expr in seed_expressions:
        if(count_seq_idx>0):
            seed_expres.append(expr)
            count_seq_idx-=1
    original=seed_expres
            
    for i in range(words_number):
        #create the vector
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(seed_expres):
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
        sentence = seed_expres[1:] + [next_word]
        previous_word=next_word
    
    print(" ".join(original) + " >>> " + generated)

    return generated


# In[47]:

# In[48]:


#execute from here
#write poem to aligned-once


# In[49]:


#setup all parameters


# In[50]:


seq_length=8


# In[51]:


min_token_length=2
base_path=base
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

metaphors_annotated=[]
wordlist=[]
training_corpus_hint="metaphors"
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
                wl = create_headlist(sentence)
                wordlist = wordlist + wl

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

vocab_file_name=training_corpus_hint+"_seq_length"+str(seq_length)
save_dir=base_path+"models/"
vocab_file = os.path.join(save_dir, vocab_file_name +"_words_vocab.pkl")
#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)


# In[52]:


import re

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


# In[22]:


import time
import sys

def print_with_time(msg):
    print('{}: {}'.format(time.ctime(), msg))
    sys.stdout.flush()


# In[23]:


import tensorflow_hub as hub

# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)

print("Done load USE.")


# In[24]:


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
                
data_file=base+'gutenberg-poetry-v001.ndjson.gz'

lines_of_alignment = list(read_input(data_file))
logging.info("Done reading #versuri=" + str(len(lines_of_alignment)))   


# In[64]:


poems_original_path=base+"more-poems/"

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(poems_original_path) if isfile(join(poems_original_path, f))]
print(onlyfiles)


# In[54]:


#the training vocabulary needed to be loaded for making predictions

vocab_file_name="metaphors"+"_seq_length"+str(seq_length)#file_name""
with open(os.path.join(save_dir, vocab_file_name+'_words_vocab.pkl'), 'rb') as f:
    words, vocab, vocabulary_inv = cPickle.load(f)
vocab_size = len(words)
logger.info("vocab_size=" + str(vocab_size))


# In[69]:


metaphors_annotated=[]
#wordlist=[]
path_metaphors=base+"metaphor_annotated.txt"
with open(path_metaphors) as f:
    for line in f:
        if("@y" in line):
            metaphor=line.split("@")[0]
            #print(metaphor)
            metaphors_annotated.append(metaphor)
            #processed = nlp(metaphor)
            #for sentence in processed.sents:
                #print("\n>" +str(sentence))
                #wl = create_headlist(sentence)
                #wordlist = wordlist + wl


# In[74]:


from sklearn.metrics.pairwise import cosine_similarity
words_number =seq_length-2
temperatures=[0.80,0.90,1.09]
topNeighbors=6


# In[75]:
from pathlib import Path



def composition_experiments(topNeighbors,words_number, temperatures):
    for poem_file in onlyfiles:
        read_path=poems_original_path+poem_file
        write_path_imagination=base+"imagined/"+poem_file
        file_name=poem_file.replace(".txt", "")
        print(file_name)

        for temperature in temperatures: 
            imagined=[]
            original=[]
            with open(write_path_imagination, 'w') as wf:    
                with open(read_path, 'r') as f:
                    for line in f:
                        original.append(line)
                        result=generate_from_imagination(line, words_number,temperature)
                        wf.write(result+"\n")
                        imagined.append(result+"\n")
                        #print(result)

            logger.info("imagination done.")
            title=original[0]
            logger.info("title=" +title)

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                start_time = time.time()
                imagined_vectorized = sess.run(embed(imagined))
                print_with_time('vectorization of imagined, done!')
                i=0
                poemSoFar=[]
                poemSoFar.append(title)
                write_path_creation=base+"composed/seq"+str(words_number) + "_temp"+str(temperature)+"_topK"+str(topNeighbors)+poem_file
                my_file = Path(write_path_creation)
                if my_file.exists():
                    print("file " + poem_file + " has already been created.")
                else:
                    with open(write_path_creation, 'w') as wcf:  
                        #lines_of_alignment=metaphors_annotated
                        for imagine_vector in imagined_vectorized:
                            ann_sim_idx= index.get_nns_by_vector(imagine_vector, topNeighbors)
                            closestPoetryLines = [[nn,lines_of_alignment[nn]] for nn in ann_sim_idx]
                            candidates=[]
                            for candidate_align in closestPoetryLines:
                                similar_idx=candidate_align[0]
                                #print(similar_idx)
                                #generated.append(versuri[similar_idx-1])
                                candidate=candidate_align[1].replace("'","")
                                if candidate not in candidates:
                                    logger.info("aligned imagination :: " + candidate)
                                    candidates.append(candidate)
                                #generated.append(versuri[similar_idx+1])
                                #print("\n")
                                                                                                          
                            
                            poem_so_far=" ".join(poemSoFar)
                            logger.info("adding... " + poem_so_far)
                            candidates.append(poem_so_far)
                            
                            candidates_vectorized = sess.run(embed(candidates))
                            matrix=cosine_similarity(candidates_vectorized)
                            
                            vec_res=matrix[len(candidates_vectorized)-1]
                            logger.info(vec_res)
                            vec_res=np.delete(vec_res, (len(candidates_vectorized)-1))
                            logger.info(vec_res)
                            if(len(vec_res)>0):
                                maxim=np.argmax(vec_res)
                                selected_coherent=candidates[maxim]
                                if selected_coherent not in poemSoFar:
                                    logger.info("\n Got" + str(maxim) +" => selected_coherent is> " +selected_coherent)
                                    print("..appended to poemSoFar.")#+ ">"  + " for >>" + testSentences[i]
                                else: 
                                    vec_res=np.delete(vec_res, maxim)
                                    print(vec_res)
                                    new_candidates=np.delete(candidates, maxim)
                                    print(new_candidates)
                                    new_maxim=np.argmax(vec_res)
                                    selected_coherent=new_candidates[new_maxim]
                                    logger.info(str(new_maxim) +" => NEW selected_coherent is> " +selected_coherent)
                                #take next best if best already in poem
                                if selected_coherent not in poemSoFar:
                                    poemSoFar.append(selected_coherent)
                                else:
                                    poemSoFar.append("None")
                            else:
                                selected_coherent="None"
                            wcf.write(selected_coherent+"\t"+imagined[i] +"\t" +original[i] +"\n")
                            i+=1
                            print("\n")
                    logger.info("\n\npoem " + file_name + " re-COMPOSED.")
                    logger.info(poemSoFar)
    logger.info("all DONE!")

# In[76]:


composition_experiments(topNeighbors,words_number, temperatures)


# In[ ]:


import gc
gc.collect()


# In[ ]:


from keras import backend as K
K.clear_session()

