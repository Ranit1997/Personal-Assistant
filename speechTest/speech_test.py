# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:32:46 2017

@author: SANDIPAN
"""

import nltk
import tensorflow as tf
import random
import pickle
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag




def create_features(features, words):
    for i in range(0,len(words)):
        
        if  words[i][0] in wh_words:
            features[i] = 1
        elif words[i][1] in noun_tags:
            features[i] = 2
        elif words[i][1] in verb_tags:
            features[i] = 3   
        elif words[i][1] in adverb_tags:
            features[i] = 4     
        elif words[i][1] not in wh_words and words[i][1] not in noun_tags and words[i][1] and words[i][1] not in verb_tags:
            features[i] = 5
        
    return features
            

    
def make_featuresets(line):

        current_words = word_tokenize(line)
        tagged_words = st.tag(current_words)
        features = [0]*(max_length)
        features_in_line = create_features(features, tagged_words)
        
        return features_in_line

        

    

#config=tf.ConfigProto()
#config.gpu_options.allow_growth = True




java_path = "C:/Program Files/Java/jdk1.8.0_91/bin/java.exe"
os.environ['JAVAHOME'] = java_path

featureset = []
global count


home = 'C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech'
_path_to_model = home + '/stanford-postagger/models/english-bidirectional-distsim.tagger' 
_path_to_jar = home + '/stanford-postagger/stanford-postagger.jar'
st = POS_Tag(model_filename=_path_to_model, path_to_jar=_path_to_jar)
 

   
max_length = pickle.load(open("max_length.p","rb"))
wh_words = ['how', 'what', 'who', 'when', 'whether', 'why', 'which', 'where']
noun_tags = ['NN', 'NNP']
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
#pronoun_tags = ['PRP', 'PRP$']
adverb_tags = ['RB', 'RBR', 'RBS']


n_classes = 2
batch_size = 1
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500


x = tf.placeholder('float')
y = tf.placeholder('float')



hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.truncated_normal([max_length, n_nodes_hl1])),
                  'bias':tf.Variable(tf.truncated_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.truncated_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.truncated_normal([n_nodes_hl3]))}
                                                         
#hidden_4_layer = {'f_fum':n_nodes_hl4,
##                  'weight':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_nodes_hl4])),
##                  'bias':tf.Variable(tf.truncated_normal([n_nodes_hl4]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.truncated_normal([n_classes])),}



l1 = tf.add(tf.matmul(x,hidden_1_layer['weight']), hidden_1_layer['bias'])
l1 = tf.nn.relu(l1)
l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    #l3 = tf.nn.relu(l3)
l3 = tf.nn.relu(l3)

output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']
                   

#features=np.zeros((1,max_length),dtype='int32')
