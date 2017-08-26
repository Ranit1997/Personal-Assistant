# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:27:20 2017

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
java_path = "C:/Program Files/Java/jdk1.8.0_91/bin/java.exe"
os.environ['JAVAHOME'] = java_path

max_length_cmd = []
featureset = []
global count


home = 'C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech'
_path_to_model = home + '/stanford-postagger/models/english-bidirectional-distsim.tagger' 
_path_to_jar = home + '/stanford-postagger/stanford-postagger.jar'
st = POS_Tag(model_filename=_path_to_model, path_to_jar=_path_to_jar)


cmd = open('command.txt','r')
non_cmd = open('non_command.txt','r')

for line in cmd:
    max_length_cmd.append(len(line.split()))
    
for line in non_cmd:
    max_length_cmd.append(len(line.split()))
    
max_length = max(max_length_cmd)


wh_words = ['how', 'what', 'who', 'when', 'whether', 'why', 'which', 'where']
noun_tags = ['NN', 'NNP', 'PRP', 'PRP$']
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adverb_tags = ['RB', 'RBR', 'RBS']

def create_features(features, words):
    for i in range(0,len(words)):
        
        if  words[i][0] in wh_words:
            features[i] = 10
        elif words[i][1] in noun_tags:
            features[i] = 20
        elif words[i][1] in verb_tags:
            features[i] = 30  
        elif words[i][1] in adverb_tags:
            features[i] = 40    
        elif words[i][1] not in wh_words and words[i][1] not in noun_tags and words[i][1] not in verb_tags and words[i][1] not in adverb_tags:
            features[i] = 50
        
    return features
            
def make_featuresets(file, classification):
    
  with open(file,'r') as f:
    contents = f.readlines()
    count = 0
    for line in contents:
        current_words = word_tokenize(line)
        count = count+1
        print(count)
        tagged_words = st.tag(current_words)
        features = [0]*(max_length)
        features_in_line = create_features(features, tagged_words)
        featureset.append([features_in_line, classification])
        
  return featureset
    
    
def featuresets_and_labels(command, non_command, test_size = 0.3):
    
    all_features = []
    train_x = []
    train_y = []
    test_x= []
    test_y = []
    all_features+= make_featuresets (command, [1,0])
    all_features+= make_featuresets (non_command, [0,1])        #changed
    #print(all_features)
#    all_features = pickle.load(open("feat.p","rb"))
    random.shuffle(all_features)
    all_features = np.array(all_features)
    testing_size = int(test_size*len(all_features))
   
    #testing_size = int(test_size*len(all_features))
    #training_size = int(len(all_features) - testing_size)
    
    #for i in range(0,training_size):
     #   train_x.append(all_features[i][0:1])
      #  train_y.append(all_features[i][1:2])
        
    #for i in range(training_size+1, len(all_features)):
    #    test_x.append(all_features[i][0:1])
    #    test_y.append(all_features[i][1:2])


    train_x = list(all_features[:,0][:-testing_size])
    train_y = list(all_features[:,1][:-testing_size])
    test_x = list(all_features[:,0][-testing_size:])
    test_y = list(all_features[:,1][-testing_size:])
   
    
    print(len(all_features))
    #print(features)
    
    return train_x, test_x, train_y, test_y


    
    


train_x, test_x, train_y, test_y = featuresets_and_labels('command.txt', 'non_command.txt')



n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
#n_nodes_hl4 = 100


n_classes = 2
batch_size = 1
#hm_epochs = 20

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1])),
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

def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    #l3 = tf.nn.relu(l3)
    l3 = tf.nn.relu(l3)
    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']
    
    return output

    
    
def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
#    # NEW:
    ratio =171.0 / (248.0 + 171.0) 
    class_weight = tf.constant([ratio, 1.0 - ratio]) 
    logits = prediction
    keep_prob = tf.placeholder(tf.float32)
    weighted_logits = tf.multiply(logits, class_weight) 
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=weighted_logits, labels=y) )
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    
    hm_epochs = 150
#    config=tf.ConfigProto()
#    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(hm_epochs):

        	epoch_loss = 0

        	i = 0
            
        	while i<len(train_x):
        		start = i
        		end=i+batch_size
        		batch_x=np.array(train_x[start:end])
        		batch_y=np.array(train_y[start:end])
        		_, c=sess.run([optimizer, cost], feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
        		epoch_loss+=c
        		i+=batch_size

        	print('Epoch', epoch+1, 'Completed out of', hm_epochs, 'loss:', epoch_loss)
	        correct=tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	        accuracy=tf.reduce_mean(tf.cast(correct, 'float'))

	        print('Accuracy:',accuracy.eval({x:train_x, y:train_y,keep_prob:1.0}))



	        correct=tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	        accuracy=tf.reduce_mean(tf.cast(correct, 'float'))

	        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
        save_path = saver.save(sess, "C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech/saved_models/classifier/model.ckpt")
        print("Model saved in file: %s" % save_path)
         
train_neural_network(x)
