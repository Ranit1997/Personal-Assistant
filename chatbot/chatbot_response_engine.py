#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:36:16 2017

@author: Ranit
"""


import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
# restore all of our data structures
import pickle
from nltk.stem.lancaster import LancasterStemmer
chatbot_stemmer = LancasterStemmer()

chatbot_data = pickle.load( open( "C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech/chatbot/training_data.p", "rb" ) )
chatbot_words = chatbot_data['words']
chatbot_classes =chatbot_data['classes']
chatbot_train_x = chatbot_data['train_x']
chatbot_train_y =chatbot_data['train_y']

# import our chat-bot intents file
import json
with open('C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech/chatbot/hardcoded_convo.json') as json_data:
    chatbot_intents = json.load(json_data)

# load our saved model
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(chatbot_train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(chatbot_train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
chatbot_new_model = tflearn.DNN(net, tensorboard_verbose=3)

chatbot_new_model.load('C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech/chatbot/model.tflearn')



def chatbot_clean_up_sentence(sentence):
    # tokenize the pattern
    chatbot_sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    chatbot_sentence_words = [chatbot_stemmer.stem(word.lower()) for word in chatbot_sentence_words]
    return chatbot_sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def chatbot_bow(sentence, words, show_details=False):
    # tokenize the pattern
    chatbot_sentence_words = chatbot_clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in chatbot_sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def chatbot_classify(sentence):
    # generate probabilities from the model
    results = chatbot_new_model.predict([chatbot_bow(sentence, chatbot_words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    global cleverbot
    for r in results:
        return_list.append((chatbot_classes[r[0]], r[1]))
       
       
        if(r[1]>0.8 and cleverbot==True):
            cleverbot=False
            print(r[1],cleverbot)
            
    # return tuple of intent and probability
    return return_list

def chatbot_response(sentence, userID='123', show_details=False):
    results = chatbot_classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in chatbot_intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return(random.choice(i['responses']))

            results.pop(0)

            if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return(random.choice(i['responses']))

                     
def chatbotSpeaks(statement):
   
      global cleverbot
      cleverbot=True
      category= [x[0] for x in chatbot_classify(statement)]
      print(category[0])
      if(cleverbot==False):
          reply=chatbot_response(statement)
      else:
          reply="pass to cleverbot"
      print("Chatbot Classifier:-", reply)
      return reply
      #print(cleverbot)
      

#chatbotSpeaks("who created you")    
    

    


