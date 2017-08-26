# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 23:35:50 2017

@author: Punyajoy Saha and Ranit
"""

from PyDictionary import PyDictionary
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
dictionary=PyDictionary()
lemmatizer = WordNetLemmatizer()

cry=['cry','weep','sad','gloomy']
smile=['grin','smile','laugh','happy']
anger=['rage','furious','anger']
suprise=['astonish','suprise']
track=['find','follow','track','search','see','watch','detect']
other_physical=['get','stand','sit','squat','move','dance','move','forward','backward']
turn=['turn','left','right','rotate']
remember=['remember']


def makenoun_list(line,word):
    noun_list=[]
    pos_tags=nltk.pos_tag(line)
    for i in pos_tags:
        if i[1]=='NN':
            noun_list.append(i[0])
        elif i[1]=='PRP' and i[0] =='me':
            noun_list.append('person')
    return noun_list  

'''def make_noun_adjective_list(line,word):
    noun_adjective_list=[]
    pos_tags=nltk.pos_tag(line)
    for i in pos_tags:
        if i[1]=='NN':
            noun_adjective_list.append(i[0])
        elif i[1]=='JJ' or i[1]=='JJR' or i[1]=='JJS':
            noun_adjective_list.append(i[0])
    return noun_adjective_list'''  

    
def makeaction_list(line,word):
    for i in line:
        if i=='right':
            action=['right']
        elif i=='left':
            action=['left']
    return action     
       

def command_analyse(command):
    emo=[]
    all_words = word_tokenize(command)
    glossary = [lemmatizer.lemmatize(i) for i in all_words]
    pos_tags=nltk.pos_tag(glossary)
    #print(pos_tags)
    for i in glossary:
        if i in cry:
            #print("sad")
            emo=['sad']
            #nouns_adjectives=make_noun_adjective_list(glossary,i)
            return(emo)
            break
        elif i in smile:
            #print("happy")
            emo=['happy']
            #nouns_adjectives=make_noun_adjective_list(glossary,i)
            return(emo)
            break
        elif i in anger:
            #print("Anger")
            emo=['angry']
            #nouns_adjectives=make_noun_adjective_list(glossary,i)
            return(emo)
            break
        elif  i in suprise:
            #print("Surprise")
            emo=['surprise']
            #nouns_adjectives=make_noun_adjective_list(glossary,i)
            return(emo)
            break
        elif  i in track:
            #print("Track")
            nouns=[]
            nouns=makenoun_list(glossary,i)
            return(nouns)
            break
        
        elif i in turn:
            action=makeaction_list(glossary,i)
            #print('turning')
            return action
        elif i in other_physical: 
            #print('Impossible Physical activity')
            action=['impossible']
            return action
            
        
