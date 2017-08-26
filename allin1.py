# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 16:53:01 2017

@author: SANDIPAN and Ranit
"""
'''t_path = ['', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\python35.zip', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\DLLs', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\lib', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\lib\\site-packages', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\lib\\site-packages\\setuptools-27.2.0-py3.5.egg']
'''
import os
import requests
import sys
re_path=['','C:\\Users\\user\\Desktop\\LABs\\IIEST HUMANOID PROGRAMME\\Open_HMD_1.0\\OpenHmnD-master\\speech\\responseEngine']
sys.path = re_path+sys.path

exec(open("C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech/responseEngine/response_engine.py").read())
exec(open("C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech/chatbot/chatbot_response_engine.py").read())
exec(open("C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech/runner.py").read())

import pickle
import numpy as np
max_length = pickle.load(open("max_length.p","rb"))
features=np.zeros((1,max_length),dtype='int32')
api_keys = pickle.load(open("api_keys.p","rb"))


import tensorflow as tf
import speechTest
from speechTest.speech_test import *
import responseEngine
from responseEngine import *
import chatbot
from chatbot import *
from cleverwrap import CleverWrap
import speech_recognition as sr
import Speak
from Speak import say
#import sentiment.predict_sentiment as sen


def cleverSpeaks(stmt):
          
    i =0
    try:
        cw = CleverWrap(api_keys[i][1])
        spoken = cw.say(stmt)
    except requests.exceptions as e:
        print (e)
    
   
    return spoken


def all_ears():
    r = sr.Recognizer()
    print("listening............")
    with sr.Microphone() as src: 
       
        audio = r.listen(src)
    msg = ''
    try:
        msg = r.recognize_google(audio) 
        #print("message-",msg.lower())
    except sr.UnknownValueError:
        msg='gibberish'
    except sr.RequestError as e:
        print("Could not request results from Google STT; {0}".format(e))
    except:
        msg='gibberish'
            
    finally:
        return msg.lower()  




def start():
        speech_return=[]
        with tf.Session() as sess:
             
             saver = tf.train.Saver()
             
             saver.restore(sess, "C:/Users/user/Desktop/LABs/IIEST HUMANOID PROGRAMME/Open_HMD_1.0/OpenHmnD-master/speech/saved_models/11/model.ckpt")
             
             stmt = input('Your statement: ')
             global idle,idleflag
             idle=True
             trigger=True
             
             #stmt=all_ears()
             
                 
                 
             if(trigger==True):
                 idle=False
             elif(stmt=="hello darsh" and trigger==False):
                 idle=False
                 trigger=True
                
             if(idle==False):    
                 if(stmt=='gibberish'):
                     i=0
                     while(i<=2 and stmt =='gibberish'):
                          i=i+1
                          print(i)
                          stmt=all_ears()
                          
                          if(i==1):
                              say.syn('Is anyone around')
                         
                     if(i==4):
                        
                         idle=True
                         trigger=False
                         print("idle:-",idle,"trigger",trigger)
                         
                 print(stmt)
                 
                 if(stmt=="what do you see"):
                     cnc=None
                     e=None
                     t=None
                     pa=None
                     sen=None
                     wdys=True
                     speech_return=[cnc,e,t,pa,sen,wdys]
                 else:
                     
                 
                     wdys=False
                     features_1 = make_featuresets(stmt)
                     features[0,:]=np.array(features_1)
                     prediction=sess.run(output,feed_dict={x:features})
                     s=np.argmax(prediction)
                     if s==0:
                         cnc='C'
                         command_list=[]
                         #print('command')
                         commclass=botActs(stmt)
                         command_list=command_analyse(stmt)
                         if(command_list!=None):
                             
                             if(commclass=='E'):
                                 e=command_list[0]
                                 t=None
                                 pa=None
                             elif(commclass=='T'):
                                e=None
                                t=command_list[0]
                                pa=None
                             elif(commclass=='PA'):
                                e=None
                                t=None
                                if(command_list[0]=='impossible'):
                                    say.syn("Sorry,I am unable to do that")
                                    pa=None
                                else:
                                    pa=command_list[0]
                            
                         sen=None
                         speech_return=[cnc,e,t,pa,sen,wdys]
                         
                            
                            
                             
                         #print(reply)
                         
                     else:
                         cnc='NC'
                         e=t=pa=None
                         sen=None
                         speech_return=[cnc,e,t,pa,sen,wdys]
                         #print('non_command')
                         reply=chatbotSpeaks(stmt)
                        
                         
                         if(reply=="pass to cleverbot"):
                             print("call cleverbot API")
                             creply = cleverSpeaks(stmt)
                             print(creply)
                             say.syn(creply) 
                             #sent=sen.find_sentiment(creply)
                         else:
                             #sent=sen.find_sentiment(reply)
                             say.syn(reply)
                   
        print(speech_return)

start()
                    