# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:56:21 2017

@author: Ranit
"""
import pyttsx3
engine=pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
rate=engine.getProperty('rate')
engine.setProperty('rate', rate-30)

def syn(statement):
     engine.say(statement)
     engine.runAndWait()
