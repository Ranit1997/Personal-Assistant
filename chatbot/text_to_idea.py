# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 18:17:38 2017

@author: Punyajoy Saha
"""

#from rasa_nlu.converters import load_data
#from rasa_nlu.config import RasaNLUConfig
#from rasa_nlu.model import Trainer
#
#training_data = load_data('data/examples/rasa/demo-rasa.json')
#trainer = Trainer(RasaNLUConfig("config_spacy.json"))
#trainer.train(training_data)
##model_directory = trainer.persist('./models/')  # Returns the directory the model is stored in

#from chatterbot import ChatBot
#
#chatbot = ChatBot(
#    'Ron Obvious',
#    trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
#)
#
## Train based on the english corpus
#chatbot.train("chatterbot.corpus.english")
#
## Get a response to an input statement
#chatbot.get_response("Hello, how are you today?")
#
#import cleverbot

api_keys = []
api_keys.append(('punyajoy','CC3garsYVrVMwyNVDInlIfuZIFQ'))
api_keys.append(('sandipan','CC3gtSPsiSQgJAuUHt2tm-73DpQ'))
api_keys.append(('ranit','CC3gstaeSBhm1KzibB3Ds6rDRbQ'))
api_keys.append(('suprotik','CC3pjZAEPh1e_sSO_0V3C-qw6sw'))
api_keys.append(('vivek','CC3pmZTMU4SqauwbfUqaWYTkMlg'))


#cb1 = cleverbot.Cleverbot('CC3garsYVrVMwyNVDInlIfuZIFQ')
#print(cb1.ask('Hi. How are you?'))

#
#from cleverwrap import CleverWrap
#cw = CleverWrap("CC3garsYVrVMwyNVDInlIfuZIFQ")

#w=cw.say("who are you.")
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

bot = ChatBot('Norman')
#bot = ChatBot(
#    'Norman',
#    storage_adapter='chatterbot.storage.SQLStorageAdapter',
#    database='./database.sqlite3'
#)
#
#bot = ChatBot(
#    'Norman',
#    storage_adapter='chatterbot.storage.SQLStorageAdapter',
#    input_adapter='chatterbot.input.TerminalAdapter',
#    output_adapter='chatterbot.output.TerminalAdapter',
#    logic_adapters=[
#        'chatterbot.logic.MathematicalEvaluation',
#        'chatterbot.logic.TimeLogicAdapter'
#    ],
#    database='./database.sqlite3'
#)
bot.set_trainer(ChatterBotCorpusTrainer)
#bot.train([
#    'How are you?',
#    'I am good.',
#    'That is good to hear.',
#    'Thank you',
#    'You are welcome.',
#])
bot.train("chatterbot.corpus.english.conversations")
print('trained')
print('Say something')
bot.set_trainer(ListTrainer)

#while True:
#    question_asked=input()
#    bot_input = bot.get_response(question_asked)
#    print(bot_input)
#    is_correct=input('is my answer correct?')
#    if is_correct=='no':
#        correct_answer=input('give correct response')
#        bot.train([question_asked,correct_answer])   
#    take_input=input('do you want to continue?')
#    if take_input=='no':
#        break
    
def botSpeaks(statement):
  
        bot_input = bot.get_response(statement)
        print(bot_input)
       
        return bot_input


