#!/usr/bin/python
# -*- coding: utf-8 -*-

import face_recognition_config as config
import sys
import time
import greetings
import random

naoqi_root = config.naoqi_root
sys.path.insert(0, config.naoqi_root)
# naoqi will be found at runtime if installed in config.naoqi_root
from naoqi import ALProxy


class Greeter:
    """
    This class implements the greeting processes. Trained persons are kept in a dictionary as well as timestamps of the
    last greeting and total number of greetings for the given person. If a person is recognized the first time, it is
    greeted after a short delay. It is repeatedly greeted after longer delays.
    """
    robotIp = config.ip
    port = 9559
    greeting_dict = None
    speech_proxy = None
    initial_time_gap = 1
    repeated_time_gap = 5

    def __init__(self):
        # greeting dict holds name as key, time stamp of last greeting, and counter for greetings in total
        # (each for a given person)
        self.greeting_dict = {}
        try:
            self.speech_proxy = ALProxy('ALTextToSpeech', self.robotIp, self.port)
            self.speech_proxy.say('I am ready!')
        except Exception, e:
            'Could not create proxy: ' + str(e)

    def check_greeting(self, name):
        # Manages the greeting dict and time delays
        if name not in [None, 'Unknown']:
            if name not in self.greeting_dict.keys():
                self.greeting_dict[name] = [time.time(), 0]
            local_time_gap = self.repeated_time_gap
            if self.greeting_dict[name][1] == 0:
                local_time_gap = self.initial_time_gap
            if self.greeting_dict[name][0] < (time.time() - local_time_gap):
                self.greet(str(name), self.greeting_dict[name][1])
                self.greeting_dict[name][0] = time.time()
                self.greeting_dict[name][1] += 1

    def greet(self, name, count):
        # Generates the greetings
        sentence = 'Hello'
        if count == 0:
            sentence = str(random.choice(greetings.initial_greetings)) % name
        if count >= 1:
            sentence = str(random.choice(greetings.repeated_greetings)) % name
        print 'INFO: Sentence is: ' + sentence
        self.speech_proxy.say(sentence)
