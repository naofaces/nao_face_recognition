#!/usr/bin/env python

import sys
import naoConfig
naoqi_root = naoConfig.naoqi_root
sys.path.insert(0, naoqi_root)
import naoqi
from naoqi import ALProxy


# sentence = "This was a triumph. I'm making a note here: Huge Success. It's hard to overstate my satisfaction. Aperture Science. We do what we must, because we can, For the good of all of us, except the ones who are dead."
sentence = "gday, mate"

tts = ALProxy('ALTextToSpeech', naoConfig.Ip, 9559)
tts.say(sentence)

