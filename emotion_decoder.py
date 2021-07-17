import os
import sys
import configparser

import keras
import numpy as np
import librosa

EMOTIONS=['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
config = configparser.ConfigParser()
config.read("config.ini")
Fs=int(config.get("espnet","fs"))


def detect_emotion(model, filename):
	data, sr = librosa.load(filename, sr=Fs, res_type='kaiser_fast')
	data = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T,axis=0)
	data = data.reshape(1,*data.shape,1)
	out = np.argmax(model.predict(data))
	return EMOTIONS[out-1]
