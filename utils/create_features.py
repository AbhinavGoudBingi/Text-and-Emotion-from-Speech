import librosa
import numpy as np
import os
import sys
import tqdm
import wavio
import logging

logging.basicConfig(filename="runtime.log", \
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
					datefmt='%d-%b-%y %H:%M:%S', \
					level=logging.DEBUG, filemode="a")

class AudioFeatures:
	"""	Build the features from audio files to be able to retrieve
		from audio path and some predefined features.
		Uses the methods of librosa for feature extractions.

		Options:
			- get_spectogram: can be used to extract spectogram of the audio file

			- get_mfccs: can be sed to get mfcc features of the audio file
			- get zcr: can be used to get zero-crossing rates of the audio file
	"""

	_modality_dict = {"01": "full-AV", "02": "video-only", "03": "audio-only"}
    _vocal_channel_dict = {"01": "speech", "02": "song"}
    _emotion_dict = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
    _emotional_intensity_dict = {"01": "normal", "02": "strong"}
    _statement_dict = {"01": "Kids are talking by the door", "02": "Dogs are sitting by the door"}
    _repetition_dict =  {"01": "1st repetition", "02": "2nd repetition"}
    _gender_dict = {0: "female", 1: "male"}
    _n_mels = 128
    _alpha = 0.95

	def __init__(self, filepath):
		self.filepath = filepath
		self.data, self.sr = wavio.read(self.filepath)
		self.spectogram = None
		self.mfccs = None
		self.zcr = None


	def get_spectogram(self, audio):
		raise NotImplementedError

	def get_mfccs(self, audio):
		raise NotImplementedError

	def get_zcr(self, audio):
		raise NotImplementedError

	def __repr__(self):
		return "AudioFeatures Object for file:{}".format(self.filepath)
