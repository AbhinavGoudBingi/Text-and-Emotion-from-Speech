import os
import time
import string
import math
import configparser

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

config = configparser.ConfigParser()
config.read("config.ini")

Fs=int(config.get("espnet","fs"))
TAG=config.get("espnet","tag_libri")
LM_TRAIN_CONFIG=config.get("espnet","lm_train_config")
LM_FILE=config.get("espnet","lm_file")
ASR_CONFIG_FILE=config.get("espnet","asr_train_config")
ASR_MODEL_FILE=config.get("espnet","asr_model_file")
DURATION=int(config.get("streamlit","duration"))

class ESPNet_Decoder:

	def __init__(self):
		#	d=ModelDownloader("dataset/") # Run this for the first time
		self.speech2text = Speech2Text(# **d.download_and_unpack(TAG),
								  asr_train_config=ASR_CONFIG_FILE,
								  asr_model_file=ASR_MODEL_FILE,
								  lm_train_config=LM_TRAIN_CONFIG,
								  lm_file=LM_FILE,
								  device="cpu",
								  minlenratio=0.0,
								  maxlenratio=0.0,
								  ctc_weight=0.5,
								  beam_size=4,
								  batch_size=0,
								  nbest=1)
		print("Model is loaded")

	def text_normalizer(self, text):
		text = text.lower()
		return text.translate(str.maketrans('', '', string.punctuation))


	def decode(self, audio_path, sr=Fs):
		print("Model is loaded. The path to the audio file is {}".format(audio_path))
		data, _ = librosa.load(audio_path, sr=sr)
		data = data*10
		start = time.time()
		complete_text=[]
		# number of data_samples in 10 secs = Fs*10
		# total #10sec_chunks = (total_data_points)/(Fs*10)
		num_chunks = math.ceil(len(data)/(Fs*DURATION))
		for i in range(num_chunks):
			nbests = self.speech2text(data[i*Fs*DURATION:(i+1)*Fs*DURATION])
			text, *_ = nbests[0]
			complete_text.append(self.text_normalizer(text))

		decoding_time = time.time()-start
		decoded_text = " ".join(complete_text)
		return decoded_text, decoding_time


if __name__=="__main__":
	audio_path = "audio.wav"
	decoder = ESPNet_Decoder()
	decoded_text, decoding_time, _ = decoder.decode(audio_path)
	print("The text decoded is:")
	print(decoded_text)
	print("Time taken to decode: {}s".format(decoding_time))
