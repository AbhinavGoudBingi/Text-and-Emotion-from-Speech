#!/usr/bin/env python3

import wave
import os
import base64
import time
import configparser

import pyaudio
import keras
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from espnet_decoder import ESPNet_Decoder
from emotion_decoder import detect_emotion

#plt.rc("text", usetex=True)
plt.style.use("bmh")

config = configparser.ConfigParser()
config.read("config.ini")
# CONSTANTS FOR AUDIO RECORDING
DURATION=int(config.get("streamlit","duration")) # record for 10 seconds
MAX_INPUT_CHANNELS=int(config.get("streamlit","max_input_channels")) # mono audio recording
DEFAULT_SAMPLE_RATE=int(config.get("streamlit","sample_rate"))# 44.1KHz sampling rate default for mic
CHUNKSIZE=int(config.get("streamlit","chunksize"))
WAVE_OUTPUT_FILE=config.get("streamlit","wave_output_file")
TEMP_OUTPUT_FILE=config.get("streamlit","temp_out_file")
INPUT_DEVICE=int(config.get("streamlit","input_device"))


class record_audio:
	def __init__(self):
		self.recorder = pyaudio.PyAudio()


	def save_recording(self,frames):
		if not frames:
			raise ValueError("No frames recorded.")
		wavfile = wave.open(WAVE_OUTPUT_FILE,"wb")
		wavfile.setnchannels(MAX_INPUT_CHANNELS)
		wavfile.setsampwidth(self.recorder.get_sample_size(pyaudio.paInt16))
		wavfile.setframerate(DEFAULT_SAMPLE_RATE)
		wavfile.writeframes(b''.join(frames))
		wavfile.close()

		#Using ffmpeg to convert the sampling rate to 16k
		os.system(f"ffmpeg -i {WAVE_OUTPUT_FILE} -ac 1 -ar 16000 -y {TEMP_OUTPUT_FILE}")
		os.system(f"mv {TEMP_OUTPUT_FILE} {WAVE_OUTPUT_FILE}")


	def record(self):
#		recorder = pyaudio.PyAudio()
		stream = self.recorder.open(format=pyaudio.paInt16,
							   channels=MAX_INPUT_CHANNELS,
							   rate=DEFAULT_SAMPLE_RATE,
							   input=True,
							   frames_per_buffer=CHUNKSIZE,)
							   #input_device_index=INPUT_DEVICE)
		frames=[]
		for i in range(0, int(DEFAULT_SAMPLE_RATE / CHUNKSIZE * DURATION)):
		#while not self.stop_audio:
			data = stream.read(CHUNKSIZE)
			frames.append(data)

		stream.stop_stream()
		stream.close()
		self.recorder.terminate()
		self.save_recording(frames)


	def display_audio(self, audio_path=WAVE_OUTPUT_FILE, sr=16000):
		data, _ = librosa.load(audio_path, sr=sr)
		fig=plt.figure(figsize=(8,3))
		librosa.display.waveplot(data, sr=sr, alpha=0.5)
		plt.grid(True)
		plt.title("Recorded Audio Waveform")
		plt.xlabel("Time")
		plt.ylabel("Amplitude")
		return fig


def get_device_info():
	recorder = pyaudio.PyAudio()
	num_devices = recorder.get_device_count()
	keys = ['name', 'index', 'maxInputChannels', 'defaultSampleRate']
	out_text=[]
	for n in range(num_devices):
		info_dict = recorder.get_device_info_by_index(n)
		values = [value for _,value in info_dict.items() if _ in keys]
		out = "\n".join([" : ".join([key,str(val)]) for key, val in zip(keys, values)])
		out_text.append(out)
	return "\n\n".join(out_text)

@st.cache(allow_output_mutation=True)
def get_espnet_decoder():
	return ESPNet_Decoder()

@st.cache(allow_output_mutation=True)
def get_emotion_decoder():
	return keras.models.load_model('dataset/Emotion_Voice_Detection_Model.h5')

def main():
	decoder = get_espnet_decoder()
	emo_decoder = get_emotion_decoder()
#	st.set_page_config(layout="centered",
#						page_icon=":smiley:",
#						page_title="CS 753 Project")
	title="Speech to Sign-Language."
	st.title(title)
	header="ASR speech to text demo."
	st.header(header)


	st.subheader("Display the working audio ports in your device to record audio.")
	device_text = ""
	if st.button("Get Device Audio Ports Info"):
		device_text=get_device_info()
		st.text(device_text)

	audio_recorder = record_audio()
	st.subheader("Record audio for prediction.")
	if st.button("Record"):
		st.write("When the prompt to Speak comes, speak for 10secs.")
		time.sleep(5)
		st.write("You can Speak Now.")
		with st.spinner(f"Recording."):
			audio_recorder.record()
			st.success("Recording done. Let's hear the recording.")
#	if st.button("Stop"):
#			audio.recorder.stop()
	# if text=="Sucess":

	st.subheader("Play the recorded audio.")
	if st.button("Play"):
		try:
			st.pyplot(audio_recorder.display_audio(),
					  width=5,
					  height=5)
			audio_file = open(WAVE_OUTPUT_FILE,"rb")
			audio_bytes = audio_file.read()
			st.audio(audio_bytes, format="audio/wav")
			audio_file.close()
		except:
			st.write("Please record sound first")

	st.subheader("Predictions")
	if st.button("Convert Speech to Text and detect Emotion"):
		decoded_text, decoding_time = decoder.decode(WAVE_OUTPUT_FILE)

		col1, col2, col3 = st.beta_columns(3)

		with col1:
			st.header("Predicted Text")
			st.subheader("The decoded text of the audio clip:")
			st.write(decoded_text)
			st.write(f"Decoding time taken: {decoding_time}s")

		with col2:
			st.header("Predicted Emotion")
			st.subheader("The decoded emotion from the audio clip is:\n")
			out = detect_emotion(emo_decoder, WAVE_OUTPUT_FILE)
			st.write(out)

		col3.header("Sign Language Video")
		col3.write("Under Construction")


if __name__=="__main__":
	main()
