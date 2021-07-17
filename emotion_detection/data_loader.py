import os
import sys
import re
import warnings
import logging
import urllib
import urllib.request
import zipfile
import random


import tqdm
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore")
logging.basicConfig(filename="runtime.log", \
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
					datefmt='%d-%b-%y %H:%M:%S', \
					level=logging.DEBUG, filemode="a")
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RAVDESS:
	_RAVDESS_URL = "https://zenodo.org/record/1188976"
	_emotions = { 'neutral': 0,
				  'calm': 1,
				  'happy': 2,
				  'sad': 3,
				  'angry': 4,
				  'fearful': 5,
				  'disgust': 6,
				  'surprised': 7 }
	_ext_audio = '.wav'
	def __init__(self, root_dir, loader, download=False):
		self.root_dir = root_dir
		self.audio_path = os.path.join(root_dir,
									   "RAVDESS",
									   "audio_files",
									   "Audio_Speech_Actors_01-24")
		if download: # Run only when downloadin is required
			self.download_RAVDESS(self.root_dir)
			logging.debug("Downloading and extraction of the RAVDESS dataset is completed.")
		self.paths = self.data_paths()
		train_paths = random.sample(self.paths, round(0.8*len(self.paths)))
		valid_paths = [i for i in self.paths if i not in train_paths]
		self.audio_paths = self.paths
		if loader=="train":
			self.audio_paths = train_paths
		elif loader=="valid":
			self.audio_paths = valid_paths



	def _RAVDESS_urls(self,url):
		""" Creates a list of urls for fetching the relevant _Audio_ Data
		:param url: the url of the original RAVDESS Dataset website
					"https://zenodo.org/record/1188976"
		:output useful_urls: the url links to the audio zipfiles
		"""
		url_segments = url.split("/")[:-2]
		req = urllib.request.Request(url)
		resp = urllib.request.urlopen(req)
		respData = resp.read()
		links = re.findall(r'href=(.*?)>',str(respData))
		useful_urls = set()
		for link in links:
			if "download=1" in link and "Audio_Speech" in link:
				#print(link)
				useful_urls.add(("/".join(url_segments)+link[1:].split(">")[0]).strip('"'))
		return useful_urls


	def download_zip(self, urls, path="dataset/RAVDESS"):
		""" Uses the RAVDESS_url() function output to download the audio zip files.
			If the path already exists, it assumes the dataset to be present and hence
			does not download the data again.
		:param urls: the list of relevant urls from which the data needs to be fetched
		:param path: the path where the zipfiles needs to be stored
					 (default: "RAVDESS")
		"""
		#pwd = os.getcwd()
		#if os.path.exists(os.path.join(pwd, path)):
		if os.path.exists(path):
			logging.debug("Directory already exists. Seeking for zip files.")
		else:
			#os.mkdir(os.path.join(pwd, path))
			os.mkdir(path)
			#zip_path = os.path.join(pwd,path+"/zip_files")
			zip_path = os.path.join(path,"zip_files")
			os.mkdir(zip_path)
			# print(zip_path)
			for i in tqdm.tqdm(urls, desc="Downloading the Audio zip files: "):
				# takes the name of the zipfile being downloaded to save
				zip_name = i.split("?")[0].split("/")[-1]
				urllib.request.urlretrieve(i, zip_path + "/" + zip_name)
			logging.debug("\nZip Files downloaded")
		return os.path.join(path,"zip_files")


	def unzip_audio_files(self, zip_path, path):
		""" This function is used to unzip all the zipfiles downloaded by the
			download_zip() function and place it to the path provided.
		:param zip_path: the path where the zipfiles are downloaded and kept
		:param path: the path where the audio files are to be downloaded
		"""
		#pwd = os.getcwd()
		#if os.path.exists(os.path.join(pwd, path)):
		if os.path.exists(path):
			logging.debug("Audio files are already extracted.")
		else:
			#os.mkdir(os.path.join(pwd, path))
			os.mkdir(path)
			#for zips in tqdm.tqdm(os.listdir(os.path.join(pwd, zip_path)), desc="Unzipping the Audio files: "):
			for zips in tqdm.tqdm(os.listdir(zip_path), desc="Unzipping the Audio files: "):
				name = zips.split(".")[-2]
				#with zipfile.ZipFile(os.path.join(os.path.join(pwd, zip_path), zips), 'r') as zip_ref:
				with zipfile.ZipFile(os.path.join(zip_path, zips), 'r') as zip_ref:
					#zip_ref.extractall(os.path.join(os.path.join(pwd, path), name))
					zip_ref.extractall(os.path.join(path, name))
			logging.debug("\nFinished unzipping.")


	def download_RAVDESS(root_dir):
		urls = self.RAVDESS_urls(self._RAVDESS_URL)
		zip_path = self.download_zip(urls, path=os.path.join(root_dir,"RAVDESS"))
		self.unzip_audio_files(zip_path=zip_path ,path=os.path.join(root_dir,"RAVDESS","audio_files"))


	def data_paths(self):
		audio_paths = []
		for _,_, files in os.walk(self.audio_path):
			for file in files:
				if file.endswith(self._ext_audio):
					identifiers = file[:-len(self._ext_audio)].split("-")
					path = os.path.join(self.audio_path,"Actor_"+identifiers[-1],file)
					audio_paths.append((path, identifiers[2]))
		return audio_paths


	def __len__(self):
		return len(self.audio_paths)


	def __getitem__(self, idx):
		holder = torch.zeros((1,10*16000))
		waveform, sample_rate = torchaudio.load(self.audio_paths[idx][0])
		waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform,
																			sample_rate,
																		    effects=[['rate', '16000']])
		if waveform.shape[0]>1:
			waveform = waveform.mean(axis=0)
			waveform = waveform.reshape(1, len(waveform))

		# emotion = [0]*len(self._emotions)
		emotion=int(self.audio_paths[idx][1][-1])-1
		holder[:,:waveform.shape[1]] = waveform
		return holder, sample_rate, torch.Tensor(emotion)


def audio_transforms(sample_rate=48000, nfeats=128):
	"""
	Transformations on the raw audio for training and validation sets
	:param nfeats: the number of MelSpectrogram Features that are required
				   (default = 128)
	:output train_audio_transforms: the pipeline to extract training data features
	:output valid_audio_transforms: the pipeline to extract vallidation data features
	For the training dataset, frequency and time-masking are also performed,
	but for the validation dataset, only MelSpectrogram features are extracted.
	Masking in the training dataset, enables robust training.
	"""
	train_audio_transforms = nn.Sequential(
								torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels = nfeats),
								torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
								torchaudio.transforms.TimeMasking(time_mask_param=35))
	valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=nfeats)

	return train_audio_transforms, valid_audio_transforms

# max_dim = 422
def data_processing(data, data_type="train"):
	spectrograms = []
	labels = []
	global max_dim

	train_audio_transforms, valid_audio_transforms = audio_transforms()

	for waveform, sample_rate, emotion in data:
		if data_type == 'train':
			spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
		else:
			spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)

		spectrograms.append(spec)
		labels.append(emotion)

	spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
	labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

	return spectrograms, labels


def get_dataset():
	train_dataset = RAVDESS(root_dir="../dataset/", loader="train",) #download=True)
	validation_dataset = RAVDESS(root_dir="../dataset/", loader="valid") #download=True)
	return train_dataset, validation_dataset


def create_data_dataloader(batch_size=32, nfeats=128):
	if DEVICE == "cuda":
		num_workers = 4
		pin_memory = True
	else:
		num_workers = 0
		pin_memory = False

	train_dataset, validation_dataset = get_dataset()
	train_audio_transforms, valid_audio_transforms = audio_transforms(nfeats=nfeats)

	train_loader = DataLoader(dataset=train_dataset,
							  batch_size=batch_size,
							  shuffle=True,
							  collate_fn=lambda x: data_processing(x, 'train'),
							  num_workers=num_workers,
							  pin_memory=pin_memory)

	test_loader = DataLoader(dataset=validation_dataset,
							 batch_size=batch_size,
							 shuffle=False,
							 drop_last=False,
							 collate_fn=lambda x: data_processing(x, 'valid'),
							 num_workers=num_workers,
							 pin_memory=pin_memory)
	return train_loader, test_loader
