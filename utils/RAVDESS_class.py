import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.data.utils import Dataset
import os


class RAVDESS(Dataset):
	def __init__(self, audio_dir="../dataset/RAVDESS/audio_files/Audio_Speech_Actors_01-24", type_="train", test_ratio=0.2):
		self.filepath = audio_dir
		self._walker = self.create_walker()
		# Creating test and train set
		test_set = np.random.choice(self._walker, size=test_ratio*len(self._walker), replace=False)
		train_set = [i for i in self._walker if i not in self.test_set]
		if type_=="train":
			self.walker = train_set
		elif type_=="test":
			self.walker = test_set


	def create_walker(self):
		filepath = []
		for folder,_,files in os.walk(self.filepath):
			if _==list():
				for file_ in files:
					filepath.append(os.path.join(folder,file_))
		return filepath

	def load(self, filepath):
		"""
		Filename identifiers :
			Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
			Vocal channel (01 = speech, 02 = song).
			Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
			Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
			Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
			Repetition (01 = 1st repetition, 02 = 2nd repetition).
			Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
		"""
		fname = filepath.split("/")[-1]
		modality, vocal_channel, emotion, emotional_intensity, statement, repitition, actor = fname.split(".")[0].split("-")
		waveform, sample_rate = torchaudio.load(filepath)
		#label = torch.zeros(size=len(list(_emotion_dict)), dtype=torch.int)
		label = list(EMOTION_DICT).index(emotion+emotional_intensity)
		return waveform, sample_rate, label

	def __get_item__(self, num, type="train"):
		if type=="train":
			filepath = self.train_set[num]
		elif type=="test":
			filepath = self.test_set[num]
		return self.load(filepath)




def collate_fn(batch):
	# A data tuple has the form:
	# waveform, sample_rate, label
	transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
	tensors, targets = [], []
	# Gather in lists, and encode labels as indices
	for waveform, sample_rate, label in batch:
		if waveform.shape[1] < 16000:
			waveform = F.pad(input=waveform, pad=(0, 16000 - waveform.shape[1]), mode='constant', value=0)
		tensors += [transform(waveform).squeeze(0).transpose(0,1)]
		targets += [label]

	# Group the list of tensors into a batched tensor
	tensors = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0).unsqueeze(1).transpose(2, 3)
	targets = torch.stack(targets)

	return tensors, targets
