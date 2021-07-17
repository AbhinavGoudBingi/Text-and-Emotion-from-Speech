# System modules required for creating the dataset
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# PyTorch modules required to create dataset
import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import LIBRISPEECH

# To convert from text to embeddings
import transform_text

DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_transform = transform_text.TextTransform()


def download_dataset(folder="./data"):
	"""
	Download LIBRISPEECH train-clean-100 and test-clean in the `data` folder
	:param folder: the folder where the data is to be downloaded
	:output train_dataset: the training dataset tensor
	:output test_dataset: the validation dataset tensor
	"""
	if not os.path.isdir(folder):
		os.makedirs(folder)
	train_dataset = LIBRISPEECH(folder, url="train-clean-100",) #download=True)
	validation_dataset = LIBRISPEECH(folder, url="test-clean",) #download=True)
	return train_dataset, validation_dataset


def audio_transforms(nfeats=128):
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
									torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels = nfeats),
									torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
									torchaudio.transforms.TimeMasking(time_mask_param=35))
	valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=nfeats)

	return train_audio_transforms, valid_audio_transforms


def data_processing(data, data_type="train"):
	spectrograms = []
	labels = []
	input_lengths = []
	label_lengths = []

	train_audio_transforms, valid_audio_transforms = audio_transforms()

	for (waveform, _, utterance, _, _, _) in data:
		if data_type == 'train':
			spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
		else:
			spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
		spectrograms.append(spec)
		label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
		labels.append(label)
		input_lengths.append(spec.shape[0]//2)
		label_lengths.append(len(label))

	spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
	labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

	return spectrograms, labels, input_lengths, label_lengths


def create_data_dataloader(batch_size=32, nfeats=128):
	if DEVICE == "cuda":
		num_workers = 4
		pin_memory = True
	else:
		num_workers = 0
		pin_memory = False

	train_dataset, validation_dataset = download_dataset()
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
