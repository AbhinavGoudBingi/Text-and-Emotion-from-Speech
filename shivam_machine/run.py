#!/usr/bin/env python3

import os
import models
import decoder
from train_test import train, validate
import create_dataloader
import tqdm
import argparse
import logging
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler

# set random seeds
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark=False

logging.basicConfig(filename="runtime.log", \
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
					datefmt='%d-%b-%y %H:%M:%S', \
					level=logging.DEBUG, filemode="a")
MODEL_PATH="deep_speech_state_dict.pth"


if __name__=="__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument("-z","--hidden-size",type=int, help="Number of Units in LSTM layer.", default=128)
	parser.add_argument("-b", "--batch-size", type=int, help="Batch Size.", default=32)
	parser.add_argument("-n", "--num-epochs", type=int, help="Number of iterations/epochs.", default=30)
	parser.add_argument("-lr", "--learning-rate", type=float, help="Learning rate for optimizer.", default=0.001)
	parser.add_argument("-cnn", "--num-cnn-layers", type=int, help="Number of CNN layers for DeepSpeech Model", default=2)
	parser.add_argument("-rnn", "--num-rnn-layers", type=int, help="Number of Bi-directional RNN layers for DeepSpeech Model", default=2)
	parser.add_argument("-nf", "--nfeats", type=int, help="Number of Mel-spectogram features", default=128)
	parser.add_argument("-s", "--stride", type=int, help="Stride Length for CNN layers for DeepSpeech Model", default=2)
	parser.add_argument("-d", "--dropout", type=float, help="Dropout for DeepSpeech Model", default=0.2)

	args = parser.parse_args()

	logging.debug("\nModel Parameters:\n")
	logging.debug('--------------------------------------\n')
	logging.debug('Number of CNN Layers         :{}'.format(args.num_cnn_layers))
	logging.debug('Stride Length                :{}'.format(args.stride))
	logging.debug('Dropout for CNN              :{}'.format(args.dropout))
	logging.debug('Number of Bi-GRU Layers      :{}'.format(args.num_rnn_layers))
	logging.debug('Hidden Size                  :{}'.format(args.hidden_size))
	logging.debug('Batch Size                   :{}'.format(args.batch_size))
	logging.debug('Learning rate                :{}'.format(args.learning_rate))
	logging.debug('Number of Epochs             :{}'.format(args.num_epochs))
	logging.debug('--------------------------------------\n\n')


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, test_loader = create_dataloader.create_data_dataloader(batch_size=32, nfeats=args.nfeats)
	labels = list("' abcdefghijklmnopqrstuvwxyz_")
	n_class=len(labels)
	model = models.SpeechRecognitionModel(n_cnn_layers = args.num_cnn_layers, \
										  n_rnn_layers = args.num_rnn_layers, \
										  rnn_dim = args.hidden_size, \
										  n_class = n_class, \
										  n_feats = args.nfeats, \
										  stride = args.stride, \
										  dropout = args.dropout).to(device)

	logging.debug("SpeechRecognitionModel loaded.")
	print("Model Loaded successfully.")
	criterion = nn.CTCLoss(blank=labels.index("_"), zero_infinity=True).to(device)
	optimizer = optim.Adam(model.parameters(),
						   lr = args.learning_rate)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
													'max',
													patience=5,
													factor=0.1,
													verbose=True,
													threshold=0.0001,
													threshold_mode='abs',
													cooldown=1)
	decoder = decoder.BeamCTCDecoder(labels=labels)
	logging.debug("\nCriterion: CTCLoss\nOptimizer: Adam\nScheduler: ReduceLROnPlateau with patience=5")
	log = OrderedDict([
						('epoch', []),
						('lr', []),
						('train_loss', []),
						('train_cer', []),
						('train_wer', []),
						('val_loss', []),
						('val_cer', []),
						('val_wer', []),
						])
	start_epoch=1
	# Load from the .pth file things that are saved
	if os.path.isfile(MODEL_PATH):
		checkpoint=torch.load(MODEL_PATH)
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		log["epoch"] = checkpoint['epoch']
		log['lr'] = checkpoint["lr"]
		log['train_loss'] = checkpoint['train_loss']
		log['train_cer']=checkpoint['train_cer']
		log['train_wer']=checkpoint['train_wer']
		log['val_loss']=checkpoint['val_loss']
		log['val_cer']=checkpoint['val_cer']
		log['val_wer']=checkpoint['val_wer']
		start_epoch = log["epoch"][-1]
		logging.debug("Using the pretrained model at: {}".format(MODEL_PATH))

	model = model.to(device)
	for epoch in tqdm.tqdm(range(start_epoch, args.num_epochs+1)):
		# cudnn.benchmark = True
		train_log = train(model, train_loader, epoch, criterion, optimizer, decoder, scheduler)
		val_log = validate(model, test_loader, epoch, criterion, optimizer, decoder)

		log['epoch'].append(epoch)
		log['lr'].append(args.learning_rate)
		log['train_loss'].append(train_log['loss'])
		log['train_cer'].append(train_log['cer'])
		log['train_wer'].append(train_log['wer'])
		log['val_loss'].append(val_log['loss'])
		log['val_cer'].append(val_log['cer'])
		log['val_wer'].append(val_log['wer'])

		if epoch>1 and val_log["wer"]<min(log["val_wer"][:-1]):
			torch.save({
						'hyperparameters': {"n_cnn_layers": args.num_cnn_layers,
											"n_rnn_layers": args.num_rnn_layers,
											"rnn_dim": args.hidden_size,
											"n_class": n_class,
											"n_feats": args.nfeats,
											"stride": args.stride,
											"dropout": args.dropout},
						'epoch': log["epoch"],
						'lr': log['lr'],
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'train_loss': log['train_loss'],
						'train_cer': log['train_cer'],
						'train_wer': log['train_wer'],
						'val_loss': log['val_loss'],
						'val_cer': log['val_cer'],
						'val_wer': log['val_wer'],
						}, MODEL_PATH)

		logging.debug("Average Test CER : {}%, with Minimum Test CER: {}%".format(
												100.0*sum(log["val_cer"])/len(log["val_cer"]),
												100*min(log["val_cer"])))
		logging.debug("Average Test WER : {}%, with Minimum Test WER: {}%".format(
												100.0*sum(log["val_wer"])/len(log["val_wer"]),
												100*min(log["val_wer"])))
