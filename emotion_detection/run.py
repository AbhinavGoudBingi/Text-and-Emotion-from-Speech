import os
import sys
from collections import OrderedDict
import logging
import warnings
warnings.filterwarnings("ignore")

import tqdm
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data_loader import create_data_dataloader
from model import EmotionRecognizer
from train_test import train, validate


# set random seeds
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark=False

logging.basicConfig(filename="runtime.log", \
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
					datefmt='%d-%b-%y %H:%M:%S', \
					level=logging.DEBUG, filemode="a")
MODEL_PATH = "emotion_recognizer.pth"


if __name__=="__main__":
	log = OrderedDict([('epoch', []),
					   ('lr', []),
					   ("train_loss", []),
					   ("val_loss", []),
					   ('train_accuracy', []),
					   ('val_accuracy', [])])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, test_loader = create_data_dataloader(batch_size=32, nfeats=128)
	model = EmotionRecognizer(input_dims=128,
				 					hidden_dims=256,
				 					num_classes=8,
				 					pool_size=8,
				 					stride=1,
				 					dropout=0.2,
				 					kernel_size=5).to(device)
	optimizer = optim.RMSprop(model.parameters(),lr=0.001, alpha=0.9)
	# print(model)
	start_epoch=1
	criterion = nn.CrossEntropyLoss()
	num_epochs=100
	for epoch in tqdm.tqdm(range(start_epoch, num_epochs+1)):
		train_log = train(model, train_loader, epoch, criterion, optimizer)
		val_log = validate(model, test_loader, epoch, criterion, optimizer)

		log['epoch'].append(epoch)
		log['lr'].append(args.learning_rate)
		log['train_loss'].append(train_log['loss'])
		log['train_accuracy'].append(train_log['accuracy'])
		log['val_loss'].append(val_log['loss'])
		log['val_accuracy'].append(val_log['accuracy'])

		if epoch>1 and train_log["accuracy"]>max(log["accuracy"][:-1]):
			torch.save({
						'hyperparameters': {"kernel_size": 5,
											"rnn_dim": 256,
											"n_feats": 128,
											"stride": 1,
											"dropout": 0.2},
						'epoch': log["epoch"],
						'lr': log['lr'],
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'train_loss': log['train_loss'],
						'train_accuracy': log['train_accuracy'],
						'val_loss': log['val_loss'],
						'val_accuracy': log['val_accuracy'],
						}, MODEL_PATH)
		logging.debug("Average Test Accuracy : {}%, with Minimum Test Accuracy: {}%".format(
												100.0*sum(log["val_accuracy"])/len(log["val_accuracy"]),
												100*min(log["val_accuracy"])))
