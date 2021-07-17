from collections import OrderedDict
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename="runtime.log", \
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
					datefmt='%d-%b-%y %H:%M:%S', \
					level=logging.DEBUG, filemode="a")

import torch
import torch.nn
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, epoch, criterion, optimizer):
	### TRAINING LOOP
	model.train()
	running_loss = 0
	accuracy = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)

		output = model(data)
		if torch.argmax(output)==target:
			accuracy += 1
		loss = criterion(output, target)
		running_loss += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	logging.debug("Epoch: {}\nTrain_loss: {:.4f}, Train Accuracy: {:.2f}%".format(epoch, running_loss/len(train_loader), 100*accuracy/(len(train_loader))))
	return OrderedDict([('loss', running_loss/len(train_loader))
						('accuracy', (accuracy/len(train_loader))*100)])

def validate(model, test_loader, epoch, criterion, optimizer):
	### TRAINING LOOP
	model.eval()
	running_loss = 0
	accuracy = 0
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data = data.to(DEVICE)
			target = target.to(DEVICE)

			output = model(data)
			if torch.argmax(output)==target:
				accuracy += 1
			loss = criterion(output, target)
			running_loss += loss.item()
	logging.debug("Epoch: {}\nVal_loss: {:.4f}, Val Accuracy: {:.2f}%".format(epoch, running_loss/len(test_loader), 100*accuracy/(len(test_loader))))
	return OrderedDict([('loss', running_loss/len(test_loader))
						('accuracy', (accuracy/len(test_loader))*100)])
