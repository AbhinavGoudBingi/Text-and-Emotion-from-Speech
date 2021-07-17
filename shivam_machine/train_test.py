from transform_text import TextTransform
import torch
import torch.nn as nn
from collections import OrderedDict
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename="runtime.log", \
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
					datefmt='%d-%b-%y %H:%M:%S', \
					level=logging.DEBUG, filemode="a")

text_transform=TextTransform()
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, epoch, criterion, optimizer, decoder, scheduler=None):
	### TRAINING LOOP
	model.train()
	running_loss = 0
	test_cer = []
	test_wer = []
	for batch_idx, (data, target, data_len, target_len) in enumerate(train_loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)

		output = model(data)

		# CTC loss
		loss = criterion(torch.log(output).transpose(0,1), target, data_len, target_len)
		running_loss += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		targets = []

		for i in range(target.shape[0]):
			targets.append(text_transform.int_to_text(target[i][:target_len[i]].tolist()))

		target_len = torch.Tensor(target_len)

		#decoded_preds = decoder.decode_(output, target_len)
		decoded_preds = decoder.decode_(output)

		for j in range(len(decoded_preds)):
			test_cer.append(decoder.cer(targets[j], decoded_preds[j]))
			test_wer.append(decoder.wer(targets[j], decoded_preds[j]))
	if scheduler:
		scheduler.step(running_loss)

	avg_cer = sum(test_cer)/len(test_cer)
	avg_wer = sum(test_wer)/len(test_wer)

	logging.debug("Epoch: {}\nTrain_loss: {:.4f}, Train CER: {:.2f}%, Train WER: {:.2f}%".format(epoch, running_loss/len(train_loader), 100*avg_cer, 100*avg_wer))
	return OrderedDict([('loss', running_loss/len(train_loader)),
						('cer', avg_cer),
						("wer", avg_wer)])

def validate(model, test_loader, epoch, criterion, optimizer, decoder):
	model.eval()
	running_loss=0
	test_cer = []
	test_wer = []
	pairs = []
	with torch.no_grad():
		for batch_idx, (data, target, data_len, target_len) in enumerate(test_loader):
			data = data.to(DEVICE)
			target = target.to(DEVICE)

			output = model(data)
			loss = criterion(torch.log(output).transpose(0,1), target, data_len, target_len)

			running_loss += loss.item()
			#reversing from int to text again
			targets = []
			for i in range(target.shape[0]):
				targets.append(text_transform.int_to_text(target[i][:target_len[i]].tolist()))

			target_len = torch.Tensor(target_len)
			#decoded_preds = decoder.decode_(output, target_len)
			decoded_preds = decoder.decode_(output)

			for j in range(len(decoded_preds)):
				test_cer.append(decoder.cer(targets[j], decoded_preds[j]))
				test_wer.append(decoder.wer(targets[j], decoded_preds[j]))
				pairs.append((targets[j], decoded_preds[j]))

	avg_cer = sum(test_cer)/len(test_cer)
	avg_wer = sum(test_wer)/len(test_wer)

	logging.debug("Val loss: {:.4f}, Val CER: {:.2f}%, Val WER: {:.2f}%".format(running_loss/len(test_loader), 100*avg_cer, 100*avg_wer))
	if epoch%10==0:
		intermediate = ["::".join(list(i)) for i in pairs]
		logging.debug(",".join(intermediate))
	return OrderedDict([('loss', running_loss/len(test_loader)),
						('cer', avg_cer),
						("wer", avg_wer)])
