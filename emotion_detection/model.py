import os
import sys

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class EmotionRecognizer(nn.Module):

	def __init__(self,
				 input_dims=128,
				 hidden_dims=256,
				 num_classes=8,
				 pool_size=8,
				 stride=1,
				 dropout=0.2,
				 kernel_size=5):
		super(EmotionRecognizer, self).__init__()
		self.conv1 = nn.Conv2d(1,
							   64,
							   kernel_size= kernel_size,
							   stride=stride,
							   padding= kernel_size//2)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.max_pool = nn.MaxPool2d(kernel_size=(8))
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(64, 64,
							   kernel_size= kernel_size,
							   stride=stride,
							   padding= kernel_size//2)
		self.lstm = nn.GRU(input_size=64,
						   hidden_size=hidden_dims,
						   num_layers=1,
						   bidirectional=False)
		self.linear = nn.Linear(hidden_dims, num_classes)
		self.softmax = nn.Softmax()

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.dropout1(x)
#		x = self.max_pool(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.dropout2(x)
		#x = nn.Flatten(x)
		x = self.lstm(x)
		x = self.linear(x)
		x = self.softmax(x)
