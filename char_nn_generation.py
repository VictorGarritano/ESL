from __future__ import print_function
import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def findFiles(path):
	return glob.glob(path)

# print(findFiles('names/*.txt'))

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

# print (n_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', unicode(s, encoding="utf-8"))
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

language_names = {}
all_languages = []

def readLines(filename):
	names = open(filename).read().strip().split('\n')
	return [unicodeToAscii(name) for name in names]

for filename in findFiles('names/*.txt'):
	language = filename.split('/')[-1].split('.')[0]
	all_languages.append(language)
	names = readLines(filename)
	language_names[language] = names

n_languages = len(all_languages)

# print ('# languages: ', n_languages, all_languages)
# print (all_letters[-1])
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size

		self.i2h = nn.Linear(n_languages + input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(n_languages + input_size + hidden_size, output_size)
		self.o2o = nn.Linear(hidden_size + output_size, output_size)
		self.dropout = nn.Dropout(0.1)
		self.softmax = nn.LogSoftmax()

	def forward(self, language, input, hidden):
		input_combined = torch.cat((language, input, hidden), 1)
		hidden = self.i2h(input_combined)
		output = self.i2o(input_combined)
		output_combined = torch.cat((hidden, output), 1)
		output = self.o2o(output_combined)
		output = self.dropout(output)
		output = self	.softmax(output)
		return output, hidden

	def initHidden(self):
		return Variable(torch.zeros(1, self.hidden_size))

def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
	language = randomChoice(all_languages)
	name = randomChoice(language_names[language])
	return language, name

def categoryTensor(language):
	li = all_languages.index(language)
	tensor = torch.zeros(1, n_languages)
	tensor[0][li] = 1
	return tensor

def inputTensor(name):
	tensor = torch.zeros(len(name), 1, n_letters)
	for li in range(len(name)):
		letter = name[li]
		tensor[li][0][all_letters.find(letter)] = 1
	return tensor

def targetTensor(name):
	letter_indexes = [all_letters.find(name[li]) for li in range(1, len(name))]
	letter_indexes.append(n_letters - 1) #EOS
	return torch.LongTensor(letter_indexes)

def randomTrainingSet():
	language, name = randomTrainingPair()
	language_tensor = Variable(categoryTensor(language))
	input_name_tensor = Variable(inputTensor(name))
	target_name_tensor = Variable(targetTensor(name))
	return language_tensor, input_name_tensor, target_name_tensor

criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(language_tensor, input_name_tensor, target_name_tensor):
	hidden = rnn.initHidden()

	rnn.zero_grad()

	loss = 0

	for i in range(input_name_tensor.size()[0]):
		output, hidden = rnn(language_tensor, input_name_tensor[i], hidden)
		loss += criterion(output, target_name_tensor[i])

	loss.backward()

	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output, loss.data[0] / input_name_tensor.size()[0]

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

rnn = RNN(n_letters, 128, n_letters)

n_epochs = 100000
print_every = 5000

start = time.time()

for epoch in range(1, n_epochs + 1):
	output, loss = train(*randomTrainingSet())
	if epoch % print_every == 0:
		print('%s (%d %d%%) %.4f' % 
			(timeSince(start), epoch, float(epoch) / n_epochs * 100, loss))

max_length = 20

def sample(language, start_letter='A'):
	language_tensor = Variable(categoryTensor(language))
	input = Variable(inputTensor(start_letter))
	hidden = rnn.initHidden()

	output_name = start_letter

	for i in range(max_length):
		output, hidden = rnn(language_tensor, input[0], hidden)
		topv, topi = output.data.topk(1)
		topi = topi[0][0]
		if topi == n_letters - 1:
			break
		else:
			letter = all_letters[topi]
			output_name += letter
		input = Variable(inputTensor(letter))

	return output_name

def samples(language, start_letters='ABC'):
	for start_letter in start_letters:
		print(sample(language, start_letter))

samples('Russian', 'RUS')
samples('German', 'GER')
samples('Spanish', 'SPA')
samples('Chinese', 'CHI')