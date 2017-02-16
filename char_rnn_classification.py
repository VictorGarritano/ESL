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

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

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

# print (n_languages)
# print (language_names['Portuguese'][:10])

def letterToIndex(letter):
	return all_letters.find(letter)

def lineToTensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for li, letter in enumerate(name):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax()

	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output)
		return output, hidden

	def initHidden(self):
		return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_languages)

'''
====================================================

Network step


input = Variable(lineToTensor('Albert'))
hidden = Variable(torch.zeros(1, n_hidden))

output, nex_hidden = rnn(input[0], hidden)
# print (output)


====================================================
'''

def categoryFromOutput(output):
	top_n, top_i = output.data.topk(1)
	language_i = top_i[0][0]
	return all_languages[language_i] , language_i

# print (categoryFromOutput(output))

def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
	language = randomChoice(all_languages)
	name = randomChoice(language_names[language])
	language_tensor = Variable(
						torch.LongTensor(
							[all_languages.index(language)]
							)
						)
	name_tensor = Variable(lineToTensor(name))
	return language, name, language_tensor, name_tensor

# for i in range(10):
# 	language, name, language_tensor, name_tensor = randomTrainingPair()
# 	print ("Language: ", language, "-- Name: ", name)

criterion = nn.NLLLoss()

learning_rate = 0.005

def train(language_tensor, name_tensor):
	hidden = rnn.initHidden()

	rnn.zero_grad()

	for i in range(name_tensor.size()[0]):
		output, hidden = rnn(name_tensor[i], hidden)

	loss = criterion(output, language_tensor)
	loss.backward()

	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output, loss.data[0]

n_epochs = 300000
print_every = 10000
# plot_every = 10000

rnn = RNN(n_letters, n_hidden, n_languages)

current_loss = 0
all_losses = []

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
	language, name, language_tensor, name_tensor = randomTrainingPair()
	output, loss = train(language_tensor, name_tensor)
	current_loss += loss

	if epoch % print_every == 0:
		guess, guess_i = categoryFromOutput(output)
		correct = 'RIGHT' if guess == language else 'WRONG (%s)' % language
		print('%d %.2f%% (%s) %.4f %s / %s %s' % 
			(epoch, float(epoch) / n_epochs * 100, timeSince(start), loss, name, guess, correct))

	# if epoch % plot_every == 0:
	# 	all_losses.append(float(current_loss) / plot_every)
 #      	current_loss = 0

confusion = torch.zeros(n_languages, n_languages)
n_confusion = 10000

def evaluate(name_tensor):
	hidden = rnn.initHidden()

	for i in range(name_tensor.size()[0]):
		output, hidden = rnn(name_tensor[i], hidden)

	return output

for i in range(n_confusion):
	language, name, language_tensor, name_tensor = randomTrainingPair()
	output = evaluate(name_tensor)
	guess, guess_i = categoryFromOutput(output)
	language_i = all_languages.index(language)
	confusion[language_i][guess_i] += 1

for i in range(n_languages):
	confusion[i] /= confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_languages, rotation=90)
ax.set_yticklabels([''] + all_languages)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.savefig('char_nn.svg', format='svg', dpi=1200)