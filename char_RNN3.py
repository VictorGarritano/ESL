from __future__ import print_function
import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.optim as optim
import random
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def findFiles(path):
	return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

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

def letterToIndex(letter):
	return all_letters.find(letter)

def lineToTensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for li, letter in enumerate(name):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)
    language_i = top_i[0][0]
    return all_languages[language_i] , language_i

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

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
                        self.input_size,
                        self.hidden_size,
                        self.num_layers)
        self.fc1 = nn.Linear(self.hidden_size, n_languages)
        self.h0 = Variable(torch.randn(
            self.num_layers * 1,1,self.hidden_size))
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        out, hn = self.rnn(x, self.h0)
        y = hn[-1].view(1,self.hidden_size)
        y = self.fc1(y)
        y = self.softmax(y)
        return y

net = Net(n_letters,256,3)

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

n_epochs = 300000

for epoch in range(1, n_epochs + 1):
    language, name, language_tensor, name_tensor = randomTrainingPair()
    optimizer.zero_grad()
    output = net(name_tensor)

    loss = criterion(output, language_tensor)
    loss.backward()

    optimizer.step()

    if epoch % 1000 == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'RIGHT' if guess == language else 'WRONG (%s)' % language
        print ('%.4f %.4f %s / %s %s' % (float(epoch) / n_epochs,
            loss.data[0], name, guess, correct))

confusion = torch.zeros(n_languages, n_languages)
n_confusion = 10000


for i in range(n_confusion):
    language, name, language_tensor, name_tensor = randomTrainingPair()
    output = net(name_tensor)
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

plt.show()