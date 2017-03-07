from __future__ import print_function
import os
import numpy as np 
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

print ("Indexing word vectors...")

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs

print('Found %s word vectors.\n' % len(embeddings_index))

print ("Processing text dataset...")

texts = []
labels_index = {}
labels = []

for name in sorted(os.listdir(TEXT_DATA_DIR)):
	path = os.path.join(TEXT_DATA_DIR, name)
	if os.path.isdir(path):
		label_id = len(labels_index)
		labels_index[name] = label_id
		for fname in sorted(os.listdir(path)):
			if fname.isdigit():
				fpath = os.path.join(path, fname)
				f = open(fpath)
				texts.append(f.read())
				f.close()
				labels.append(label_id)

print ('Found %s texts.' % len(texts))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.\n' % len(word_index))

print ("Padding sentences...")
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print ("Done\n")

print ("Labels to categorical...")
labels = to_categorical(np.asarray(labels))
print ('Shape of data tensor: ', data.shape)
print ('Shape of label tensor: ', labels.shape)
print ("Done\n")

print ("Splitting data on train validation set...")
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
print ("number of validations samples: ", nb_validation_samples)
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
print ("Done\n")

print ("Preparing embedding matrix...")

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
	if i>= MAX_NB_WORDS:
		continue
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
print ("Done\n")

print ("Building pre-trained embedding layer...")
embedding_layer = Embedding(nb_words,
							EMBEDDING_DIM,
							weights=[embedding_matrix],
							input_length=MAX_SEQUENCE_LENGTH,
							trainable=False)
print ("Done\n")

print("Training model...\n")
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
			  optimizer='rmsprop',
			  metrics=['acc'])

print (model.summary())

"""
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 1000)          0                                            
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 1000, 100)     2000000     input_1[0][0]                    
____________________________________________________________________________________________________
convolution1d_1 (Convolution1D)  (None, 996, 128)      64128       embedding_1[0][0]                
____________________________________________________________________________________________________
maxpooling1d_1 (MaxPooling1D)    (None, 199, 128)      0           convolution1d_1[0][0]            
____________________________________________________________________________________________________
convolution1d_2 (Convolution1D)  (None, 195, 128)      82048       maxpooling1d_1[0][0]             
____________________________________________________________________________________________________
maxpooling1d_2 (MaxPooling1D)    (None, 39, 128)       0           convolution1d_2[0][0]            
____________________________________________________________________________________________________
convolution1d_3 (Convolution1D)  (None, 35, 128)       82048       maxpooling1d_2[0][0]             
____________________________________________________________________________________________________
maxpooling1d_3 (MaxPooling1D)    (None, 1, 128)        0           convolution1d_3[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 128)           0           maxpooling1d_3[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           16512       flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 20)            2580        dense_1[0][0]                    
====================================================================================================
Total params: 2,247,316
Trainable params: 247,316
Non-trainable params: 2,000,000
____________________________________________________________________________________________________

"""
model.fit(x_train, y_train, validation_data=(x_val, y_val),
		  nb_epoch=4, batch_size=128)