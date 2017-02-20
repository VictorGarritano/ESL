import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import torch

import argparse

parser = argparse.ArgumentParser(description='Quora Duplicate Questions')
parser.add_argument('--load', type=bool, default=False, help='loads preprocessed dataframe')

args = parser.parse_args()

load = args.load

if not load:
	df = pd.read_csv('quora_duplicate_questions.tsv', delimiter='\t')

	df['text1'] = df['text1'].apply(lambda x: unicode(str(x), 'utf-8'))
	df['text2'] = df['text2'].apply(lambda x: unicode(str(x), 'utf-8'))

	questions = list(df['text1']) + list(df['text2'])


	tfidf = TfidfVectorizer(lowercase=False, )
	tfidf.fit_transform(questions)

	word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

	nlp = spacy.load('en')

	vecs1 = []

	for qu in tqdm(list(df['text1'])):
		doc = nlp(qu)
		mean_vec = np.zeros([len(doc), 300])
		for word in doc:
			vec = word.vector 

			try:
				idf = word2tfidf[str(word)]
			except:
				idf = 0
			mean_vec += vec * idf
		mean_vec = mean_vec.mean(axis=0)
		vecs1.append(mean_vec)
	df['q1_feats'] = list(vecs1)

	vecs2 = []

	for qu in tqdm(list(df['text2'])):
		doc = nlp(qu)
		mean_vec = np.zeros([len(doc), 300])
		for word in doc:
			vec = word.vector 

			try:
				idf = word2tfidf[str(word)]
			except:
				idf = 0
			mean_vec += vec * idf
		mean_vec = mean_vec.mean(axis=0)
		vecs2.append(mean_vec)
	df['q2_feats'] = list(vecs2)

	pd.to_pickle(df, 'data/1_df.pkl')

else:
	print ("Loading Dataframe...")
	df = pd.read_pickle('data/1_df.pkl')
	print ("Dataframe loaded")

