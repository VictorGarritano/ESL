import numpy as np 
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
	_kl_divergence)

from sklearn.utils.extmath import _ravel

RS = 20150101

import matplotlib.pyplot as plt 
import matplotlib.patheffects as PathEffects 
import matplotlib

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context('notebook', font_scale=1.5,
	rc={'lines.linewidth': 2.5})

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

digits = load_digits()
print digits.data.shape
# print (digits['DESCR'])

nrows, ncols = 2, 5
plt.figure(figsize=(6,3))
plt.gray()
for i in range(ncols * nrows):
	ax = plt.subplot(nrows, ncols, i + 1)
	ax.matshow(digits.images[i, ...])
	plt.xticks([]); plt.yticks([])
	plt.title(digits.target[i])
print 'saving generated digits...'
plt.savefig('digits-generated.png', dpi=150)

X = np.vstack([digits.data[digits.target==i]
	for i in range(10)])
y = np.hstack([digits.target[digits.target==i] 
	for i in range(10)])

print 'computing t-SNE...'
digits_proj = TSNE(random_state=RS).fit_transform(X)

print 'scattering t-SNE generated points...'
def scatter(x, colors):
	palette = np.array(sns.color_palette('hls', 10))
	f = plt.figure(figsize=(8,8))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
		c=palette[colors.astype(np.int)])
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')

	print 'writing labels...'
	txts = []
	for i in range(10):
		xtext, ytext = np.median(x[colors == i, :], axis=0)
		txt = ax.text(xtext, ytext, str(i), fontsize=24)
		txt.set_path_effects([
			PathEffects.Stroke(linewidth=5, foreground="w"),
			PathEffects.Normal()])
		txts.append(txt)
	return f, ax, sc, txts

scatter(digits_proj, y)
print 'saving t-SNE generated digits...'
plt.savefig('digits_tsne-generated.png', dpi=120)