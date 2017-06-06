from sklearn.manifold import TSNE
from sklearn.datasets import fetch_mldata
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt 

mnist = fetch_mldata('MNIST original')

X, y = mnist.data, mnist.target

print X.shape, y.shape

sns.set_style('darkgrid')
sns.set_palette('muted')

shuf = np.c_[X, y]
np.random.shuffle(shuf)
# print shuf[:10]

sample_shuf = shuf[:2000]
print sample_shuf.shape
# print sample_shuf[:10]
sample_X, sample_y = sample_shuf[:,:-1], sample_shuf[:,-1]
print sample_X.shape, sample_y.shape

digits = TSNE(verbose=1, perplexity=40).fit_transform(sample_X)

print 'scattering t-SNE generated points...'

def scatter(x, colors):
	palette = np.array(sns.color_palette('hls', 10))
	f = plt.figure(figsize=(10,10))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
		c=palette[colors.astype(np.int)])
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')
	plt.show()

scatter(digits, sample_y)