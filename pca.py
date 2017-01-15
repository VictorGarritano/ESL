from sklearn.datasets import load_diabetes
import numpy as np 
from numpy.linalg import svd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

X, y = load_diabetes(return_X_y=True)
X_r = scale(X, with_std=False, with_mean=True) #mean = 0
(N, p) = X_r.shape
print N 
print p 
U, d, V = svd(X_r, full_matrices=False)
D = np.diag(d)
print U.shape 	#N x p orthogonal matrix, whose columns
				#are called left singular vectors

print V.shape	#p x p orthogonal matrix, whose columns
				#are called right singular vectors

print D.shape	#p x p diagonal matrix, with diagonal
				#elements known as singular values

PC = np.dot(U, D) #Principal components of X_r

fig = plt.figure()
plt.scatter(PC[:, 0], PC[:, 1]) #The first two principal
								#components of the data
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('The first two principal components of the data')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(PC[:, 0], PC[:, 1], PC[:, 2])
ax.set_xlabel('First PC')
ax.set_ylabel('Second PC')
ax.set_zlabel('Third PC')
plt.show()