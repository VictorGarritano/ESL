from nn_experimental import *

import matplotlib.pyplot as plt 
import numpy as np 
import sklearn
import sklearn.datasets
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

np.random.seed(1337)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
# plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
# plt.show()

"""
from http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

"""

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title("Decision Boundary for hidden layer size 5")
    plt.show()

net = neural_net(eta=1e-2, numrounds=20000, hidden_nodes=[5])

net.fit(X, y.reshape(y.shape[0], 1))
plot_decision_boundary(lambda x: net.predict(x))
