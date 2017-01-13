import pandas as pd 
import numpy as np
from sklearn import preprocessing
from numpy.linalg import inv

data = pd.read_csv('prostate.csv')

data.loc[data["train"] == 'T', "train"] = 1
data.loc[data["train"] == 'F', "train"] = 0
data = data.drop("id", 1)

#======= Least Squares =======

#Train
data_train = data[data["train"] == 1]
data_train = data_train.drop("train", 1)

y = data_train[["lpsa"]]
X = data_train.drop("lpsa", 1)
X_train = X.as_matrix()
X_train = preprocessing.scale(X_train)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]	
					
y_train = y.apply(lambda x: preprocessing.scale(x))

#Test
X_test = data[data["train"] == 0]
X_test = X_test.drop('lpsa', 1)
X_test = X_test.drop('train', 1)
X_test = X_test.as_matrix()
X_test = preprocessing.scale(X_test)
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

y_test = data[data["train"] == 0]
y_test = y_test[['lpsa']]
y_test = y_test.apply(lambda x: preprocessing.scale(x))
		
beta_ls = np.dot(
	inv(np.dot(X_train.T, X_train)), 
	np.dot(X_train.T, y_train))	

y_hat_ls = np.dot(X_test, beta_ls)

mse_ls = np.mean((np.square(y_test - y_hat_ls)))

print mse_ls

#=============================