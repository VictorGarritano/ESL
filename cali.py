import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import os.path

df = pd.read_csv('cal_housing.csv', header=None, 
	names=['MedHouseVal','MedInc', 'HouseAge', 
	'TotRooms', 'TotBedrms','Population',
	'Households','Latitude', 'Longitude'], dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(
	df.drop('MedHouseVal',1), df[['MedHouseVal']],
	test_size=0.2)

params = {'loss': 'huber','n_estimators': 1000, 
		'criterion': 'mae', 'max_depth': None,
		'max_leaf_nodes': 6}


clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
print (clf.score(X_test, y_test))

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(
    	y_test.as_matrix().reshape(-1)
    	, y_pred)

train_score = clf.train_score_

np.save('train_score_california.npy', train_score)
np.save('test_score_california.npy', test_score)

if os.path.exists("train_score_california.npy") and os.path.exists("test_score_california.npy"):
	print ("Done")
else:
	print("Save error")
