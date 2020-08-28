# Imports
import numpy as np
from csv import reader
from collections import Counter

# Rough Draft
#* Calculate distances between the input and training data
#* Find the nearest neighbors based on these distances
#* Get our accuracy score for our predictions

# calculate the Euclidean distance between two vectors
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class my_knn():
	def __init__(self, k):
		self.k = k

	def knn_fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def knn_predict(self, P):
		knn_predictions = [self._knn(i) for i in P]
		return np.array(knn_predictions)

	def _knn(self, x):
		pred_distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
		
		k_indices = np.argsort(pred_distance)[:self.k]
		
		k_nearest_labels = [self.y_train[i] for i in k_indices]
		
		knn_predictions = Counter(k_nearest_labels).most_common(1)
		return knn_predictions[0][0]