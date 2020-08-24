# Class KNN:
    # fit method to fit the data
    # predict method to make the predictions

# Imports
import numpy as np
from csv import reader

class my_knn():
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(row1, row2):
	    distance = 0.0
	    for i in range(len(row1)-1):
		    distance += (row1[i] - row2[i])**2
	    return np.sqrt(distance)

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return np.sqrt(distance)
 
# Test distance function
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
row0 = dataset[0]
for row in dataset:
	distance = euclidean_distance(row0, row)
	print(f'The Euclidean Distance is: {distance}')

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = []
	for train_row in train:
		e_dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, e_dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = []
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Test distance function
neighbors = get_neighbors(dataset, dataset[0], 3)
for neighbor in neighbors:
	print(f'The Neighbors are: {neighbor}')

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# Test distance function
prediction = predict_classification(dataset, dataset[0], 3)
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))

# Load a CSV file
def load_csv(filename):
	dataset = []
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

# data = load_csv('CS_Build_Week_Algorithms/iris.txt')
# print(data) 

# Test get_accuracy
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = get_accuracy(testSet, predictions)
print(f'The accuracy score is: {accuracy}')

