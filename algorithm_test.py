# Imports
import numpy as np
from algorithm import my_knn
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data

load_data = datasets.load_iris()
data = load_data.data
target = load_data.target

# Split the data into x train, test and y train, test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)


# Instantiate model using sklearn
sklearn_model = KNeighborsClassifier(n_neighbors=3)

# fit the model using sklearn
sklearn_model.fit(X_train, y_train)

# Complete a prediction using the sklearn model
predict = sklearn_model.predict(X_test)

# Sklearn model accurace score
print("Sklearn model accuracy:{0:.3f}".format(accuracy_score(y_test, predict)))


# Setting sklearn model y pred
y_pred = sklearn_model.predict([X_test[0]])
print("Sklearn model y_pred", y_pred)

# Instantiate my knn model
my_knn_model = my_knn(k = 3)

# Fit my knn model
my_knn_model.knn_fit(X_train, y_train)

# Complete predictions using my knn model
predictions = my_knn_model.knn_predict(X_test)

# My models accuracy score
my_accuracy = np.sum(predictions == y_test) / len(y_test)
print("My knn model accuracy score:{0:.3}".format(my_accuracy))


# Setting my knn model y pred
y_pred = my_knn_model.knn_predict([X_test[0]])
print("My knn model y_pred", y_pred)