import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

#----------------------------------------------------------
# Example in the pdf
np.random.seed(0)
indices = np.random.permutation(len(iris_x))
iris_x_random = iris_x[indices]
iris_y_random = iris_y[indices]
iris_x_train = iris_x_random[:-10]
iris_y_train = iris_y_random[:-10]
iris_x_test = iris_x_random[-10:]
iris_y_test = iris_y_random[-10:]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(iris_x_train,iris_y_train)

from sklearn.metrics import accuracy_score

iris_y_result = knn.predict(iris_x_test)


#----------------------------------------------------------


# Question1
# error of the classifier
print("Question1: ")
print("The predicted result: ")
print(iris_y_result)
print("The correct result: ")
print(iris_y_test)
error = 1 - accuracy_score(iris_y_test,iris_y_result)
print("The error is : {0}".format(error))

# the optional k of knn
k_op = 0
rate_max = 0
for i in range(1,len(iris_x_train)+1):
	knn_op = KNeighborsClassifier(n_neighbors = i)
	knn_op.fit(iris_x_train,iris_y_train)
	knn_op_result = knn_op.predict(iris_x_test)
	if rate_max < accuracy_score(iris_y_test,knn_op_result):
		rate_max = accuracy_score(iris_y_test,knn_op_result)
		k_op = i
print("The optimal parameter k is : {0}".format(k_op) + " and its accuracy is : {0}".format(rate_max))
print()


#----------------------------------------------------------

# Question2
print("Question2: ")
# use 2 other classifiers
# First one: svm
from sklearn import svm
clf1 = svm.SVC(gamma=0.001, C=100.)
clf1.fit(iris_x_train,iris_y_train)
clf1.predict(iris_x_test)
clf1_y_result = clf1.predict(iris_x_test)
accuracy_clf1 = accuracy_score(iris_y_test,clf1_y_result)
print("The accuracy of svm : {0}".format(accuracy_clf1))

# Second one: random forest
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(max_depth=2, random_state=0)
clf2.fit(iris_x_train,iris_y_train)
clf2.predict(iris_x_test)
clf2_y_result = clf2.predict(iris_x_test)
accuracy_clf2 = accuracy_score(iris_y_test,clf2_y_result)
print("The accuracy of rf : {0}".format(accuracy_clf2))

# Use cross-validation to evaluate the classifiers
from sklearn.model_selection import KFold

kf = KFold(n_splits = 15)
kf.get_n_splits(iris_x)
rate1 = []
rate2 = []
rate3 = []
for train_index, test_index in kf.split(iris_x):
	x_train, x_test = iris_x_random[train_index], iris_x_random[test_index]
	y_train, y_test = iris_y_random[train_index], iris_y_random[test_index]
	clf1 = KNeighborsClassifier()
	clf2 = svm.SVC(gamma=0.001, C=100.)
	clf3 = RandomForestClassifier(max_depth=2, random_state=0)
	clf1.fit(x_train,y_train)
	clf2.fit(x_train,y_train)
	clf3.fit(x_train,y_train)
	y_result1 = clf1.predict(x_test)
	y_result2 = clf2.predict(x_test)
	y_result3 = clf3.predict(x_test)
	rate1.append(accuracy_score(y_test, y_result1))
	rate2.append(accuracy_score(y_test, y_result2))
	rate3.append(accuracy_score(y_test, y_result3))

print("The average accuracy of knn is : {0}".format(sum(rate1)/len(rate1)))
print("The average accuracy of svm is : {0}".format(sum(rate2)/len(rate2)))
print("The average accuracy of rf is : {0}".format(sum(rate3)/len(rate3)))

# Compare evaluation results of the three classifiers

list_rate = [sum(rate1)/len(rate1),sum(rate2)/len(rate2),sum(rate3)/len(rate3)]
index_max = list_rate.index(max(list_rate))

if index_max == 0:
	print("The best is : knn")
elif index_max == 1:
	print("The best is : svm")
else:
	print("The best is : rf")

print()
#----------------------------------------------------------
print("Question3: ")


# Write a majority class classifier: 
# a classifier that predicts the class label that is more frequent in the dataset
class NewClassifier:
	def __init__ (self): 
		self.max_y = 0

# The function fit will save the the most frequent value of Y 
	def fit(self, X, Y):
		from collections import Counter
		list_y = np.unique(Y)
		num_y = []
		for i in range(0,len(list_y)):
			num_y.append(Counter(Y)[list_y[i]])
		self.max_y = list_y[num_y.index(max(num_y))]
		return self

	def predict(self , X):
		list_result = []
		for i in range(0,len(X)):
			list_result.append(self.max_y)
		return list_result

clff = NewClassifier()
clff.fit(iris_x_train,iris_y_train)
clff_y_result = clff.predict(iris_x_test)
accuracy_clff = accuracy_score(iris_y_test,clff_y_result)
print("The accuracy of newClf : {0}".format(accuracy_clff))


# Use the majority class classifier to evaluate one dataset
# and justify why the evaluation results using the new classifier are correct
rate4 = []
kf = KFold(n_splits = 10)
for train_index, test_index in kf.split(iris_x_train):
	x_train, x_test = iris_x_train[train_index], iris_x_train[test_index]
	y_train, y_test = iris_y_train[train_index], iris_y_train[test_index]
	clf4 = NewClassifier()
	clf4.fit(x_train,y_train)
	y_result4 = clf4.predict(x_test)
	rate4.append(accuracy_score(y_test, y_result4))
print("The average accuracy of new clf is : {0}".format(sum(rate4)/len(rate4)))


# Create another classifier with higher performance than the majority class classifier
class NewNewClassifier:
	def __init__ (self): 
		self.train_x = []
		self.train_y = []

# This method will classify by the sum of the data, the keep their average of sum
# The classifier will return the result whose average of sum is most closed to that of test data
	def fit(self, X, Y):
		list_y = np.unique(Y)
		sort_x = [[] for i in range(len(list_y))]
		for i in range(0,len(X)):
			sort_x[list_y.tolist().index(Y[i])].append(sum(X[i]))
		for j in range(0,len(sort_x)):
			self.train_x.append(sum(sort_x[j])/len(sort_x[j]))
		self.train_y = list_y
		return self

	def predict(self , X):
		list_result = []
		for i in range(len(X)):
			list_compare = []
			for j in range(len(self.train_x)):
				list_compare.append(abs(self.train_x[j] - sum(X[i])))
			list_result.append(self.train_y[list_compare.index(min(list_compare))])
		return list_result
clfff = NewNewClassifier()
clfff.fit(iris_x_train,iris_y_train)
clfff_y_result = clfff.predict(iris_x_test)
accuracy_clfff = accuracy_score(iris_y_test,clfff_y_result)
print("The accuracy of NewNewClf : {0}".format(accuracy_clfff))

rate5 = []
kf = KFold(n_splits = 10)
for train_index, test_index in kf.split(iris_x_train):
	x_train, x_test = iris_x_train[train_index], iris_x_train[test_index]
	y_train, y_test = iris_y_train[train_index], iris_y_train[test_index]
	clf5 = NewNewClassifier()
	clf5.fit(x_train,y_train)
	y_result5 = clf5.predict(x_test)
	rate5.append(accuracy_score(y_test, y_result5))
print("The average accuracy of NewNewClf is : {0}".format(sum(rate5)/len(rate5)))