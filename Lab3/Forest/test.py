import pandas as pd
import numpy as np

df = pd.read_csv('all/train-set.csv')
train_data = df.values
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators = 10)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 7)
model = model.fit(train_data[0:-1000,15:-1],train_data[0:-1000,-1])
test_data = train_data[-1000:,15:-1]
output = model.predict(test_data)
output = model.predict(test_data)
standard = train_data[-1000:,-1]
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(standard,output)
print(accuracy)