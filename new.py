import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv', names = header)
array = data.values
x = array[:, 0:8]
y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_x= scaler.fit_transform(x)
(x_train, x_test, y_train, y_test) = train_test_split(rescaled_x,y, test_size=0.3)

model = DecisionTreeClassifier(max_depth=1000, min_samples_split=50,
                               min_samples_leaf=5)


model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

acc = accuracy_score(y_test, y_pred)
print(acc)