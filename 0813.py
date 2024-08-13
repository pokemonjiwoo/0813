import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv', names = header)
#데이터 전처리 : Min-Max 스케일링
array = data.values
x = array[:, 0:8]
y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_x= scaler.fit_transform(x)
#데이터 분할
(x_train, x_test, y_train, y_test) = train_test_split(rescaled_x,y, test_size=0.3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#모델 선택 및 학습
model = LinearRegression()
model.fit(x_train, y_train)
y_train = model.predict(x_train)
y_pred = model.predict(x_test)
y_pred_binary = (y_pred>0.5).astype(int)

#예측 정확도 확인
acc = accuracy_score(y_pred_binary, y_test)
print(acc)

df_y_test = pd.DataFrame(y_test)
df_y_pred_binary = pd.DataFrame(y_pred_binary)
df_y_test.to_csv('./data/y_test.csv')
df_y_pred_binary.to_csv('./data/y_pred.csv')

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)),y_test, color = 'blue', label = 'Actual values',
            marker = 'o')
plt.scatter(range(len(y_pred_binary)), y_pred_binary, color = 'r'
                  , label = 'Predicted Values', marker = 'x')
plt.title('coA')
plt.xlabel('Index')
plt.ylabel('class 0, 1')
plt.legend()
plt.show()

