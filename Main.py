import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from learn.split_data import split
from learn.linear_model import LinearRegression
import learn.error_score as score


df = pd.read_csv('data/student_scores.csv')
X = df['Hours'].values
y = df['Scores'].values
X2 = df.iloc[:, :-1].values
y2 = df.iloc[:, 1].values

test_size = 0.2
data_length = len(X)
data_test_length = np.floor(data_length * test_size)
data_train_length = np.ceil(data_length - data_test_length)

X_train, X_test, y_train, y_test = split(X, y, test_size=test_size, 
                                         shuffle=True)

regresor = LinearRegression()

regresor.fit_data(X_train, y_train)

print()
print('Data Train : ' + str(int(data_train_length)))
print('Data Test  : ' + str(int(data_test_length)))
print()
print('intercept  : ' + str(regresor.intercept))
print('slope      : ' + str(regresor.slope))
print()
y_pred = regresor.predic(X_test)
print('mae score  : ' + str(score.mae_score(y_test, y_pred)))
print('mse score  : ' + str(score.mse_score(y_test, y_pred)))
print('R2 score   : ' + str(score.r2_score(y_test, y_pred)))
print()

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head())

X_res, y_res = regresor.trained()

plt.title('Student Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.scatter(X2, y2, c='b', label='Data Train')
plt.scatter(X_test, y_test, c='g', label='Data Test')
plt.scatter(X_test, y_pred, c='y', label='Data Predic')
plt.plot(X_res, y_res, c='r', label='Regressor')

plt.legend()
plt.show()