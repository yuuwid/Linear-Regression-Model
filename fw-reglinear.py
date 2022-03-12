import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv('data/data3.csv')

dataset.plot(x='x', y='y', style='o')

plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print('intercept  : ' + str(regressor.intercept_))
print('slope      : ' + str(regressor.coef_))

y_pred = regressor.predict(X_test)
print('R2 score   : ' + str(r2_score(y_test, y_pred)))


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head())

plt.plot(X_test, y_pred, c='r')
# plt.show()