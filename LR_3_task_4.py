import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
ypred = regr.predict(X_test)

print("regr.coef_ =", regr.coef_)
print("regr.intercept_ =", regr.intercept_)
print("r2_score =", round(r2_score(y_test, ypred), 2))
print("Mean_absolute_error =", round(mean_absolute_error(y_test, ypred), 2))
print("Mean_squared_error =", round(mean_squared_error(y_test, ypred), 2))

fig, ax = plt.subplots()
ax.scatter(y_test, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
