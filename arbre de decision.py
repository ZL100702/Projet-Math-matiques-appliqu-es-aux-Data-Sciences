import numpy as np
from sklearn.tree import DecisionTreeRegressor

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 2.5, 3, 3.5, 4])


tree = DecisionTreeRegressor()
tree.fit(X, y)
