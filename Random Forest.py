import numpy as np
from sklearn.ensemble import RandomForestRegressor

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 2.5, 3, 3.5, 4])


rf = RandomForestRegressor(n_estimators=10)
rf.fit(X, y)
