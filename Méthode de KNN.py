from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
