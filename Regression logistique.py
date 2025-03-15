from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, random_state=0)
model = LogisticRegression()
model.fit(X, y)
