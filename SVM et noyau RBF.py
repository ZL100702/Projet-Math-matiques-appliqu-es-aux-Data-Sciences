from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Génération d’un dataset
X, y = datasets.make_moons(n_samples=200, noise=0.2, random_state=42)

# Séparation en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle SVM avec noyau RBF (Radial Basis Function)
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# Prédictions et score
accuracy = svm_model.score(X_test, y_test)
print(f"Précision du SVM : {accuracy:.2f}")

# Visualisation de la séparation des classes
def plot_decision_boundary(model, X, y):
    h = .02  # Taille de la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.show()

plot_decision_boundary(svm_model, X, y)
