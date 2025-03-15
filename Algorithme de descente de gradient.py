import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)  # Initialisation des paramètres
    for _ in range(epochs):
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta -= lr * gradient
    return theta

# Exemple d'utilisation
X = np.array([[1, 1], [1, 2], [1, 3]])  # Ajout de la colonne de biais
y = np.array([2, 2.5, 3.5])
theta = gradient_descent(X, y)
print("Theta optimisé:", theta)
