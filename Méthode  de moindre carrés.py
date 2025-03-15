import numpy as np

def moindres_carres(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y  # Formule normale

# Exemple
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 2.5, 3.5])
theta = moindres_carres(X, y)
print("Theta optimisé (moindres carrés) :", theta)
