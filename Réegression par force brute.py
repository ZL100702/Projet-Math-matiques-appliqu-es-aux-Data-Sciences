import numpy as np
import itertools

# Données d'exemple (x : feature, y : cible)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 2.8, 3.6, 4.5, 5.1])

# Définition de la fonction d'erreur (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Recherche par force brute des meilleurs coefficients (a, b) pour y = aX + b
a_values = np.linspace(0, 2, 100)  # Essai de 100 valeurs pour a
b_values = np.linspace(0, 2, 100)  # Essai de 100 valeurs pour b

best_a, best_b = 0, 0
min_error = float('inf')

# Tester toutes les combinaisons possibles de (a, b)
for a, b in itertools.product(a_values, b_values):
    y_pred = a * X + b
    error = mean_squared_error(y, y_pred)
    if error < min_error:
        min_error = error
        best_a, best_b = a, b

print(f"Meilleurs coefficients trouvés : a = {best_a:.4f}, b = {best_b:.4f}")
print(f"Erreur minimale : {min_error:.6f}")
