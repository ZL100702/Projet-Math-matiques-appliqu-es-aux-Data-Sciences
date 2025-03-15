from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification

# Génération d'un jeu de données
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, random_state=42)

# Création et entraînement du modèle Naïve Bayes
nb_model = GaussianNB()
nb_model.fit(X, y)

# Prédictions
accuracy = nb_model.score(X, y)
print(f"Précision du modèle Naïve Bayes : {accuracy:.2f}")
