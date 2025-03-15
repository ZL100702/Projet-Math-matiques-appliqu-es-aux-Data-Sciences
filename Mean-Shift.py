import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

# Génération d’un jeu de données avec 3 clusters
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Création et entraînement du modèle Mean-Shift
meanshift = MeanShift()
meanshift.fit(X)

# Prédiction des clusters
labels = meanshift.predict(X)

# Visualisation des clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', s=30)
plt.scatter(meanshift.cluster_centers_[:, 0], meanshift.cluster_centers_[:, 1], c='black', marker='x', s=100, label='Centres Mean-Shift')
plt.title("Clustering avec Mean-Shift")
plt.legend()
plt.show()
