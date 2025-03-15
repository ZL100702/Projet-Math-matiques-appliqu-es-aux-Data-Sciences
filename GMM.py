import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Génération d’un dataset avec 3 clusters
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Création et entraînement du modèle GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Prédiction des clusters
labels = gmm.predict(X)

# Visualisation des clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100, label='Centres GMM')
plt.title("Clustering avec Modèle de Mélange Gaussien (GMM)")
plt.legend()
plt.show()
