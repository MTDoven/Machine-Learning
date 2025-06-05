import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

dbscan = DBSCAN(eps=0.5, min_samples=12)
dbscan_labels = dbscan.fit_predict(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

dbscan_silhouette = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
kmeans_silhouette = silhouette_score(X, kmeans_labels)

print("DBSCAN Silhouette Score:", dbscan_silhouette)
print("K-means Silhouette Score:", kmeans_silhouette)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap="viridis", s=50)
plt.title("DBSCAN Clustering")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", s=50)
plt.title("K-means Clustering")
plt.colorbar()

plt.show()
