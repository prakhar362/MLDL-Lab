import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic dataset
np.random.seed(42)
data_size = 300

data = {
    "income": np.random.normal(50000, 15000, data_size),
    "spending_score": np.random.normal(50, 20, data_size),
    "savings": np.random.normal(20000, 10000, data_size),
    "online_freq": np.random.normal(10, 5, data_size)
}

df = pd.DataFrame(data)
print(df.head())

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -----------------------------------
# K-MEANS CLUSTERING
# -----------------------------------

# Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

# Optimal k (example)
k_opt = 4

kmeans = KMeans(n_clusters=k_opt, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Silhouette Score
print("K-Means Silhouette Score:", silhouette_score(X_scaled, kmeans_labels))


# -----------------------------------
# HIERARCHICAL CLUSTERING
# -----------------------------------

# Dendrogram
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10,5))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Apply Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=4)
hc_labels = hc.fit_predict(X_scaled)

# Silhouette Score
print("Hierarchical Clustering Silhouette Score:", silhouette_score(X_scaled, hc_labels))

