import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Create synthetic dataset
np.random.seed(42)
levels = np.random.randint(1, 100, 100)
scores = np.random.randint(0, 1000, 100)
playtime = np.random.randint(1, 50, 100)

data = pd.DataFrame({'Level': levels, 'Score': scores, 'Playtime': playtime})

# Step 1: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 2: Finding the optimal number of clusters using the Elbow method
inertia_values = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)

# Plotting the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# From the Elbow method, choose an appropriate K value (e.g., K=3)
optimal_k = 3  # Set based on Elbow plot

# Step 3: K-Means Clustering with optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=100, random_state=42)
initial_labels = kmeans.fit_predict(scaled_data)

# Final clusters and metrics
final_labels = kmeans.labels_
epoch_size = kmeans.n_iter_
inertia = kmeans.inertia_  # Error rate as inertia

# Display results
print("Improved K-Means Clustering Results:")
print("Initial Clusters (random labels):", initial_labels)
print("Final Clusters:", final_labels)
print("Epoch Size (iterations):", epoch_size)
print("Inertia (Error Rate):", inertia)

# Plotting the final clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['Level'], data['Score'], c=final_labels, cmap='viridis', marker='o')
plt.title('Improved K-Means Clustering of Game Players')
plt.xlabel('Level')
plt.ylabel('Score')
plt.colorbar(label='Cluster')
plt.show()
