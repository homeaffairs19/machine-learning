import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# Create synthetic dataset
np.random.seed(42)
levels = np.random.randint(1, 100, 100)
scores = np.random.randint(0, 1000, 100)
playtime = np.random.randint(1, 50, 100)

data = pd.DataFrame({'Level': levels, 'Score': scores, 'Playtime': playtime})

# Step 1: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 2: K-Distance Graph
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(scaled_data)
distances, indices = neighbors_fit.kneighbors(scaled_data)
distances = np.sort(distances[:, 4], axis=0)

# Plotting the k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('K-Distance Graph')
plt.xlabel('Points sorted by distance')
plt.ylabel('Distance to 5th nearest neighbor')
plt.grid()
plt.show()

# Initial DBSCAN with known parameters
fixed_eps = 0.8  # Adjusted based on the K-Distance graph
fixed_min_samples = 3  # Lowered to encourage clustering
dbscan = DBSCAN(eps=fixed_eps, min_samples=fixed_min_samples)
dbscan_labels = dbscan.fit_predict(scaled_data)

# Count clusters and noise
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

# Display results
print("\nInitial DBSCAN Clustering Results:")
print("Final Clusters (DBSCAN labels):", dbscan_labels)
print("Number of clusters:", n_clusters)
print("Number of noise points (Error Rate):", n_noise)

# Step 3: Evaluate if clustering was successful
if n_clusters > 0:
    # Function to evaluate different eps and min_samples
    def evaluate_dbscan(eps_values, min_samples_values):
        best_score = -1
        best_params = (None, None)

        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(scaled_data)

                # Count valid clusters
                unique_labels = set(labels)
                n_valid_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

                # Check if valid clusters exist
                if n_valid_clusters > 1:  # At least two clusters needed for silhouette score
                    score = silhouette_score(scaled_data, labels)
                    print(f"Evaluating eps={eps:.2f}, min_samples={min_samples}: score={score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)

        return best_params

    # Wider ranges for eps and min_samples
    eps_values = np.arange(0.5, 1.5, 0.1)  # Adjusted eps range
    min_samples_values = range(2, 10)  # Adjusted min_samples range

    # Get best parameters
    best_eps, best_min_samples = evaluate_dbscan(eps_values, min_samples_values)

    if best_eps is None or best_min_samples is None:
        print("No valid parameters found after initial clustering.")
    else:
        print(f"Best parameters found - eps: {best_eps}, min_samples: {best_min_samples}")
else:
    print("No clusters formed with initial parameters.")
