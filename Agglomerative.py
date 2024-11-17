import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as shc
import re

X = pd.read_csv("test.csv") # load the dataset
names = X['NAME'] # store the names to be used to label matches later
X = X.drop('NAME', axis=1) # remove names column from the dataset (not used to train model)

# Scale and normalize the data 
# We scale and normalize to help make the dataset uniform and prevent certain features or data points from 
# disproportionately affecting the results, leading to more accurate and meaningful clusters.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)

# Reduce dimensions using principal component analysis making it possible to plot dendogram
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal, columns=['P1', 'P2'])

# Create a dendrogram to visualize clusters
plt.figure(figsize=(12, 8))
plt.title('Dendrogram for Roommate Preferences (20 Clusters)')
dendrogram = shc.dendrogram(shc.linkage(X_principal, method='ward')) # Plot the dendrogram

# Agglomerative Clustering 
ac20 = AgglomerativeClustering(n_clusters=20) # set the number of clusters for agglomerative clustering
labels_20 = ac20.fit_predict(X_normalized) # fit it using the data we loaded, cleaned, scaled and normalized

# If there are people that are not in a cluster print "some individuals were not assigned to a cluster"
assert len(labels_20) == len(names), "Some individuals were not assigned to a cluster."
assert len(set(labels_20)) == 20, "The clustering did not result in 20 clusters."

# Organize the individuals into clusters
clusters = {i: [] for i in range(20)}
for label, name in zip(labels_20, names):
    clusters[label].append(name) # group names by their cluster labels 

# Calculate similarity score within each cluster
output_data = {}
for cluster_id, members in clusters.items():
    # Only process clusters with multiple members
    if len(members) > 1:
        indices = [names[names == member].index[0] for member in members]
        cluster_data = X_normalized.iloc[indices].to_numpy()
        distances = squareform(pdist(cluster_data, metric='euclidean'))
        
        max_distance = distances.max()
        similarity_scores = 1 - (distances / max_distance)
        
        # Process each individual in the cluster
        for i, person_index in enumerate(indices):
            person_name = names[person_index] # Get the person's name
            closest_indices = distances[i].argsort()[1:6] # Find the 5 closest members
            raw_scores = [similarity_scores[i, j] for j in closest_indices] # Raw similarity scores
            
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            # Adjust the similarity scores to a 0.5-1 scale
            adjusted_scores = [
                0.5 + (score - min_score) ** 0.7 / (max_score - min_score) ** 0.7 * 0.5
                for score in raw_scores
            ]
            
            # Format the matches with adjusted scores
            matches = ", ".join(f"{members[j]} (score: {round(adjusted_scores[k], 2)})"
                                for k, j in enumerate(closest_indices))
            
            output_data[person_name] = matches
# Save clustering results to a CSV file
output_df = pd.DataFrame(list(output_data.items()), columns=['Person', 'Top Matches'])
output_df.to_csv("Agglomerative_matches.csv", index=False)

print("Top matches with adjusted scores saved to 'top_matches_grouped.csv'.")

# Visualize clustering results
plt.figure(figsize=(8, 8))
plt.scatter(X_principal['P1'], X_principal['P2'], c=labels_20, cmap='tab20', s=50)
plt.title("Agglomerative Clustering (20 Clusters)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.show()