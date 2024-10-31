import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as shc
import re

X = pd.read_csv("test.csv")
names = X['NAME']
X = X.drop('NAME', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal, columns=['P1', 'P2'])

plt.figure(figsize=(12, 8))
plt.title('Dendrogram for Roommate Preferences (20 Clusters)')
dendrogram = shc.dendrogram(shc.linkage(X_principal, method='ward'))

ac20 = AgglomerativeClustering(n_clusters=20)
labels_20 = ac20.fit_predict(X_normalized)

assert len(labels_20) == len(names), "Some individuals were not assigned to a cluster."
assert len(set(labels_20)) == 20, "The clustering did not result in 20 clusters."

clusters = {i: [] for i in range(20)}
for label, name in zip(labels_20, names):
    clusters[label].append(name)

top_matches = {}
for cluster_id, members in clusters.items():
    if len(members) > 1:
        indices = [names[names == member].index[0] for member in members]
        cluster_data = X_normalized.iloc[indices].to_numpy()
        distances = squareform(pdist(cluster_data, metric='euclidean'))
        for i, person_index in enumerate(indices):
            person_name = names[person_index]
            closest_indices = distances[i].argsort()[1:6]
            top_matches[person_name] = [members[j] for j in closest_indices]

for person in sorted(top_matches.keys(), key=lambda x: int(x.split('_')[1])):
    matches = top_matches[person]
    person_num = re.sub(r"Person_", "", person)
    match_nums = [re.sub(r"Person_", "", match) for match in matches]
    print(f"{person_num}: {', '.join(match_nums)}")

plt.figure(figsize=(8, 8))
plt.scatter(X_principal['P1'], X_principal['P2'], c=labels_20, cmap='tab20', s=50)
plt.title("Agglomerative Clustering (20 Clusters)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.show()
