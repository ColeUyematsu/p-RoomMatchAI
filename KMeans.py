from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

X = pd.read_csv("test.csv")
pca = PCA(n_components=2)
names = X['NAME']
X = X.drop('NAME',axis=1)

scaledX = StandardScaler().fit_transform(X)
normedX = pd.DataFrame(normalize(scaledX))

kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(normedX)
clusters = kmeans.cluster_centers_
kmeans_labels = kmeans.predict(normedX)

projectedX = pca.fit_transform(normedX)
plt.scatter(projectedX[:,0],projectedX[:,1],c="blue")
plt.scatter(clusters[:,0],clusters[:,1],c="orange")

clusters_dict = {i: [] for i in range(20)}
for label, name in zip(kmeans_labels, names):
    clusters_dict[label].append(name)

top_matches = {}
all_scores = []
for cluster_id, users in clusters_dict.items():
    if len(users) > 1:
        indices = [names[names == member].index[0] for member in users]
        cluster_data = normedX.iloc[indices].to_numpy()
        distances = squareform(pdist(cluster_data, metric='euclidean'))
        
        for i, person_index in enumerate(indices):
            person_name = names[person_index]
            closest_indices = distances[i].argsort()[1:6]
            scores = [1 / (1 + distances[i][j]) for j in closest_indices]
            all_scores.extend(scores)
            top_matches[person_name] = [(users[j], score) for j, score in zip(closest_indices, scores)]

# min_score, max_score = min(all_scores), max(all_scores)
# for person, matches in top_matches.items():
#     top_matches[person] = [(match[0], (match[1] - min_score) / (max_score - min_score)) for match in matches]

for person in sorted(top_matches.keys(), key=lambda x: int(x.split('_')[1])):
    matches = top_matches[person]
    match_info = ', '.join([f"{match[0]} (Score: {match[1]:.2f})" for match in matches])
    print(f"Top matches for {person}: {match_info}")

max_matches = max(len(matches) for matches in top_matches.values())

padded_matches = {}
for person, matches in top_matches.items():
    padded_matches[person] = matches + [(None, None)] * (max_matches - len(matches))

sorted_keys = sorted(padded_matches.keys(), key=lambda x: int(x.split('_')[1]))
df = pd.DataFrame.from_dict({k: padded_matches[k] for k in sorted_keys}, orient="index")

print(df)

plt.show()
