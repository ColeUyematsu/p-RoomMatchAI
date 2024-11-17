from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

X = pd.read_csv("test.csv") # reads csv file into pandas df
names = X['NAME'] # saving the names
X = X.drop('NAME',axis=1) # removing the column of names from df

# scaling the data
scaledX = StandardScaler().fit_transform(X)
normedX = pd.DataFrame(normalize(scaledX)) 

# fitting the model
kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(normedX)
clusters = kmeans.cluster_centers_ # saving the clusters
kmeans_labels = kmeans.predict(normedX) # predicting from our model

# pca for visualization of data
pca = PCA(n_components=2)
projectedX = pca.fit_transform(normedX)

# graphing the data
plt.scatter(projectedX[:,0],projectedX[:,1],c="blue")
plt.scatter(clusters[:,0],clusters[:,1],c="orange")

# creating dictionary for every individual
clusters_dict = {i: [] for i in range(20)}
for label, name in zip(kmeans_labels, names):
    clusters_dict[label].append(name)

top_matches = {} # empty dict to store the matches
all_scores = [] # empty list to store the scores

# scoring and assigning
for cluster_id, users in clusters_dict.items():
    if len(users) > 1:
        # creating the indices for each person
        indices = [names[names == member].index[0] for member in users]
        cluster_data = normedX.iloc[indices].to_numpy()
        # getting the euclidean distances between all people
        distances = squareform(pdist(cluster_data, metric='euclidean'))

        # loop to go through very person; scoring the distances for everyone
        for i, person_index in enumerate(indices):
            person_name = names[person_index]
            closest_indices = distances[i].argsort()[1:6] # getting the closest people
            scores = [1 / (1 + distances[i][j]) for j in closest_indices] # scoring
            all_scores.extend(scores)
            # assigning the top matches per person with the score
            top_matches[person_name] = [(users[j], score) for j, score in zip(closest_indices, scores)]

### experimental min-max normalization
# min_score, max_score = min(all_scores), max(all_scores)
# for person, matches in top_matches.items():
#     top_matches[person] = [(match[0], (match[1] - min_score) / (max_score - min_score)) for match in matches]

# making it look nicer
for person in sorted(top_matches.keys(), key=lambda x: int(x.split('_')[1])):
    matches = top_matches[person]
    match_info = ', '.join([f"{match[0]} (Score: {match[1]:.2f})" for match in matches])
    print(f"Top matches for {person}: {match_info}")

### part of the min-max normalization
# max_matches = max(len(matches) for matches in top_matches.values())

# padded_matches = {}
# for person, matches in top_matches.items():
#     padded_matches[person] = matches + [(None, None)] * (max_matches - len(matches))

# # turning it back into a pandas df
# sorted_keys = sorted(padded_matches.keys(), key=lambda x: int(x.split('_')[1]))
# df = pd.DataFrame.from_dict({k: padded_matches[k] for k in sorted_keys}, orient="index")

plt.show()
