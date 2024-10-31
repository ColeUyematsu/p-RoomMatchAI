from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("test.csv")
X = data.to_numpy()[:,:-1]

### Unscaled
# note: maybe can use this algo to get the clusters, then within each cluster use KNN to get 5 best matches?
scaledX = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(scaledX)
clusters = kmeans.cluster_centers_

plt.scatter(scaledX[:,0],scaledX[:,1],c="blue")
plt.scatter(clusters[:,0],clusters[:,1],c="orange")

y_preds = kmeans.predict(scaledX)

headers = data.iloc[:, -1].tolist()

clusters_dict = {i: [] for i in range(5)}  # Creating a dictionary for clusters

for i, pred in enumerate(y_preds):
    clusters_dict[pred].append(headers[i])

# Print grouped users
for cluster_id, users in clusters_dict.items():
    print(f"Group {cluster_id + 1}: {users}")

plt.show()

