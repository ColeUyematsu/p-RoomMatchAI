from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


data = pd.read_csv("test.csv") # reads converts the CSV -> Pandas DF
X = data.to_numpy()[:,1:].T # on assumption that each question is a row, each column is a person


pca = PCA(n_components=2)


### PCA Projected (i tried it unprojected, can change it back if projecting is unncessary)
'''Fitting the data to our model'''
scaledX = StandardScaler().fit_transform(X) # first scale the data
projectedX = pca.fit_transform(scaledX) # project PCA
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(projectedX) # fit kmeans algo. to projected data


clusters = kmeans.cluster_centers_ # our cluster centers


'''Graphing our data and cluster points'''
plt.scatter(projectedX[:,0],scaledX[:,1],c="green")
plt.scatter(clusters[:,0],clusters[:,1],c="red")


y_preds = kmeans.predict(projectedX) # sorting each person to a cluster
headers = data.columns.tolist()[1:] # each user in order


clusters_dict = {i: [] for i in range(5)}  # Creating a dictionary for clusters


'''Sorts each user to their group based on cluster'''
for i in range(len(y_preds)):
    clusters_dict[y_preds[i]].append(headers[i])


'''Print grouped users'''
for cluster_id, users in clusters_dict.items():
    print(f"Group {cluster_id + 1}: {users}")


plt.show()

