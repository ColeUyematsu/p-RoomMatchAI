import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class RoommateMatcher:
    def __init__(self, data_path):
        """
        Initialize the RoommateMatcher with a path to the CSV file.
        
        Parameters:
        data_path (str): Path to CSV file containing personality data
        """
        self.data = pd.read_csv(data_path)
        self.names = self.data['NAME']
        self.X = self.data.drop('NAME', axis=1)
        self.processed_data = None
        self.knn_model = None
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare the data by scaling and normalizing"""
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Normalize the data
        self.processed_data = normalize(X_scaled)
        
        # Fit the KNN model
        self.knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')  # 6 because first match will be self
        self.knn_model.fit(self.processed_data)
        
    def find_matches(self, person_name, n_matches=5):
        """
        Find the top matches for a given person.
        
        Parameters:
        person_name (str): Name of the person to find matches for
        n_matches (int): Number of matches to return (default 5)
        
        Returns:
        list: List of tuples containing (name, similarity_score)
        """
        if person_name not in self.names.values:
            raise ValueError(f"Person '{person_name}' not found in dataset")
            
        # Get the index of the person
        person_idx = self.names[self.names == person_name].index[0]
        
        # Get the person's data
        person_data = self.processed_data[person_idx].reshape(1, -1)
        
        # Find k nearest neighbors
        distances, indices = self.knn_model.kneighbors(person_data)
        
        # Convert distances to similarity scores (1 / (1 + distance))
        similarity_scores = 1 / (1 + distances[0])
        
        # Create list of matches (excluding the person themselves)
        matches = [(self.names.iloc[idx], score) 
                  for idx, score in zip(indices[0][1:], similarity_scores[1:])]
        
        return matches[:n_matches]
    
    def visualize_matches(self, person_name, n_matches=5):
        """
        Create a PCA visualization of all individuals, highlighting the matches for a given person.
        
        Parameters:
        person_name (str): Name of the person to find matches for
        n_matches (int): Number of matches to highlight (default 5)
        """
        # Perform PCA
        pca = PCA(n_components=2)
        X_principal = pca.fit_transform(self.processed_data)
        
        # Get matches
        matches = self.find_matches(person_name, n_matches)
        match_names = [match[0] for match in matches]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot all points
        plt.scatter(X_principal[:, 0], X_principal[:, 1], 
                   c='gray', alpha=0.5, label='All individuals')
        
        # Plot person of interest
        person_idx = self.names[self.names == person_name].index[0]
        plt.scatter(X_principal[person_idx, 0], X_principal[person_idx, 1],
                   c='red', s=100, label='Selected person')
        
        # Plot matches
        match_indices = [self.names[self.names == name].index[0] for name in match_names]
        plt.scatter(X_principal[match_indices, 0], X_principal[match_indices, 1],
                   c='blue', s=100, label='Top matches')
        
        plt.title(f"PCA Visualization of Roommate Matches for {person_name}")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.legend()
        plt.show()

# Example usage
def main():
    # Initialize the matcher
    matcher = RoommateMatcher("test.csv")
    
    # Find matches for a specific person
    person_name = "Person_2"  # Replace with actual name
    matches = matcher.find_matches(person_name)
    
    # Print results
    print(f"\nTop matches for {person_name}:")
    for name, score in matches:
        print(f"{name}: Similarity score = {score:.3f}")
    
    # Visualize the matches
    matcher.visualize_matches(person_name)

if __name__ == "__main__":
    main()