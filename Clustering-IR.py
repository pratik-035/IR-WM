
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans, AgglomerativeClustering 
from sklearn.metrics import silhouette_score 
from scipy.cluster.hierarchy import dendrogram, linkage 

# Load a subset of 20 Newsgroups dataset (4 categories for simplicity) 
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

newsgroups = fetch_20newsgroups(subset='all', categories=categories, 
remove=('headers', 'footers', 'quotes')) 

# Convert text data into numerical format using TF-IDF vectorization 
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000) 
X = vectorizer.fit_transform(newsgroups.data) 


# K-MEANS CLUSTERING 

num_clusters = len(categories) 

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) 
kmeans_labels = kmeans.fit_predict(X) 
silhouette_avg = silhouette_score(X, kmeans_labels) 

# Display K-Means results 
print("\n=== K-Means Clustering Results ===") 
print(f"Number of Clusters: {num_clusters}") 
print(f"Silhouette Score: {silhouette_avg:.4f}") 

# HIERARCHICAL CLUSTERING 
 
hierarchical = AgglomerativeClustering(n_clusters=num_clusters, 
metric='euclidean', linkage='ward') 
 
hierarchical_labels = hierarchical.fit_predict(X.toarray()) 
 
 
 
# Plot dendrogram 
plt.figure(figsize=(10, 5)) 
plt.title("Hierarchical Clustering Dendrogram") 
Z = linkage(X.toarray(), method='ward') 
dendrogram(Z, truncate_mode="level", p=10) 
plt.show() 
 
 
# Display Hierarchical clustering results 
 
print("\n=== Hierarchical Clustering Results ===") 
print(f"Number of Clusters: {num_clusters}") 
