import pandas as pd 
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture 
import matplotlib.pyplot as plt 
 
np.random.seed(0)  
num_samples = 100 
num_features = 2 
 
data = np.random.rand(num_samples, num_features) 
df = pd.DataFrame(data, columns=['X', 'Y']) 
df.to_csv('test_data.csv', index=False) 
 
print("CSV file 'test_data.csv' has been generated successfully.") 
 
data = pd.read_csv('test_data.csv') 
 
X = data.values 
k = 3 
 
kmeans = KMeans(n_clusters=k) 
kmeans.fit(X) 
kmeans_labels = kmeans.labels_ 
kmeans_centers = kmeans.cluster_centers_ 
 
em = GaussianMixture(n_components=k) 
em.fit(X) 
em_labels = em.predict(X) 
em_centers = em.means_ 
 
print("K-means labels:") 
print(kmeans_labels) 
print("EM labels:") 
print(em_labels) 
 
plt.figure(figsize=(12, 5)) 
 
plt.subplot(1, 2, 1) 
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis') 
plt.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], marker='*', 
s=300, c='r') 
plt.title('K-means Clustering') 
 
plt.subplot(1, 2, 2) 
plt.scatter(X[:, 0], X[:, 1], c=em_labels, cmap='viridis') 
plt.scatter(em_centers[:, 0], em_centers[:, 1], marker='*', s=300, 
c='r') 
plt.title('EM Clustering') 
 
plt.show() 
