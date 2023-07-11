import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# load data from csv file to a Pandas DataFrame
c_data = pd.read_csv('E:\CustomerSegmentation\Mall_Customers.csv')

# show first 5 rows
c_data.head()

# show number of rows and columns
c_data.shape

# getting some informations about the dataset
c_data.info()

# check for missing values
c_data.isnull().sum()

X = c_data.iloc[:,[3,4]].values
print(X)

# find wcss value for different number of clusters

wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)

# plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title('Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print(Y)

# Plot all the clusters
plt.figure(figsize=(8, 8))
for cluster_label in range(5):
    plt.scatter(X[Y==cluster_label, 0], X[Y==cluster_label, 1], s=50, label='Cluster {}'.format(cluster_label + 1))

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='pink', label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()