from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from traitlets import Bunch

# Base de dados
iris_data = load_iris()
wine_data = load_wine()

# DataFrames
df_iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df_wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
# Normalização
scaler = StandardScaler()
df_iris_scaled = scaler.fit_transform(df_iris)
df_wine_scaled = scaler.fit_transform(df_wine)

## use elbow method to define wich is the best k 
# apply K-means in a range of clusters and plot the graph
df = pd.DataFrame(df_iris_scaled, columns= iris_data.feature_names)
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_iris) # finding centroid
    print(kmeans.labels_)
    df_iris["clusters"] = kmeans.labels_
    print(df_iris)
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

