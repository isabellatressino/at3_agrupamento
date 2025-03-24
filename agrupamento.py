from sklearn.datasets import load_iris, load_wine
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# K-means
kmeans = KMeans(n_clusters=3).fit(df_iris_scaled)
kmeans = KMeans(n_clusters=3).fit(df_wine)