from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carrega a base de dados Iris
data = load_iris()
df_iris = pd.DataFrame(data.data, columns=data.feature_names)

# Normaliza os dados
scaler = StandardScaler()
df_iris_scaled = scaler.fit_transform(df_iris)

# K-Means
kmeans = KMeans(n_clusters=3).fit(df_iris_scaled)

