import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_wine
from scipy.cluster.hierarchy import dendrogram, linkage

# Base de dados
iris_data = load_iris()
wine_data = load_wine()

# Conversão em DataFrame
df_iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df_wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Verificar se precisamos fazer a normalização
print("Iris Não-normalizado:")
print(df_iris.head())

print("\nWine Não-normalizado:")
print(df_wine.head())

# Normalização
scaler = StandardScaler()
matriz_iris_scaled = scaler.fit_transform(df_iris)
matriz_wine_scaled = scaler.fit_transform(df_wine)

# Converter a matriz normalizada de volta para DataFrame
df_iris_normalized = pd.DataFrame(matriz_iris_scaled, columns=df_iris.columns)
df_wine_normalized = pd.DataFrame(matriz_wine_scaled, columns=df_wine.columns)  

# Exibir normalizados
print("\nIris Normalizado:")
print(df_iris_normalized.head())

print("\nWine Normalizado:")
print(df_wine_normalized.head())

# Função do método do cotovelo
def elbow_method(df_scaled, dataset_name):
    distance = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(df_scaled)
        distance.append(kmeans.inertia_)

    plt.figure(figsize=(6,4))
    plt.plot(k_range, distance)
    plt.xlabel('Número de Clusters')
    plt.ylabel('Distância Total (Inércia)')
    plt.title(f'Método do Cotovelo - {dataset_name}')
    plt.show()

# Aplicar método do cotovelo
elbow_method(df_iris_normalized, "Iris")
elbow_method(df_wine_normalized, "Wine")

# Defininindo o número de clusters com base no gráfico do cotovelo
k_iris = 3  
k_wine = 3 

# Aplicar K-Means
kmeans_iris = KMeans(n_clusters=k_iris, random_state=0, n_init=10)
kmeans_wine = KMeans(n_clusters=k_wine, random_state=0, n_init=10)

# Adicionando a coluna do cluster
df_iris_normalized['Cluster'] = kmeans_iris.fit_predict(df_iris_normalized)
df_wine_normalized['Cluster'] = kmeans_wine.fit_predict(df_wine_normalized)

# Exibir as primeiras linhas com os clusters
print("\nIris com Clusters:")
print(df_iris_normalized.head())

print("\nWine com Clusters:")
print(df_wine_normalized.head())

# Função que realiza a PCA e plota o grafico
def plot_clusters(df_normalized, dataset_name):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_normalized.drop(columns=['Cluster']))

    plt.figure(figsize=(6,4))
    sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1], hue=df_normalized['Cluster'], palette='viridis', s=50)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title(f"Clusters - {dataset_name}")
    plt.show()

plot_clusters(df_iris_normalized, "Iris")
plot_clusters(df_wine_normalized, "Wine")

# Função que faz os algoritmos linkage e plota o dedograma
def plot_dendrogram(df_normalized, dataset_name, method):
    plt.figure(figsize=(6,4))
    linked = linkage(df_normalized.drop(columns=['Cluster']), method=method)
    dendrogram(linked)
    plt.xlabel("Amostras")
    plt.ylabel("Distância")
    plt.title(f"Dendrograma - {dataset_name} ({method})")
    plt.show()

plot_dendrogram(df_iris_normalized,'Iris','single')
plot_dendrogram(df_wine_normalized,'Wine','single')

plot_dendrogram(df_iris_normalized,'Iris','complete')
plot_dendrogram(df_wine_normalized,'Wine','complete')

plot_dendrogram(df_iris_normalized,'Iris','average')
plot_dendrogram(df_wine_normalized,'Wine','average')

plot_dendrogram(df_iris_normalized,'Iris','ward')
plot_dendrogram(df_wine_normalized,'Wine','ward')