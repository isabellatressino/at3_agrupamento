from preprocessing import normalize_data
from cluster_utils import elbow_method, kmeans, hierarchical_clustering, silhouette_scores
from visualization import plot_clusters, plot_dendrogram
from sklearn.datasets import load_iris, load_wine

import pandas as pd

def main():

    # Base de dados
    iris_data = load_iris()
    wine_data = load_wine()

    # Conversão para DataFrame
    df_iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df_wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

    # # Verificar se precisamos fazer a normalização
    print("Iris Não-normalizado:")
    print(df_iris.head())

    print("\nWine Não-normalizado:")
    print(df_wine.head())

    # Normalização
    df_iris_normalized = normalize_data(df_iris)
    df_wine_normalized = normalize_data(df_wine)

    # Exibir normalizados
    print("\nIris Normalizado:")
    print(df_iris_normalized.head())

    print("\nWine Normalizado:")
    print(df_wine_normalized.head())

    # Método do cotovelo
    elbow_method(df_iris_normalized, "Iris")
    elbow_method(df_wine_normalized, "Wine")

    # KMeans 
    df_iris_normalized['Cluster_KMeans'] = kmeans(df_iris_normalized, 3)
    df_wine_normalized['Cluster_KMeans'] = kmeans(df_wine_normalized, 3)

    # Visualização dos clusters
    plot_clusters(df_iris_normalized, "Iris")
    plot_clusters(df_wine_normalized, "Wine")

    # Dendrograma
    plot_dendrogram(df_iris_normalized,'Iris','single')
    plot_dendrogram(df_wine_normalized,'Wine','single')

    plot_dendrogram(df_iris_normalized,'Iris','complete')
    plot_dendrogram(df_wine_normalized,'Wine','complete')

    plot_dendrogram(df_iris_normalized,'Iris','average')
    plot_dendrogram(df_wine_normalized,'Wine','average')

    plot_dendrogram(df_iris_normalized,'Iris','ward')
    plot_dendrogram(df_wine_normalized,'Wine','ward')

    # Hierarchical clustering
    df_iris_normalized['Cluster_Hierarchical'] = hierarchical_clustering(df_iris_normalized, 'complete', 3)
    df_wine_normalized['Cluster_Hierarchical'] = hierarchical_clustering(df_wine_normalized, 'ward', 3)

    # Silhouette scores
    silhouette_scores(df_iris_normalized, df_wine_normalized)

if __name__ == "__main__":
    main()
