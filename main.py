from preprocessing import normalize_data
from cluster_utils import elbow_method, elbow_bisecting_kmeans, elbow_silhouette_method, kmeans, hierarchical_clustering, test_silhouette_scores, bisect_kmeans
from visualization import plot_clusters, plot_dendrogram, plot_bisect, plot_original_iris, plot_original_wine
from sklearn.datasets import load_iris, load_wine

import pandas as pd

def main():

    iris_data = load_iris()
    wine_data = load_wine()

    df_iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df_wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

    plot_original_iris()

    df_wine['class'] = wine_data.target
    plot_original_wine(df_wine)

    print("Iris Não-normalizado:")
    print(df_iris.head())

    print("\nWine Não-normalizado:")
    print(df_wine.head())

    # Normalização
    df_iris_normalized = normalize_data(df_iris)
    df_wine_normalized = normalize_data(df_wine)

    print("\nIris Normalizado:")
    print(df_iris_normalized.head())

    print("\nWine Normalizado:")
    print(df_wine_normalized.head())

    elbow_method(df_iris, "Iris")
    elbow_method(df_wine_normalized, "Wine")
    elbow_bisecting_kmeans(df_iris, "Iris")
    elbow_bisecting_kmeans(df_wine_normalized, "Wine")
    elbow_silhouette_method(df_iris, "Iris")
    elbow_silhouette_method(df_wine_normalized, "Wine")


    df_iris['Cluster_KMeans'] = kmeans(df_iris, 4)
    df_iris_normalized['Cluster_KMeans'] = df_iris['Cluster_KMeans']
    df_wine_normalized['Cluster_KMeans'] = kmeans(df_wine_normalized, 3)

    plot_clusters(df_iris_normalized, "Iris")
    plot_clusters(df_wine_normalized, "Wine")


    bisect_iris = bisect_kmeans(df_iris, 3)
    bisect_wine = bisect_kmeans(df_wine_normalized, 3)
    plot_bisect(df_iris_normalized, bisect_iris, "Bisect Iris")
    plot_bisect(df_wine_normalized, bisect_wine, "Bisect Wine")

    plot_dendrogram(df_iris,'Iris','single')
    plot_dendrogram(df_wine_normalized,'Wine','single')

    plot_dendrogram(df_iris,'Iris','complete')
    plot_dendrogram(df_wine_normalized,'Wine','complete')

    plot_dendrogram(df_iris,'Iris','average')
    plot_dendrogram(df_wine_normalized,'Wine','average')

    plot_dendrogram(df_iris,'Iris','ward')
    plot_dendrogram(df_wine_normalized,'Wine','ward')

    df_iris['Cluster_Hierarchical'] = hierarchical_clustering(df_iris, 'ward', 2)
    df_wine_normalized['Cluster_Hierarchical'] = hierarchical_clustering(df_wine_normalized, 'ward', 3)

    score_irirs_kmeans_raw,score_iris_b_kmeans_raw, score_irirs_linkage_raw = test_silhouette_scores(df_iris)
    print("="*12)   
    print("Silhouette scores - Iris S/ Normalização")
    print("\nKmeans")
    print(score_irirs_kmeans_raw)
    print("\nB-Kmeans")
    print(score_iris_b_kmeans_raw)
    print("\nLinkage Family")
    print(score_irirs_linkage_raw)
    score_irirs_kmeans,score_iris_b_kmeans, score_irirs_linkage = test_silhouette_scores(df_iris_normalized)
    print("\nSilhouette scores - Iris Normalizado")
    print("\nKmeans")
    print(score_irirs_kmeans)
    print("\nB-Kmeans")
    print(score_iris_b_kmeans)
    print("\nLinkage Family")
    print(score_irirs_linkage)

    print("="*12)
    score_wine_kmeans_raw,score_wine_b_kmeans_raw, score_wine_linkage_raw = test_silhouette_scores(df_wine)
    print("\nSilhouette scores - Wine S/ Normalização")
    print("\nKmeans")
    print(score_wine_kmeans_raw)
    print("\nB-Kmeans")
    print(score_wine_b_kmeans_raw)
    print("\nLinkage Family")
    print(score_wine_linkage_raw)

    score_wine_kmeans,score_wine_b_kmeans, score_wine_linkage = test_silhouette_scores(df_wine_normalized)
    print("\nSilhouette scores - Wine Normalizado")
    print("\nKmeans")
    print(score_wine_kmeans)
    print("\nB-Kmeans")
    print(score_wine_b_kmeans)
    print("\nLinkage Family")
    print(score_wine_linkage)

if __name__ == "__main__":
    main()
