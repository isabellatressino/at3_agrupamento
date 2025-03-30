from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

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

def kmeans(df_scaled, k):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    return kmeans.fit_predict(df_scaled)

def hierarchical_clustering(df_scaled, method, k):
    linked = linkage(df_scaled, method=method)
    return fcluster(linked, t=k, criterion='maxclust')

def silhouette_scores(df_iris_normalized, df_wine_normalized):
    silhouette_iris_kmeans = silhouette_score(df_iris_normalized.drop(columns=['Cluster_KMeans', 'Cluster_Hierarchical']), df_iris_normalized['Cluster_KMeans'])
    silhouette_wine_kmeans = silhouette_score(df_wine_normalized.drop(columns=['Cluster_KMeans', 'Cluster_Hierarchical']), df_wine_normalized['Cluster_KMeans'])
    silhouette_iris_hierarchical = silhouette_score(df_iris_normalized.drop(columns=['Cluster_KMeans', 'Cluster_Hierarchical']), df_iris_normalized['Cluster_Hierarchical'])
    silhouette_wine_hierarchical = silhouette_score(df_wine_normalized.drop(columns=['Cluster_KMeans', 'Cluster_Hierarchical']), df_wine_normalized['Cluster_Hierarchical'])

    print(f"Silhouette Score - Iris (KMeans): {silhouette_iris_kmeans}")
    print(f"Silhouette Score - Wine (KMeans): {silhouette_wine_kmeans}")
    print(f"Silhouette Score - Iris (Hierarchical): {silhouette_iris_hierarchical}")
    print(f"Silhouette Score - Wine (Hierarchical): {silhouette_wine_hierarchical}")
