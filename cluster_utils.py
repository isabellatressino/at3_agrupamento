from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, BisectingKMeans

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


def test_silhouette_scores(df_normalized):
    methods = ['single', 'complete', 'average', 'ward']
    cluster_range = range(2, 11) 
    
    results_kmeans = []
    results_linkage = []

    for n_clusters in cluster_range:
        # K-Means
        kmeans_labels = kmeans(df_normalized, n_clusters)
        kmeans_score = silhouette_score(df_normalized, kmeans_labels)
        results_kmeans.append([n_clusters, round(kmeans_score, 4)])

        # Linkage
        for method in methods:
            hierarchical_labels = hierarchical_clustering(df_normalized, method, n_clusters)
            hierarchical_score = silhouette_score(df_normalized, hierarchical_labels)
            results_linkage.append([method, n_clusters, round(hierarchical_score, 4)])

    results_kmeans_df = pd.DataFrame(results_kmeans,columns=['N_Clusters', 'Silhouette Score'])
    results_linkage_df = pd.DataFrame(results_linkage, columns=['Method', 'N_Clusters', 'Silhouette Score'])
    
    return results_kmeans_df, results_linkage_df

def bisect_kmeans(df_scaled, k):
    bisect = BisectingKMeans(n_clusters=k, random_state=0).fit(df_scaled)
    return  bisect.labels_