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

def elbow_bisecting_kmeans(df,dataset_name, max_k=10, random_state=0):
    inertias = []
    ks = range(2, max_k + 1)

    for k in ks:
        model = BisectingKMeans(n_clusters=k, random_state=random_state)
        model.fit(df)
        inertias.append(model.inertia_)

    # Plot do cotovelo
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o')
    plt.title(f'Elbow - Bisecting KMeans - {dataset_name}')
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inércia (WSS)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def elbow_silhouette_method(df_scaled, dataset_name):
    silhouette_scores = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(df_scaled)
        labels = kmeans.labels_
        score = silhouette_score(df_scaled, labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(6, 4))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette - Método do Cotovelo (Elbow) - {dataset_name}')
    plt.grid(True)
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
    results_b_kmeans = []
    results_linkage = []

    for n_clusters in cluster_range:
        # K-Means
        kmeans_labels = kmeans(df_normalized, n_clusters)
        b_kmeans_labels = bisect_kmeans(df_normalized, n_clusters)
        kmeans_score = silhouette_score(df_normalized, kmeans_labels)
        b_kmeans_score= silhouette_score(df_normalized, b_kmeans_labels )
        results_kmeans.append([n_clusters, round(kmeans_score, 4)])
        results_b_kmeans.append([n_clusters, round(b_kmeans_score, 4)])

        # Linkage
        for method in methods:
            hierarchical_labels = hierarchical_clustering(df_normalized, method, n_clusters)
            hierarchical_score = silhouette_score(df_normalized, hierarchical_labels)
            results_linkage.append([method, n_clusters, round(hierarchical_score, 4)])

    results_kmeans_df = pd.DataFrame(results_kmeans,columns=['N_Clusters', 'Silhouette Score'])
    results_b_kmeans_df = pd.DataFrame(results_b_kmeans,columns=['N_Clusters', 'Silhouette Score'])
    results_linkage_df = pd.DataFrame(results_linkage, columns=['Method', 'N_Clusters', 'Silhouette Score'])
    
    return results_kmeans_df, results_b_kmeans_df, results_linkage_df

def bisect_kmeans(df_scaled, k):
    df_scaled = df_scaled.drop(columns=['Cluster_KMeans'])
    bisect = BisectingKMeans(n_clusters=k, random_state=0).fit(df_scaled)
    return  bisect.fit_predict(df_scaled)