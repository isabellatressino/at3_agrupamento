import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_clusters(df_normalized, dataset_name):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_normalized.drop(columns=['Cluster_KMeans']))

    plt.figure(figsize=(6,4))
    sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1], hue=df_normalized['Cluster_KMeans'], palette='viridis', s=50)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title(f"Clusters - {dataset_name}")
    plt.show()

def plot_dendrogram(df_normalized, dataset_name, method):
    plt.figure(figsize=(6,4))
    linked = linkage(df_normalized.drop(columns=['Cluster_KMeans']), method=method)
    dendrogram(linked)
    plt.xlabel("Amostras")
    plt.ylabel("Distância")
    plt.title(f"Dendrograma - {dataset_name} ({method})")
    plt.show()
