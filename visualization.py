import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px


def plot_clusters(df_normalized, dataset_name):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_normalized.drop(columns=['Cluster_KMeans']))

    plt.figure(figsize=(6,4))
    sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1], hue=df_normalized['Cluster_KMeans'], palette='viridis', s=50)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title(f"Clusters - {dataset_name}")
    plt.show()

def plot_bisect(df_scaled, labels, dataset_name):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_scaled.drop(columns=['Cluster_KMeans']))

    plt.figure(figsize=(6,4))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette='viridis', s=50)
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
 
def plot_original_iris ():
    df = px.data.iris()
    features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color="species"
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()


def plot_original_wine (df):
    features = df.select_dtypes(include='number').columns

    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color="class"
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()