import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed/clean_leadership_constructs.csv")
tsne_results = np.load("data/processed/tsne_results.npy")


def find_optimal_clusters(data, max_clusters):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, cluster_labels))
    return silhouette_scores.index(max(silhouette_scores)) + 2


# Find optimal number of clusters
max_clusters = 20
optimal_clusters = find_optimal_clusters(tsne_results, max_clusters)
print(f"Optimal number of clusters: {optimal_clusters}")

# Perform clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(tsne_results)
df["Cluster"] = cluster_labels

# Visualize clusters
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap="viridis"
)
plt.colorbar(scatter)
plt.title(f"Nomological Network with {optimal_clusters} Clusters")
plt.savefig("results/nomological_network_clusters.png")
plt.close()

# Analyze clusters
cluster_sizes = df["Cluster"].value_counts().sort_values(ascending=False)
print("\nCluster sizes:")
print(cluster_sizes)


# Function to get most common words in a cluster
def get_common_words(items, n=10):
    words = " ".join(items).lower().split()
    word_freq = pd.Series(words).value_counts()
    return word_freq.head(n)


# Analyze and print details for each cluster
for cluster in cluster_sizes.index:
    cluster_items = df[df["Cluster"] == cluster]
    print(f"\n\nCluster {cluster} (Size: {len(cluster_items)}):")
    print("\nConstruct distribution:")
    print(cluster_items["Construct"].value_counts())
    print("\nMost common words:")
    print(get_common_words(cluster_items["Item"]))
    print("\nSample items:")
    print(
        cluster_items["Item"]
        .sample(n=min(5, len(cluster_items)))
        .to_string(index=False)
    )

# Save detailed cluster information
with open("results/cluster_analysis_details.txt", "w") as f:
    for cluster in cluster_sizes.index:
        cluster_items = df[df["Cluster"] == cluster]
        f.write(f"\n\nCluster {cluster} (Size: {len(cluster_items)}):\n")
        f.write("\nConstruct distribution:\n")
        f.write(cluster_items["Construct"].value_counts().to_string())
        f.write("\n\nMost common words:\n")
        f.write(get_common_words(cluster_items["Item"]).to_string())
        f.write("\n\nAll items in this cluster:\n")
        for _, row in cluster_items.iterrows():
            f.write(f"{row['Construct']}: {row['Item']}\n")

print(
    "\nDetailed cluster analysis has been saved to 'results/cluster_analysis_details.txt'"
)
