import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from datetime import datetime
import os


def load_data():
    df = pd.read_csv("data/processed/clean_leadership_constructs.csv")
    tsne_results = np.load("data/processed/tsne_results.npy")
    return df, tsne_results


def find_optimal_clusters(data, max_clusters):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, cluster_labels))
    return silhouette_scores.index(max(silhouette_scores)) + 2


def perform_clustering(tsne_results, optimal_clusters):
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    return kmeans.fit_predict(tsne_results)


def get_example_items(cluster_items, n=3):
    return cluster_items["Item"].sample(n=min(n, len(cluster_items))).tolist()


def visualize_clusters(df, tsne_results, optimal_clusters):
    plt.figure(figsize=(20, 16))
    scatter = plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=df["Cluster"],
        cmap="viridis",
        alpha=0.7,
    )
    plt.colorbar(scatter)

    for cluster in range(optimal_clusters):
        cluster_items = df[df["Cluster"] == cluster]
        if not cluster_items.empty:
            cluster_center = tsne_results[df["Cluster"] == cluster].mean(axis=0)
            example_items = get_example_items(cluster_items)
            plt.annotate(
                f"Cluster {cluster}:\n" + "\n".join(example_items),
                xy=cluster_center,
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                fontsize=8,
                wrap=True,
            )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.text(
        0.95,
        0.05,
        f"Generated on: {current_time}",
        fontsize=10,
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
    )

    plt.title(f"Nomological Network with {optimal_clusters} Clusters", fontsize=20)
    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(
        "results/nomological_network_clusters.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_nomological_network(df, tsne_results):
    plt.figure(figsize=(20, 16))
    unique_constructs = df['Construct'].unique()
    color_map = plt.cm.get_cmap('tab20')
    colors = {construct: color_map(i/len(unique_constructs)) for i, construct in enumerate(unique_constructs)}

    for construct in unique_constructs:
        mask = df['Construct'] == construct
        plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=[colors[construct]],
            label=construct,
            alpha=0.7,
            s=50
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title("Nomological Network of Leadership Constructs", fontsize=20)
    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(
        "results/nomological_network_by_construct.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def analyze_clusters(df):
    cluster_sizes = df["Cluster"].value_counts().sort_values(ascending=False)
    print("\nCluster sizes:")
    print(cluster_sizes)

    def get_common_words(items, n=10):
        words = " ".join(items).split()
        word_freq = pd.Series(words).value_counts()
        return word_freq.head(n)

    for cluster in cluster_sizes.index:
        cluster_items = df[df["Cluster"] == cluster]
        print(f"\n\nCluster {cluster} (Size: {len(cluster_items)}):")
        print("\nConstruct distribution:")
        print(cluster_items["Construct"].value_counts())
        print("\nMost common words:")
        print(get_common_words(cluster_items["Processed_Item"]))
        print("\nSample items:")
        print(
            cluster_items["Processed_Item"]
            .sample(n=min(5, len(cluster_items)))
            .to_string(index=False)
        )


def main():
    df, tsne_results = load_data()
    print("Data loaded successfully.")

    # Create nomological network plot by construct
    plot_nomological_network(df, tsne_results)
    print("Nomological network plot by construct saved.")

    max_clusters = 20
    optimal_clusters = find_optimal_clusters(tsne_results, max_clusters)
    print(f"Optimal number of clusters: {optimal_clusters}")

    try:
        cluster_labels = perform_clustering(tsne_results, optimal_clusters)
        df["Cluster"] = cluster_labels
        print("Clustering completed successfully.")

        visualize_clusters(df, tsne_results, optimal_clusters)
        print("Cluster visualization saved.")

        analyze_clusters(df)
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Skipping clustering and visualization.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()