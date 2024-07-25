import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(csv_file, npy_file):
    """Load the processed CSV data and embeddings."""
    df = pd.read_csv(csv_file)
    embeddings = np.load(npy_file)
    return df, embeddings


def plot_nomological_network(embeddings, constructs):
    """Create a plot of the nomological network."""
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plot_df = pd.DataFrame(
        {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "Construct": constructs,
        }
    )

    plt.figure(figsize=(20, 16))
    sns.set(style="whitegrid")

    palette = sns.color_palette("husl", n_colors=len(plot_df["Construct"].unique()))

    sns.scatterplot(
        data=plot_df, x="x", y="y", hue="Construct", palette=palette, s=100, alpha=0.7
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.title("Nomological Network of Leadership Constructs", fontsize=20)
    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)

    plt.tight_layout()
    return plt


def save_plot(plt, file_path):
    """Save the plot to a file."""
    plt.savefig(file_path, dpi=300, bbox_inches="tight")


def main():
    csv_file = os.path.join("data", "processed", "clean_leadership_constructs.csv")
    npy_file = os.path.join("data", "processed", "item_embeddings.npy")
    output_file = os.path.join("results", "nomological_network_plot.png")

    df, embeddings = load_data(csv_file, npy_file)
    plt = plot_nomological_network(embeddings, df["Construct"])
    save_plot(plt, output_file)

    print(f"Nomological network plot saved to {output_file}")


if __name__ == "__main__":
    main()
