import pandas as pd
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.manifold import TSNE

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_data(file_path):
    """Load the processed CSV data."""
    return pd.read_csv(file_path)


def get_embeddings(texts):
    """Get embeddings for the given texts using OpenAI API."""
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings"):
        response = client.embeddings.create(model="text-embedding-3-large", input=text)
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)


def save_embeddings(embeddings, file_path):
    """Save the embeddings to a numpy file."""
    np.save(file_path, embeddings)


def perform_tsne(embeddings):
    """Perform t-SNE dimensionality reduction."""
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(embeddings)


def save_tsne_results(tsne_results, file_path):
    """Save the t-SNE results to a numpy file."""
    np.save(file_path, tsne_results)


def main():
    # Define file paths
    input_file = os.path.join("data", "processed", "clean_leadership_constructs.csv")
    embeddings_output_file = os.path.join("data", "processed", "item_embeddings.npy")
    tsne_output_file = os.path.join("data", "processed", "tsne_results.npy")

    # Load data
    df = load_data(input_file)
    print(f"Loaded {len(df)} items from {input_file}")

    # Get embeddings
    embeddings = get_embeddings(df["Item"].tolist())

    # Save embeddings
    save_embeddings(embeddings, embeddings_output_file)

    print(f"Embeddings saved to {embeddings_output_file}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Perform t-SNE
    tsne_results = perform_tsne(embeddings)

    # Save t-SNE results
    save_tsne_results(tsne_results, tsne_output_file)

    print(f"t-SNE results saved to {tsne_output_file}")
    print(f"t-SNE results shape: {tsne_results.shape}")


if __name__ == "__main__":
    main()
