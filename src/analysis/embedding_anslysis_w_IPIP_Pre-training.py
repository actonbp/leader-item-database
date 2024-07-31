import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers.evaluation import TripletEvaluator
import random
from umap.umap_ import UMAP
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

# Load IPIP data
def load_ipip_data(file_path):
    return pd.read_excel(file_path)

# Prepare IPIP data
def prepare_ipip_data(df):
    constructs = list(df.label.unique())
    df_train, df_test = train_test_split(df, test_size=0.10, random_state=999)
    
    constructs_dict = {}
    construct_overlap = {}
    for _, row in df_test.iterrows():
        construct = row["label"]
        item = row["text"].lower()
        if construct not in constructs_dict:
            constructs_dict[construct] = []
        constructs_dict[construct].append(item)
        if item not in construct_overlap:
            construct_overlap[item] = []
        construct_overlap[item].append(construct)
    
    return df_train, constructs, constructs_dict, construct_overlap

# Create IPIP training examples
def create_ipip_training_examples(df_train, constructs):
    train_examples = []
    for _, row in df_train.iterrows():
        text = row["text"].lower()
        construct_label = constructs.index(row["label"])
        example = InputExample(texts=[text], label=construct_label)
        train_examples.append(example)
    return train_examples

# Create IPIP evaluation triplets
def create_ipip_evaluation_triplets(constructs_dict, construct_overlap):
    anchors, positives, negatives = [], [], []
    for construct in constructs_dict:
        for item in constructs_dict[construct]:
            other_constructs = list(set(constructs_dict.keys()) - set(construct_overlap[item]))
            other_related_items = [i for i in constructs_dict[construct] if i != item]
            if other_constructs and other_related_items:
                random_construct = random.choice(other_constructs)
                negative = random.choice(constructs_dict[random_construct])
                positive = random.choice(other_related_items)
                anchors.append(item)
                positives.append(positive)
                negatives.append(negative)
    return anchors, positives, negatives

# Train model
def train_model(model, train_examples, anchors, positives, negatives):
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    train_loss = losses.BatchSemiHardTripletLoss(model)
    evaluator = TripletEvaluator(anchors, positives, negatives)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              epochs=100, 
              evaluator=evaluator, 
              warmup_steps=100, 
              evaluation_steps=1300, 
              output_path="construct_model")
    return model

# Load leadership data
def load_leadership_data(file_path):
    df = pd.read_csv(file_path)
    return df['Item'], df['Construct'].tolist()

# Train IPIP model
def train_ipip_model(ipip_file_path):
    # Load IPIP data
    ipip_df = load_ipip_data(ipip_file_path)
    df_train, constructs, constructs_dict, construct_overlap = prepare_ipip_data(ipip_df)
    
    # Create training examples
    train_examples = create_ipip_training_examples(df_train, constructs)
    
    # Create evaluation triplets
    anchors, positives, negatives = create_ipip_evaluation_triplets(constructs_dict, construct_overlap)
    
    # Load the model
    model = SentenceTransformer("construct_model")
    
    # Train the model
    model = train_model(model, train_examples, anchors, positives, negatives)
    
    return model

# Visualize IPIP clusters
def visualize_ipip_clusters(ipip_file_path, model):
    # Load IPIP data
    ipip_df = load_ipip_data(ipip_file_path)
    ipip_items = ipip_df['text'].tolist()
    ipip_constructs = ipip_df['label'].tolist()

    # Generate embeddings for IPIP items
    ipip_embeddings = model.encode(ipip_items, normalize_embeddings=True)

    # Perform dimensionality reduction
    reducer = UMAP(min_dist=0.0001, n_neighbors=4)
    ipip_coordinates = reducer.fit_transform(ipip_embeddings)

    # Map IPIP constructs to numerical values
    unique_constructs = list(set(ipip_constructs))
    construct_to_num = {construct: i for i, construct in enumerate(unique_constructs)}
    construct_nums = [construct_to_num[construct] for construct in ipip_constructs]

    # Plot the results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(ipip_coordinates[:, 0], ipip_coordinates[:, 1], c=construct_nums, cmap="tab20")
    plt.legend(handles=scatter.legend_elements()[0], labels=unique_constructs, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.title("IPIP Constructs Embedding Visualization")
    plt.savefig("results/ipip_constructs_visualization.png")
    plt.close()

# Main execution
if __name__ == "__main__":
    # Train the model with IPIP data
    ipip_model = train_ipip_model("data/processed/ipip_data.xlsx")

    # Load leadership items
    leadership_items, leadership_scales = load_leadership_data("data/processed/clean_leadership_constructs.csv")

    # Generate embeddings for leadership items
    leadership_embeddings = ipip_model.encode(leadership_items, normalize_embeddings=True)

    # Perform dimensionality reduction
    reducer = UMAP(min_dist=0.0001, n_neighbors=4)
    coordinates = reducer.fit_transform(leadership_embeddings)

    # Map leadership scales to numerical values
    unique_scales = list(set(leadership_scales))
    scale_to_num = {scale: i for i, scale in enumerate(unique_scales)}
    scale_nums = [scale_to_num[scale] for scale in leadership_scales]

    # Plot the results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=scale_nums, cmap="tab20")
    plt.legend(handles=scatter.legend_elements()[0], labels=unique_scales, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.title("Leadership Constructs Embedding Visualization")
    plt.savefig("results/leadership_constructs_visualization.png")
    plt.close()

    # Find optimal number of clusters
    silhouette_scores = []
    for n_clusters in range(2, 20):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates)
        score = metrics.silhouette_score(coordinates, cluster_labels)
        silhouette_scores.append(score)
        print(f"Clusters: {n_clusters}, Silhouette Score: {score}")

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"Optimal number of clusters: {optimal_clusters}")

    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(coordinates)

    # Save results
    results_df = pd.DataFrame({
        "Item": leadership_items,
        "Scale": leadership_scales,
        "Cluster": cluster_labels,
        "UMAP_1": coordinates[:, 0],
        "UMAP_2": coordinates[:, 1]
    })
    results_df.to_csv("results/leadership_constructs_analysis.csv", index=False)

    print("Analysis complete. Results saved in the 'results' directory.")

    # Visualize IPIP clusters
    visualize_ipip_clusters("data/processed/ipip_data.xlsx", ipip_model)