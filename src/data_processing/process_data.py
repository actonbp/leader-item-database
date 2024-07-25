import pandas as pd
import os


def load_data(file_path):
    """Load the CSV data."""
    return pd.read_csv(file_path)


def clean_construct_name(scale):
    """Clean the construct name to focus on the conceptual construct."""
    construct_mapping = {
        "Ethical Leadership": "Ethical Leadership",
        "LMX7": "LMX",
        "LMXMDM": "LMX",
        "Servant leadership": "Servant Leadership",
        "Authentic Leadership": "Authentic Leadership",
        "Empowering Leadership Questionnaire ELQ": "Empowering Leadership",
        "MLQ": "Transformational Leadership",
        "Inclusive Leadership": "Inclusive Leadership",
        "Paternalistic Leadership": "Paternalistic Leadership",
        "SelfLeadership Questionnaire": "Self Leadership",
        "Paradoxical Leadership": "Paradoxical Leadership",
        "Public Leadership": "Public Leadership",
    }
    return construct_mapping.get(scale, scale)


def filter_leadership_items(df):
    """Filter the dataframe to include only specified leadership constructs."""
    leadership_constructs = [
        "Ethical Leadership",
        "LMX",
        "Servant Leadership",
        "Authentic Leadership",
        "Empowering Leadership",
        "Transformational Leadership",
        "Inclusive Leadership",
        "Paternalistic Leadership",
        "Self Leadership",
        "Paradoxical Leadership",
        "Public Leadership",
    ]

    df["Construct"] = df["Scale"].apply(clean_construct_name)

    mlq_items = [2, 6, 8, 9, 10, 13, 14, 15, 18, 19, 21, 23, 25, 26, 29, 30, 31, 32, 34]
    mlq_mask = (df["Scale"] == "MLQ") & (df["Item_Number"].isin(mlq_items))
    df.loc[mlq_mask, "Construct"] = "Transformational Leadership"

    servant_items = df[df["Construct"] == "Servant Leadership"]
    non_servant_items = df[df["Construct"] != "Servant Leadership"]
    selected_servant_items = servant_items.sample(n=75, random_state=42)

    df_filtered = pd.concat([non_servant_items, selected_servant_items])
    return df_filtered[df_filtered["Construct"].isin(leadership_constructs)]


def create_clean_dataset(df):
    """Create a clean dataset with construct labels and original items."""
    clean_df = df[["Item", "Construct"]].copy()
    clean_df = clean_df.reset_index(drop=True)
    clean_df.index.name = "Item_ID"
    return clean_df


def save_processed_data(df, file_path):
    """Save the processed dataframe to a CSV file."""
    df.to_csv(file_path)


def main():
    # Define file paths
    input_file = os.path.join("data", "raw", "Cleaned_Item_Database.csv")
    output_file = os.path.join("data", "processed", "clean_leadership_constructs.csv")

    # Load data
    df = load_data(input_file)
    print(f"Loaded {len(df)} items from {input_file}")

    # Process data
    df_leadership = filter_leadership_items(df)
    clean_df = create_clean_dataset(df_leadership)

    # Save processed data
    save_processed_data(clean_df, output_file)

    print(f"Clean leadership constructs data saved to {output_file}")
    print(f"Number of leadership items: {len(clean_df)}")
    print(f"Unique constructs: {clean_df['Construct'].unique()}")
    print("\nSample of processed data:")
    print(clean_df.head())
    print("\nItem counts per construct:")
    print(clean_df["Construct"].value_counts())
    print("\nExample items:")
    for _, row in clean_df.sample(n=5).iterrows():
        print(f"{row['Construct']}: {row['Item']}")


if __name__ == "__main__":
    main()
