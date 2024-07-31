import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(file_path):
    """Load the CSV data with error handling for encoding issues."""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin-1')

def remove_common_words(text):
    common_words = set(['my', 'manager', 'supervisor', 'leader', 'leadership', 'supervision', 'hisher', 'heshe', 'myself', 'leader', 'work', 'group', 'members',
                        'is', 'not', 'if', 'what', 'when', 'also', 'as', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                        'shall', 'should', 'may', 'might', 'must', 'can', 'could',
                        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
                        'to', 'for', 'of', 'with', 'by', 'from', 'up', 'down',
                        'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she',
                        'it', 'we', 'they', 'them', 'their', 'his', 'her', 'its',
                        'our', 'your', 'me', 'him', 'us'])
    words = text.split()
    return ' '.join([word for word in words if word.lower() not in common_words])

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove common words
    text = remove_common_words(text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def final_process_with_openai(text):
    """Use OpenAI API to refine the processed text."""
    prompt = f"These are simplified leadership items that are being fed into an embedding model, and have stop words and redundant words removed already. Please complete the final cleanup by removing words that do not make sense or clean up so the embedding model can make use of these words. Return only the original version of words or the improved version:\n\n{text}"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that refines leadership questionnaire items."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        temperature=0.5,
    )
    
    return response.choices[0].message.content.strip()

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

def filter_leadership_items(df, preprocessing_level):
    """Preprocess and clean construct names, and perform final processing based on the chosen level."""
    df["Construct"] = df["Scale"].apply(clean_construct_name)
    
    if preprocessing_level == "none":
        df["Processed_Item"] = df["Item"]
    elif preprocessing_level == "simple":
        df["Processed_Item"] = df["Item"].apply(preprocess_text)
    elif preprocessing_level == "advanced":
        df["Processed_Item"] = df["Item"].apply(preprocess_text)
        tqdm.pandas(desc="Final processing with OpenAI")
        df["Processed_Item"] = df["Processed_Item"].progress_apply(final_process_with_openai)
    
    return df

def create_clean_dataset(df):
    """Create a clean dataset with construct labels, original items, and processed items."""
    clean_df = df[["Item", "Processed_Item", "Construct"]].copy()
    clean_df = clean_df.reset_index(drop=True)
    clean_df.index.name = "Item_ID"
    return clean_df

def save_processed_data(df, file_path):
    """Save the processed dataframe to a CSV file."""
    df.to_csv(file_path)

def main():
    # Define file paths
    input_file = os.path.join("data", "raw", "Cleaned_Item_Database_v2.csv")
    output_file = os.path.join("data", "processed", "clean_leadership_constructs.csv")

    # Load data
    df = load_data(input_file)
    print(f"Loaded {len(df)} items from {input_file}")

    # Ask user for preprocessing level
    while True:
        preprocessing_level = input("Enter preprocessing level (none/simple/advanced): ").lower()
        if preprocessing_level in ["none", "simple", "advanced"]:
            break
        print("Invalid input. Please enter 'none', 'simple', or 'advanced'.")

    # Process data
    df_leadership = filter_leadership_items(df, preprocessing_level)
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
    print("\nExample items (original and processed):")
    for _, row in clean_df.sample(n=5).iterrows():
        print(f"{row['Construct']}:")
        print(f"Original: {row['Item']}")
        print(f"Processed: {row['Processed_Item']}")
        print()

if __name__ == "__main__":
    main()