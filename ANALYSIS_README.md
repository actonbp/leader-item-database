# Leader Item Database Analysis Pipeline

This README provides instructions for running the analysis pipeline for the Leader Item Database project. For general project information, please refer to the [main README](./README.md).


## Prerequisites

Before running the analysis, ensure you have:

1. Python 3.7 or higher installed
2. pip (Python package installer)
3. An OpenAI API key

## Setup

1. Clone this repository (if you haven't already):
   ```
   git clone https://github.com/actonbp/leader-item-database.git
   cd leader-item-database
   ```

2. Install the project and its dependencies:
   ```
   pip install -e .
   ```
   
   For development, you can install additional dependencies:
   ```
   pip install -e .[dev]
   ```

3. Set up your OpenAI API key:
   - Create a file named `.env` in the project root directory
   - Add your API key to the file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

4. Ensure the raw data file `Cleaned_Item_Database.csv` is in the `data/raw/` directory.

## Running the Analysis

You can run the analysis in two ways:

### Option 1: Run Individual Scripts

Run each script in the following order:

1. Data Processing:
   ```
   python src/data_processing/process_data.py
   ```

2. Embedding Analysis:
   ```
   python src/analysis/embedding_analysis.py
   ```

3. Visualization:
   ```
   python src/visualization/plot_nomological_network.py
   ```

### Option 2: Run the Full Pipeline

Use the provided shell script to run all steps in sequence:

1. Make the script executable:
   ```
   chmod +x run_analysis.sh
   ```

2. Run the script:
   ```
   ./run_analysis.sh
   ```

## Output

After running the analysis:

- Processed data will be saved in `data/processed/cleaned_leadership_items.csv`
- Embeddings will be saved in `data/processed/item_embeddings.npy`
- The nomological network plot will be saved in `results/nomological_network_plot.png`

## Troubleshooting

If you encounter any issues:

1. Check that all prerequisites are met and setup steps are completed.
2. Ensure your OpenAI API key is correct and has the necessary permissions.
3. Review any error messages in the console output.
4. If running individual scripts, check the output after each step.

For more detailed information about the project structure and components, refer to the [main README](./README.md).

## Contributing

[Include contribution guidelines or link to them in the main README]

## License

For license information, please refer to the [main README](./README.md).

