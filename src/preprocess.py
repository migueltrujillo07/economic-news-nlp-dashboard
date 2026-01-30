import pandas as pd
import os
import re
from urllib.parse import urlparse

class DataPreprocessor:
    def __init__(self, input_path="data/raw", output_path="data/processed"):
        self.input_path = input_path
        self.output_path = output_path
        
        # Ensure the output directory exists
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def clean_title(self, text):
        """
        Standardizes the title by removing HTML tags, special characters, and extra whitespace.
        """
        if not isinstance(text, str):
            return ""
        # Remove HTML and non-alphanumeric characters (keeping spaces)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip().lower()

    def process_data(self, custom_output_name=None):
        """
        Main pipeline: Loads the latest raw file, cleans it, and saves it.
        Args:
            custom_output_name (str): Optional filename for the output CSV.
        """
        files = [f for f in os.listdir(self.input_path) if f.endswith('.csv')]
        if not files:
            print("No raw files found in input path.")
            return
        
        # Pick the most recent file based on creation time
        latest_file = max([os.path.join(self.input_path, f) for f in files], key=os.path.getctime)
        print(f"Loading raw data from: {latest_file}")
        
        df = pd.read_csv(latest_file)
        
        # Data Cleaning Phase
        # 1. Deduplication (Critical for data integrity)
        df = df.drop_duplicates(subset=['url'])
        
        # 2. String Normalization
        df['clean_title'] = df['title'].apply(self.clean_title)
        
        # 3. Source Extraction (Feature Engineering for Analytics)
        df['source_name'] = df['url'].apply(lambda x: urlparse(x).netloc.replace('www.', ''))
        
        # Define output filename
        if custom_output_name:
            filename = custom_output_name if custom_output_name.endswith('.csv') else f"{custom_output_name}.csv"
        else:
            # Default naming convention
            filename = os.path.basename(latest_file).replace('gdelt_', 'processed_')
            
        output_full_path = os.path.join(self.output_path, filename)
        
        # Save the cleaned dataset
        df.to_csv(output_full_path, index=False)
        print(f"Processed data successfully saved to: {output_full_path}")
        
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    # Example: Giving a specific name for the EDA phase
    preprocessor.process_data(custom_output_name="clean_economic_news_v1")