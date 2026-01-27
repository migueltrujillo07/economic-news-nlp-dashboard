import pandas as pd
import os
import re
from urllib.parse import urlparse

class DataPreprocessor:
    def __init__(self, input_path="data/raw", output_path="data/processed"):
        self.input_path = input_path
        self.output_path = output_path
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def clean_title(self, text):
        """
        Cleans the article title by removing special characters and extra spaces.
        """
        if not isinstance(text, str):
            return ""
        # Remove HTML tags and special characters
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip().lower()

    def extract_domain(self, url):
        """
        Extracts the source domain from a URL (e.g., reuters.com).
        """
        try:
            return urlparse(url).netloc.replace('www.', '')
        except:
            return "unknown"

    def process_latest_file(self):
        """
        Loads the most recent file from raw, cleans it, and saves it to processed.
        """
        files = [f for f in os.listdir(self.input_path) if f.endswith('.csv')]
        if not files:
            print("No raw files found.")
            return
        
        # Get the latest file based on creation time
        latest_file = max([os.path.join(self.input_path, f) for f in files], key=os.path.getctime)
        print(f"Processing: {latest_file}")
        
        df = pd.read_csv(latest_file)
        
        # 1. Remove duplicates based on URL and Title
        initial_count = len(df)
        df = df.drop_duplicates(subset=['url'])
        df = df.drop_duplicates(subset=['title'])
        
        # 2. Clean titles
        df['clean_title'] = df['title'].apply(self.clean_title)
        
        # 3. Extract source domain
        df['source_name'] = df['url'].apply(self.extract_domain)
        
        # 4. Filter out empty titles
        df = df[df['clean_title'] != ""]
        
        final_count = len(df)
        print(f"Cleaning complete. Removed {initial_count - final_count} duplicates/empty rows.")
        
        # Save to processed
        output_filename = os.path.basename(latest_file).replace('gdelt_', 'processed_')
        output_full_path = os.path.join(self.output_path, output_filename)
        df.to_csv(output_full_path, index=False)
        print(f"Saved processed data to: {output_full_path}")
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process_latest_file()