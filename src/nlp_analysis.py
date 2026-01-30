import pandas as pd
import os
import torch
from transformers import pipeline

class EconomicNLPAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initializes the NLP pipeline with FinBERT.
        FinBERT is a pre-trained NLP model to analyze sentiment of financial text.
        """
        print(f"Initializing model: {model_name}...")
        # Check if GPU is available, otherwise use CPU
        device = 0 if torch.cuda.is_available() else -1
        
        # The pipeline handles tokenization and classification internally
        self.analyzer = pipeline("sentiment-analysis", model=model_name, device=device)
        self.input_file = "data/processed/clean_economic_news_v1.csv"
        self.output_dir = "data/results/"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run_inference(self):
        """
        Loads the processed data, performs sentiment analysis, and saves the results.
        """
        if not os.path.exists(self.input_file):
            print(f"Error: {self.input_file} not found.")
            return

        # 1. Load data
        df = pd.read_csv(self.input_file)
        print(f"Analyzing {len(df)} headlines...")

        # 2. Perform Inference
        # We pass the cleaned titles to the model
        titles = df['clean_title'].fillna("").tolist()
        
        # truncation=True ensures we don't exceed the model's max token limit
        results = self.analyzer(titles, truncation=True)

        # 3. Process results
        # The model returns a list of dicts: [{'label': 'positive', 'score': 0.99}, ...]
        df['sentiment_label'] = [res['label'] for res in results]
        df['sentiment_score'] = [res['score'] for res in results]

        # 4. Save to results folder
        output_path = os.path.join(self.output_dir, "sentiment_results_v1.csv")
        df.to_csv(output_path, index=False)
        
        print(f"Success! Results saved to {output_path}")
        print("\n--- Sentiment Distribution ---")
        print(df['sentiment_label'].value_counts())

if __name__ == "__main__":
    analyzer = EconomicNLPAnalyzer()
    analyzer.run_inference()