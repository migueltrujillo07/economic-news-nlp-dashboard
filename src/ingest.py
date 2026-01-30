import requests
import pandas as pd
import os
from datetime import datetime
import re

class GdeltIngestion:
    """
    Handles data extraction from GDELT DOC API for economic news.
    """
    def __init__(self, output_path="data/raw"):
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.output_path = output_path
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def fetch_economic_news(self, keyword="economy", timespan = "1m", max_records=250, country=None):
        """
        Queries GDELT DOC API with time constraints.
        
        Args:
            keyword (str): Search term.
            timespan (str): Time range (e.g., '24h', '1w', '15min').
            max_records (int): Limit of articles to retrieve.
        """
        params = {
            "query": f'{keyword} sourcelang:english',
            "mode": "artlist",
            "maxrecords": max_records,
            "timespan": timespan,  # <--- Nuevo parámetro de tiempo
            "format": "json"
        }
        
        if country:
            params["query"] += f" sourcecountry:{country}"

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "articles" in data:
                df = pd.DataFrame(data["articles"])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Creamos un nombre seguro: solo letras, números y guiones bajos
                safe_keyword = re.sub(r'[^\w\s-]', '', keyword).strip().replace(' ', '_')
                # Limitamos el largo para que no sea un nombre eterno
                safe_keyword = safe_keyword[:30] 
                
                filename = f"gdelt_{safe_keyword}_{timestamp}.csv"
                
                full_path = os.path.join(self.output_path, filename)
                df.to_csv(full_path, index=False)
                print(f"Successfully saved {len(df)} articles to {full_path}")
                return df
            else:
                print("No articles found for the given query.")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to GDELT API: {e}")
            return None

if __name__ == "__main__":
    ingestor = GdeltIngestion()
    
    # Query mejorada para el mercado global
    economic_query = '(economy OR inflation OR "interest rates" OR GDP OR recession OR unemployment OR "central bank" OR "Federal Reserve" OR ECB)'
    
    # Cambiamos sourcelang a english (o lo omitimos, ya que es el default)
    # Aumentamos max_records a 250 para tener una buena base
    news_df = ingestor.fetch_economic_news(
        keyword=economic_query, 
        timespan="1m", 
        max_records=250
    )