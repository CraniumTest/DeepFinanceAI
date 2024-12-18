import json
import pandas as pd
from transformers import pipeline

class SentimentModel:
    def __init__(self, sentiment_data_path):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.sentiment_data_path = sentiment_data_path
    
    def load_data(self):
        with open(self.sentiment_data_path, 'r') as f:
            return json.load(f)
    
    def analyze_sentiment(self, text):
        return self.sentiment_pipeline(text)

    def analyze_and_collect(self):
        data = self.load_data()
        results = []
        for item in data:
            sentiment = self.analyze_sentiment(item['text'])
            results.append({**item, 'sentiment': sentiment})
            print(f"Processed: {item['id']}")
        return pd.DataFrame(results)
