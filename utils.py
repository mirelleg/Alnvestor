from transformers import pipeline
import torch
from collections import Counter

def sentiment(text):
    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "text-classification", 
        model="ProsusAI/finbert",
        device=device,
        torch_dtype=torch.float32    # 🔥 加这一行，确保是实tensor
    )

    chunk_size = 450

    if len(text) <= chunk_size:
        result = pipe(text, truncation=True)[0]['label']
        return result
    
    sentiments = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        sentiment_result = pipe(chunk, truncation=True)[0]['label']
        sentiments.append(sentiment_result)
    
    sentiment_counts = Counter(sentiments)
    majority_sentiment, _ = sentiment_counts.most_common(1)[0]

    return majority_sentiment