api_key = "d2ja7o1r01qqoajabc60d2ja7o1r01qqoajabc6g"    
import requests
from datetime import date, timedelta
from transformers import pipeline

 
api_key = "d2ja7o1r01qqoajabc60d2ja7o1r01qqoajabc6g"    
symbol = "INFY"            

 
to_date = date.today()
from_date = to_date - timedelta(days=365)

url = "https://finnhub.io/api/v1/company-news"
params = {
    "symbol": symbol,
    "from": from_date.isoformat(),
    "to": to_date.isoformat(),
    "token": api_key
}

print(f"Fetching news for {symbol} ({from_date} → {to_date})...")
r = requests.get(url, params=params)

 
classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

if r.status_code == 200:
    articles = r.json()

    if not articles:
        print("No news found :(")
    else:
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

        print(f"\nFound {len(articles)} articles:\n")

        for i, art in enumerate(articles, start=1):
            headline = art.get("headline", "")
            summary = art.get("summary", "")
            text = headline + " " + summary

            
            result = classifier(text[:512])  # FinBERT limit
            sentiment = result[0]["label"].lower()
            score = result[0]["score"]

            sentiment_counts[sentiment] += 1

            print(f"{i}. {headline}")
            print("   Source:", art.get("source"))
            print("   Date:", art.get("datetime"))
            print("   Link:", art.get("url"))
            print("   Sentiment:", sentiment, f"(confidence {score:.2f})\n")

        # ---- Statistics (ignoring neutral) ----
        pos = sentiment_counts["positive"]
        neg = sentiment_counts["negative"]
        total = pos + neg

        print("\n📊 Sentiment Statistics (excluding Neutral):")
        if total > 0:
            pos_perc = (pos / total) * 100
            neg_perc = (neg / total) * 100
            print(f"  Positive: {pos} ({pos_perc:.1f}%)")
            print(f"  Negative: {neg} ({neg_perc:.1f}%)")
        else:
            print("  No positive or negative news found.")

else:
    print("Error:", r.status_code, r.text)
