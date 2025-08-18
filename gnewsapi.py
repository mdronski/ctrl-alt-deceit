import requests
from transformers import pipeline


api_key = "7642ab5e95effb7866058ed237ba83d5" 
search = "HSBC"  

url = "https://gnews.io/api/v4/search"


classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")


all_articles = []
pages_to_fetch = 5   

for page in range(1, pages_to_fetch + 1):
    params = {
        "q": search,
        "lang": "en",
        "max": 10,     
        "page": page,
        "sortby": "publishedAt",
        "token": api_key
    }

    print(f"Fetching page {page}...")
    r = requests.get(url, params=params)

    if r.status_code == 200:
        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            break  # stop if no more results
        all_articles.extend(articles)
    else:
        print("Error:", r.status_code, r.text)
        break

print(f"\n✅ Total articles collected: {len(all_articles)}\n")


if not all_articles:
    print("No news found :(")
else:
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for i, art in enumerate(all_articles, start=1):
        title = art["title"]
        description = art.get("description", "")
        text = title + " " + description 


        result = classifier(text[:512])
        sentiment = result[0]["label"].lower()
        score = result[0]["score"]

        sentiment_counts[sentiment] += 1

        print(f"{i}. {title}")
        print("   From:", art["source"]["name"])
        print("   Date:", art["publishedAt"])
        print("   Link:", art["url"])
        print("   Sentiment:", sentiment, f"(confidence {score:.2f})\n")

   
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
