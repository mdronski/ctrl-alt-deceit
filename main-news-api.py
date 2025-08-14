from newsapi import NewsApiClient
from textblob import TextBlob

def main():
    API_KEY = "5a248d7325f64806988f558ace817e8f"
    counterparty = "HSBC"

    newsapi = NewsApiClient(api_key=API_KEY)

    response = newsapi.get_everything(
        q=counterparty,
        language='en',
        sort_by='publishedAt',
        page_size=10
    )

    articles = response.get('articles', [])

    if not articles:
        print("No articles found.")
        return

    for article in articles:
        title = article.get('title')
        content = article.get('content') or article.get('description')
        source = article.get('source', {}).get('name', 'Unknown')
        date = article.get('publishedAt')
        url = article.get('url')

        sentiment = get_sentiment(content)

        print(f"Title: {title}")
        print(f"Source: {source}")
        print(f"Date: {date}")
        print(f"URL: {url}")
        print(f"Sentiment: {sentiment}")
        print("-" * 50)

def get_sentiment(text):
    if not text:
        return "Neutral"

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

if __name__ == "__main__":
    main()
