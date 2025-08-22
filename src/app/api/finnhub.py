import datetime
import requests
from transformers import pipeline
from langchain.tools import tool

classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

API_KEY = "d2ja7o1r01qqoajabc60d2ja7o1r01qqoajabc6g"

@tool
def finnhub_tool(symbol: str) -> str:
    """
    Fetch the latest news articles for a stock symbol using Finnhub, and return titles, sources, dates, URLs, and sentiment.
    """
    to_date = datetime.date.today()
    from_date = to_date - datetime.timedelta(days=365)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol.upper(),
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return f"Error fetching data from Finnhub: {response.status_code} - {response.text}"

    articles = response.json()
    if not articles:
        return f"No articles found for {symbol}."

    report = []
    for article in articles[:5]:  # limit to latest 5 for brevity
        title = article.get("headline")
        summary = article.get("summary", "")
        content = f"{title}. {summary}"
        source = article.get("source", "Unknown")
        date = article.get("datetime")
        url = article.get("url")

        # Run FinBERT sentiment
        result = classifier(content[:512])
        sentiment = result[0]["label"].capitalize()

        report.append(
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Date: {date}\n"
            f"URL: {url}\n"
            f"Sentiment: {sentiment}\n"
            + "-" * 40
        )

    return "\n".join(report)
