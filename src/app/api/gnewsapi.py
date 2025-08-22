import requests
from langchain.tools import tool
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

GNEWS_API_KEY = "7642ab5e95effb7866058ed237ba83d5"

@tool
def gnews_tool(query: str) -> str:
    """
    Fetch the latest news articles for a keyword or company using GNews API and return titles, sources, dates, URLs, and sentiment.
    """
    url = "https://gnews.io/api/v4/search"
    all_articles = []
    pages_to_fetch = 2

    for page in range(1, pages_to_fetch + 1):
        params = {
            "q": query,
            "lang": "en",
            "max": 10,
            "page": page,
            "sortby": "publishedAt",
            "token": GNEWS_API_KEY
        }

        r = requests.get(url, params=params)
        if r.status_code != 200:
            return f"Error fetching data from GNews: {r.status_code} - {r.text}"

        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            break  # stop if no more results
        all_articles.extend(articles)

    if not all_articles:
        return f"No articles found for '{query}'."

    report = []
    for art in all_articles[:5]:  # limit to latest 5 for concise output
        title = art.get("title", "No Title")
        description = art.get("description", "")
        text = f"{title}. {description}".strip()

        source = art.get("source", {}).get("name", "Unknown")
        date = art.get("publishedAt", "Unknown")
        url = art.get("url", "#")

        # Sentiment Analysis with FinBERT
        result = classifier(text[:512])
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
