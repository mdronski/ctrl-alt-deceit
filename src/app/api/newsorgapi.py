from langchain.tools import tool
from newsapi import NewsApiClient
from textblob import TextBlob

def get_sentiment(text: str) -> str:
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

@tool
def newsapi_tool(query: str) -> str:
    """
    Fetch the latest news articles for a company or keyword and return titles, sources, dates, URLs, and sentiment.
    """
    api_key = "5a248d7325f64806988f558ace817e8f"
    newsapi = NewsApiClient(api_key=api_key)

    response = newsapi.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=5,
    )

    articles = response.get("articles", [])
    if not articles:
        return f"No articles found for {query}."

    report = []
    for article in articles:
        title = article.get("title")
        content = article.get("content") or article.get("description")
        source = article.get("source", {}).get("name", "Unknown")
        date = article.get("publishedAt")
        url = article.get("url")
        sentiment = get_sentiment(content)

        report.append(
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Date: {date}\n"
            f"URL: {url}\n"
            f"Sentiment: {sentiment}\n"
            + "-" * 40
        )

    return "\n".join(report)
