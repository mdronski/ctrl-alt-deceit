import datetime
import os
import requests
from langchain.tools import tool

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

@tool
def finnhub_tool(symbol: str) -> str:
    """
    Fetch latest company news from Finnhub.
    Returns raw article info (title, content, source, date, URL).
    """
    to_date = datetime.date.today()
    from_date = to_date - datetime.timedelta(days=365)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol.upper(),
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": FINNHUB_API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error fetching data from Finnhub: {response.status_code} - {response.text}"

    articles = response.json()
    if not articles:
        return f"No articles found for {symbol}."

    report = []
    for article in articles[:5]:
        title = article.get("headline", "No Title")
        summary = article.get("summary") or ""
        content = f"{title}. {summary}".strip()
        source = article.get("source", "Unknown")
        date = article.get("datetime", "Unknown")
        url = article.get("url", "#")

        report.append(
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Date: {date}\n"
            f"URL: {url}\n"
            f"Content: {content}\n"
            + "-" * 40
        )

    return "\n".join(report)
