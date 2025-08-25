import os
import requests
from langchain.tools import tool

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

@tool
def gnews_tool(query: str) -> str:
    """
    Fetch latest news articles for a query.
    Returns raw article info (title, content, source, date, URL) for LLM analysis.
    """
    url = "https://gnews.io/api/v4/search"
    all_articles = []

    for page in range(1, 3):  # fetch 2 pages max
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
            break
        all_articles.extend(articles)

    if not all_articles:
        return f"No articles found for '{query}'."

    report = []
    for art in all_articles[:5]:
        title = art.get("title", "No Title")
        description = art.get("description") or ""
        content = f"{title}. {description}".strip()
        source = art.get("source", {}).get("name", "Unknown")
        date = art.get("publishedAt", "Unknown")
        url = art.get("url", "#")

        report.append(
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Date: {date}\n"
            f"URL: {url}\n"
            f"Content: {content}\n"
            + "-" * 40
        )

    return "\n".join(report)
