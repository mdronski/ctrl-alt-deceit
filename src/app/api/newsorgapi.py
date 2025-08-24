import os
from langchain.tools import tool
from newsapi import NewsApiClient

@tool
def newsapi_tool(query: str) -> str:
    """
    Fetch latest news articles for a query.
    Returns raw article info (title, content, source, date, URL) for LLM analysis.
    """
    api_key = os.getenv("NEWSAPI_API_KEY")
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
        title = article.get("title", "No Title")
        description = article.get("description") or ""
        content = f"{title}. {description}".strip()
        source = article.get("source", {}).get("name", "Unknown")
        date = article.get("publishedAt", "Unknown")
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
