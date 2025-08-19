from polygon import RESTClient
from langchain_core.tools import tool

client = RESTClient(api_key="doYsnLZqJmyklTl8ifPYqCGhJqaYvhyR")


news_articles = client.list_ticker_news(
	"AAPL",
	params={"published_utc.gte": "2025-07-03"},
	order="desc",
	limit=2
	)

for article in news_articles:
    print(f"{article.title} [Insights: {article.insights}]")



