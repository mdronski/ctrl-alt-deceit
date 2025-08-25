from app.tools.api.finnhub import finnhub_tool
from app.tools.api.gnewsapi import gnews_tool
from app.tools.api.newsorgapi import newsapi_tool
from app.tools.tavily import tavily

TOOLS = [
    tavily,
    newsapi_tool,
    gnews_tool,
    finnhub_tool
]