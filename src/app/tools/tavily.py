from langchain_tavily import TavilySearch

tavily = TavilySearch(
    max_results=5,
    topic="general",
)