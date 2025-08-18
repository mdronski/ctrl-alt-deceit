from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from newsorgapi import fetch_news_with_sentiment

GOOGLE_API_KEY = input("Enter your Google API Key: ").strip()
TAVILY_API_KEY = input("Enter your Tavily API Key: ").strip()

if not GOOGLE_API_KEY or not TAVILY_API_KEY:
    raise ValueError("Both Google and Tavily API keys are required!")

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# Tools
tavily_tool = TavilySearch(max_results=5, tavily_api_key=TAVILY_API_KEY)

if __name__ == "__main__":
    query = "HSBC"

    # MANUALLY
    news = fetch_news_with_sentiment.invoke(query)
    scandals = tavily_tool.invoke(f"{query} scandals")

    # Combine with the LLM
    combined_summary = llm.invoke(
        f"Here is recent news with sentiment:\n{news}\n\n"
        f"And here is Tavily search info about scandals:\n{scandals}\n\n"
        f"Please summarize them together into a single coherent report."
    )

    print("\n================= Final Summary =================")
    print(combined_summary.content if hasattr(combined_summary, "content") else combined_summary)
