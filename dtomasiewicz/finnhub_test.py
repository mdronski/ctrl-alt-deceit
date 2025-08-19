import finnhub
from langchain.tools import tool


# Setup client
finnhub_client = finnhub.Client(api_key="d2hccupr01qon4ebi1p0d2hccupr01qon4ebi1pg")

# print(finnhub_client.symbol_lookup('HSBC'))

@tool
def finnhub_tool(company_name: str, date_from: str, date_to: str) -> list:
    """
    Retrieves company news from Finnhub for a specified company and date range.

    Args:
        company_name (str): The name of the company (e.g., 'Apple').
        date_from (str): The start date for the news search in 'YYYY-MM-DD' format.
        date_to (str): The end date for the news search in 'YYYY-MM-DD' format.

    Returns:
        list: A list of news articles for the specified company and date range.
              Returns an empty list if no news is found or an error occurs during the process.
    """
    try:
        lookup_result = finnhub_client.symbol_lookup(company_name)
        if not lookup_result['result']:
            print(f"Error: No symbol found for company name '{company_name}'.")
            return []
        symbol = lookup_result['result'][0]['symbol']
        news_articles = finnhub_client.company_news(symbol, _from=date_from, to=date_to)
        return news_articles
    except finnhub.FinnhubAPIException as e:
        print(f"Finnhub API Error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

# print(symbol)
# print(len(res))

with open("file_name.json", 'w', encoding='utf-8') as file:
    result = finnhub_tool.run({"company_name": "Tesla", "date_from": "2025-06-01", "date_to": "2025-08-01"})
    file.write(str(result))

print("Finished")