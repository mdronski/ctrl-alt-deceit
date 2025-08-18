import finnhub

# Setup client
finnhub_client = finnhub.Client(api_key="d2hccupr01qon4ebi1p0d2hccupr01qon4ebi1pg")

# print(finnhub_client.symbol_lookup('HSBC'))


# Stock candles
res = finnhub_client.company_news('HSBC', _from="2020-06-01", to="2025-08-01")
print(res)
print(len(res))