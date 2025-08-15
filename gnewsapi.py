import requests
api_key = "7642ab5e95effb7866058ed237ba83d5"  # my gnews api key
search = "Infosys"  # what to search for

url = "https://gnews.io/api/v4/search"
params = {
    "q": search,
    "lang": "en",
    "max": 10,
    "token": api_key,
    "sortby": "publishedAt"
}

print("getting news for", search)
r = requests.get(url, params=params)

if r.status_code == 200:
    data = r.json()
    articles = data["articles"]
    if len(articles) == 0:
        print("no news found :(")
    else:
        for i, art in enumerate(articles):
            print(i+1, art["title"])
            print("   from:", art["source"]["name"])
            print("   date:", art["publishedAt"])
            print("   link:", art["url"])
            print("   desc:", art["description"])
            print()
else:
    print("error:", r.status_code)
    print(r.text)
