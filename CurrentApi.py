import requests
from transformers import pipeline
import gradio as gr

# ---- Currents API Setup ----
CURRENTS_KEY = "kYGv3Z5J7kPefNy3Sc7eem00RwL-AJ-lEegIIy69tiVt3BWG"

# ---- Sentiment Model Setup ----
classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_company_news(company_name):
    url = "https://api.currentsapi.services/v1/search"
    params = {
        "keywords": company_name,
        "language": "en",
        "apiKey": CURRENTS_KEY
    }

    r = requests.get(url, params=params)

    if r.status_code != 200:
        return f"❌ Error: {r.status_code} - {r.text}"

    data = r.json()
    articles = data.get("news", [])
    if not articles:
        return f"No news found for **{company_name}**."

    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    results = []

    for i, art in enumerate(articles, start=1):
        title = art.get("title", "")
        description = art.get("description", "")
        text = title + " " + description

        sentiment = None
        score = 0
        if text.strip():
            result = classifier(text[:512])  # avoid length issues
            sentiment = result[0]["label"].lower()
            score = result[0]["score"]
            sentiment_counts[sentiment] += 1

        results.append(
            f"### {i}. {title}\n"
            f"- **Sentiment:** {sentiment} (confidence {score:.2f})\n"
            f"- **Date:** {art.get('published')}\n"
            f"- [Read more]({art.get('url')})\n"
        )

    # ---- Statistics (ignoring neutral) ----
    pos = sentiment_counts["positive"]
    neg = sentiment_counts["negative"]
    total = pos + neg
    stats = "## 📊 Sentiment Summary (excluding Neutral)\n"
    if total > 0:
        pos_perc = (pos / total) * 100
        neg_perc = (neg / total) * 100
        stats += f"- Positive: {pos} ({pos_perc:.1f}%)\n"
        stats += f"- Negative: {neg} ({neg_perc:.1f}%)\n"
    else:
        stats += "- No positive or negative news found."

    return "\n".join(results) + "\n\n" + stats


# ---- Gradio Interface ----
demo = gr.Interface(
    fn=analyze_company_news,
    inputs=gr.Textbox(label="Enter Company Name"),
    outputs=gr.Markdown(),
    title="Company News Sentiment Analyzer",
    description="Enter a company name (e.g., Infosys, Apple, Tesla) to fetch latest news and analyze sentiment."
)

demo.launch()
