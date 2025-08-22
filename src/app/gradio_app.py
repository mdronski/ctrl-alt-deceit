import os
import io
import datetime
import tempfile

import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv

from textblob import TextBlob

from app.api.newsorgapi import newsapi_tool
from app.api.finnhub import finnhub_tool
from app.api.gnewsapi import gnews_tool

load_dotenv()

from app.agent import agent

# =========================
# Functions
# =========================

def ask_agent(company_name: str, rows: list[dict]) -> str:
    # TODO conduct valid prompt engineering
    """
    Sends structured article data to the LLM agent for summarization.
    """
    if not rows:
        return f"No articles found to analyze for {company_name}."

    news_text = "\n".join(
        f"- {r['date']} · {r['source']} — {r['title']} ({r['sentiment']})\n  {r['url']}"
        for r in rows
    )

    prompt = f"""
                You are a due diligence analyst. Analyze the following recent news about the company '{company_name}'.
                
                Your task is to identify **potential red flags** that may affect business risk, reputation, legal issues, or compliance concerns.
                
                Here are the recent articles:
                {news_text}
                
                Return your analysis as a **markdown-formatted summary**, grouped by issue type if possible. Highlight any concerns such as fraud, regulation, fines, lawsuits, etc.
            """

    response = agent.invoke({"messages": [{"role": "user", "content": prompt.strip()}]})
    return response["messages"][-1].content

def _sentiment_label(text: str) -> str:
    if not text:
        return "Neutral"
    pol = TextBlob(text).sentiment.polarity
    if pol > 0.1:
        return "Positive"
    if pol < -0.1:
        return "Negative"
    return "Neutral"

def fetch_news_combined(company: str):
    """
    Fetches news from NewsAPI, GNews, and Finnhub.
    Returns a unified list of dicts and an optional error message.
    """
    tools = [newsapi_tool, gnews_tool, finnhub_tool]
    articles_all = []
    errors = []

    for tool_fn in tools:
        try:
            result = tool_fn(company)
            if "No articles found" in result:
                continue

            # Split by separator line
            articles = result.split("-" * 40)
            for a in articles:
                if not a.strip():
                    continue
                lines = a.strip().split("\n")
                d = {}
                for line in lines:
                    if line.startswith("Title:"):
                        d["title"] = line.replace("Title:", "").strip()
                    elif line.startswith("Source:"):
                        d["source"] = line.replace("Source:", "").strip()
                    elif line.startswith("Date:"):
                        d["date"] = line.replace("Date:", "").strip()[:10]
                    elif line.startswith("URL:"):
                        d["url"] = line.replace("URL:", "").strip()
                    elif line.startswith("Sentiment:"):
                        d["sentiment"] = line.replace("Sentiment:", "").strip()
                if d:
                    articles_all.append(d)
        except Exception as e:
            errors.append(f"{tool_fn.__name__} failed: {str(e)}")

    if not articles_all:
        return [], "No articles found from any source." + (f" Errors: {errors}" if errors else "")

    return articles_all, None


def fetch_news(company: str):
    """
    Calls fetch_news_with_sentiment() and returns list of dicts for Gradio table.
    """
    report_str = newsapi_tool(company)
    if "No articles found" in report_str:
        return [], "No articles found."

    # Split by article blocks
    articles = report_str.split("-" * 40)
    rows = []
    for a in articles:
        if not a.strip():
            continue
        lines = a.strip().split("\n")
        d = {}
        for line in lines:
            if line.startswith("Title:"):
                d["title"] = line.replace("Title:", "").strip()
            elif line.startswith("Source:"):
                d["source"] = line.replace("Source:", "").strip()
            elif line.startswith("Date:"):
                d["date"] = line.replace("Date:", "").strip()[:10]
            elif line.startswith("URL:"):
                d["url"] = line.replace("URL:", "").strip()
            elif line.startswith("Sentiment:"):
                d["sentiment"] = line.replace("Sentiment:", "").strip()
        if d:
            rows.append(d)
    return rows, None

def make_sentiment_pie(rows):
    # Count sentiments
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in rows:
        counts[r["sentiment"]] = counts.get(r["sentiment"], 0) + 1

    # Define colors (Positive=green, Neutral=blue, Negative=red)
    colors = ["green", "blue", "red"]
    labels = [f"{k} ({v})" for k, v in counts.items()]

    # Create pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts.values(),
        labels=labels,
        autopct="%1.0f%%",
        startangle=140,
        colors=colors,
        explode=(0.05, 0.05, 0.05),
        shadow=True,
        wedgeprops={'edgecolor': 'black'}
    )
    ax.axis("equal")  # keep circle

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def build_scandal_timeline(company: str, rows):
    """Use negative articles as 'scandals' and build a simple timeline."""
    negatives = [r for r in rows if r["sentiment"] == "Negative"]
    if not negatives:
        return f"No notable negative events found for **{company}** in recent articles."
    lines = []
    for r in sorted(negatives, key=lambda x: x["date"] or "", reverse=True):
        lines.append(f"- **{r['date']}** — [{r['title']}]({r['url']}) · _{r['source']}_")
    return "\n".join(lines)


def compute_risk(company: str, summary_md: str, rows):
    """Lightweight heuristic for a 0–100 risk score."""
    neg = sum(1 for r in rows if r["sentiment"] == "Negative")
    total = max(len(rows), 1)
    neg_ratio = neg / total

    summary_lower = (summary_md or "").lower()
    red_keywords = [
        "aml", "money laundering", "sanction", "fraud", "bribe",
        "regulator", "fine", "penalty", "scandal", "lawsuit", "investigation",
        "whistleblower", "corruption", "sanctions", "non-compliance"
    ]
    kw_hits = sum(k in summary_lower for k in red_keywords)

    # weights
    score = round(60 * neg_ratio + 8 * min(kw_hits, 4))
    score = max(0, min(100, score))

    if score < 40:
        label = "🟢 Low Risk"
    elif score < 70:
        label = "🟡 Medium Risk"
    else:
        label = "🔴 High Risk"

    return f"**Risk Score:** **{score}/100** → {label}"


def extract_tags(summary_md: str):
    tags = []
    tag_map = {
        "#AML": ["aml", "anti-money laundering", "money laundering"],
        "#Fraud": ["fraud", "scam", "embezzlement"],
        "#ESG": ["esg", "environment", "sustain", "governance", "social"],
        "#Reputation": ["reputation", "public image", "backlash"],
        "#Compliance": ["regulator", "compliance", "fine", "penalty"],
        "#Sanctions": ["sanction"],
        "#Litigation": ["lawsuit", "litigation", "class action"],
    }
    text = (summary_md or "").lower()
    for tag, kws in tag_map.items():
        if any(k in text for k in kws):
            tags.append(tag)
    if not tags:
        tags = ["#GeneralRisk"]
    return " ".join(f"`{t}`" for t in tags)


def export_markdown(company: str, summary: str, rows, timeline_md: str, risk_md: str, tags_md: str):
    today = datetime.date.today().isoformat()
    md = [
        f"# Company Risk Report: {company}",
        f"_Generated: {today}_",
        "",
        "## 📑 Summary",
        summary or "*No summary available.*",
        "",
        "## 📊 Sentiment Snapshot",
        f"- Positive: {sum(1 for r in rows if r['sentiment']=='Positive')}",
        f"- Neutral: {sum(1 for r in rows if r['sentiment']=='Neutral')}",
        f"- Negative: {sum(1 for r in rows if r['sentiment']=='Negative')}",
        "",
        "### Recent Articles",
        ]
    if rows:
        for r in rows:
            md.append(f"- **{r['date']}** · _{r['source']}_ — [{r['title']}]({r['url']}) · **{r['sentiment']}**")
    else:
        md.append("_No recent articles._")

    md += [
        "",
        "## 📰 Scandal Timeline",
        timeline_md or "_No items._",
        "",
        "## ⚠️ Risk Assessment",
        risk_md or "_N/A_",
        "",
        "## 🏷️ Tags",
        tags_md or "_N/A_",
        "",
        ]

    content = "\n".join(md)
    path = os.path.join(tempfile.gettempdir(), f"{company}_risk_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# =========================
# Gradio UI
# =========================

with gr.Blocks(theme=gr.themes.Soft(), css="""
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f5f5f5;
        font-family: 'Segoe UI', sans-serif;
    }
    #header { text-align: center; margin-bottom: 24px; color: #ffffff; }
    #company-section { text-align: center; margin-bottom: 8px; }
    #result-box {
        background-color: #2d2d2d; color: #f5f5f5; padding: 22px;
        border-radius: 20px; box-shadow: 0 12px 25px rgba(0,0,0,0.25);
        transition: box-shadow 0.3s ease; line-height: 1.6;
    }
    #result-box:hover { box-shadow: 0 18px 35px rgba(0,0,0,0.35); }
    .gr-button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white; font-weight: 700; border-radius: 12px; padding: 12px 20px;
        transition: transform 0.2s ease;
    }
    .gr-button:hover { background: linear-gradient(90deg, #764ba2, #667eea); transform: scale(1.05); }
    .summary-header {
        background: #3a3a3a; color: #ffffff !important; padding: 12px;
        border-radius: 12px; font-size: 18px; text-align: center; margin-bottom: 10px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }
""") as app:

    # Header
    gr.Markdown(
        "# 🏢 Company Risk Analyzer\nAnalyze potential **red flags** for business partnerships.",
        elem_id="header"
    )

    # Input + Start
    with gr.Row(elem_id="company-section"):
        company_name = gr.Textbox(label="🔍 Company Name", placeholder="e.g., HSBC, Tesla, Binance...", lines=1)
    start_button = gr.Button("🚀 Start Analysis", variant="primary")

    # Tabs
    with gr.Tabs():
        with gr.Tab("📑 Summary"):
            summary_hdr = gr.Markdown("📑 Summary of Findings", elem_classes="summary-header")
            summary_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("📊 Sentiment"):
            sentiment_img = gr.Image(type="filepath", label="Sentiment Distribution (Recent News)")
            headlines_tbl = gr.Dataframe(headers=["date", "source", "title", "sentiment", "url"], interactive=False)

        with gr.Tab("📰 Scandals"):
            timeline_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("⚠️ Risk"):
            risk_md = gr.Markdown("", elem_id="result-box")
            tags_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("📄 Export"):
            export_btn = gr.Button("Generate Markdown Report")
            export_file = gr.File(label="Download Report (.md)")

    # -------- logic
    def analyze(company: str):
        # Fetch news
        rows, err = fetch_news_combined(company)

        # LLM summary
        summary = ask_agent(company, rows)

        if err:
            pie_path = None
            timeline = f"_News error_: {err}"
        else:
            # Save pie chart safely in system temp dir
            buf = make_sentiment_pie(rows)
            tmp_dir = tempfile.gettempdir()
            pie_path = os.path.join(tmp_dir, f"{company.replace(' ', '_')}_sentiment.png")
            with open(pie_path, "wb") as f:
                f.write(buf.read())

            # Build scandal timeline from negative news
            timeline = build_scandal_timeline(company, rows)

        # 3) Risk + tags
        risk = compute_risk(company, summary, rows)
        tags = extract_tags(summary)

        # dataframe rows (ordered columns)
        table = [
            [r["date"], r["source"], r["title"], r["sentiment"], r["url"]]
            for r in rows
        ]

        return summary, pie_path, table, timeline, risk, tags

    start_button.click(
        fn=analyze,
        inputs=[company_name],
        outputs=[summary_md, sentiment_img, headlines_tbl, timeline_md, risk_md, tags_md],
        show_progress="full"
    )

    def do_export(company: str, summary: str, table, timeline: str, risk: str, tags: str):
        rows = []
        if table is not None and not table.empty:
            for row in table.itertuples(index=False):
                rows.append({"date": row[0], "source": row[1], "title": row[2], "sentiment": row[3], "url": row[4]})
        path = export_markdown(company, summary, rows, timeline, risk, tags)
        return path

    export_btn.click(
        fn=do_export,
        inputs=[company_name, summary_md, headlines_tbl, timeline_md, risk_md, tags_md],
        outputs=[export_file],
        show_progress=True
    )

def main():
    app.launch()

if __name__ == "__main__":
    main()
