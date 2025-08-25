import hashlib
import io
import os
import re
import tempfile
from datetime import datetime, timezone, date
from difflib import SequenceMatcher
from urllib.parse import urlparse
from transformers import pipeline
import gradio as gr
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from app.agent import agent
from app.api.finnhub import finnhub_tool
from app.api.gnewsapi import gnews_tool
from app.api.newsorgapi import newsapi_tool

load_dotenv()
classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# =========================
# Keyword tiers (HARD/ MEDIUM / SOFT)
# =========================

HARD_KEYWORDS = {
    # enforcement & criminal / insolvency
    "fraud", "bribery", "embezzlement", "corruption", "money laundering", "aml",
    "sanctions evasion", "insider trading",
    "fine", "fines", "penalty", "penalties", "settlement",
    "lawsuit", "litigation", "class action",
    "indictment", "arrest", "criminal charges",
    "bankruptcy", "insolvency", "collapse", "liquidation", "administration"
}

MEDIUM_KEYWORDS = {
    # regulatory & supervisory without proven wrongdoing
    "regulatory scrutiny", "regulator scrutiny", "scrutiny from regulators",
    "regulatory probe", "probe", "investigation", "inquiry",
    "supervisory review", "compliance review", "regulatory review",
    "watchlist", "warning letter"
}

SOFT_NEGATIVE_TERMS = {
    # business hygiene / strategy – should not be scandals by themselves
    "revamp", "restructuring", "restructure", "layoffs", "job cuts", "headcount",
    "exit clients", "client exits", "client offboarding", "offboarding",
    "reorg", "re-organization", "guidance cut", "profit warning", "downgrade"
}


# =========================
# Parsing / Normalization helpers
# =========================

def parse_to_utc(date_time_raw: str | None) -> datetime | None:
    """
        Parse to timezone-aware UTC datetime.
    """
    if not date_time_raw:
        return None
    try:
        iso = str(date_time_raw).replace("Z", "+00:00")
        d = datetime.fromisoformat(iso)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        val = float(str(date_time_raw).strip())
        return datetime.fromtimestamp(val, tz=timezone.utc)
    except Exception:
        pass
    # yyyy-mm-dd
    try:
        d = datetime.strptime(str(date_time_raw)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return d
    except Exception:
        return None


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (title or "").lower())).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def url_key(url: str | None) -> str:
    """Normalize URL to host + path (without query/fragment) for dedup."""
    if not url:
        return ""
    try:
        u = urlparse(url)
        return f"{u.netloc.lower()}:{u.path}"
    except Exception:
        return url or ""


def deduplicate_articles(articles: list[dict], similarity_threshold: float = 0.92) -> list[dict]:
    """
    Dedup by:
      1) exact normalized title hash
      2) same URL host+path
      3) fuzzy title similarity across already-kept items
    Keeps the first occurrence.
    """
    unique: list[dict] = []
    seen_hashes = set()
    seen_urls = set()

    for art in articles:
        t = art.get("title", "")
        nk = normalize_title(t)
        h = hashlib.md5(nk.encode()).hexdigest()
        ukey = url_key(art.get("url"))

        if h in seen_hashes or (ukey and ukey in seen_urls):
            continue

        is_dup = False
        for u in unique:
            if ukey and url_key(u.get("url")) == ukey:
                is_dup = True
                break
            if title_similarity(t, u.get("title", "")) >= similarity_threshold:
                is_dup = True
                break

        if not is_dup:
            seen_hashes.add(h)
            if ukey:
                seen_urls.add(ukey)
            unique.append(art)

    return unique


def finbert_sentiment(text: str) -> str:
    if not text:
        return "Neutral"
    result = classifier(text[:512])[0]["label"].lower()
    return {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}.get(result, "Neutral")


def adjust_sentiment(title: str, sentiment: str) -> str:
    """
    Downgrades "Negative" to "Neutral" for soft business terms (revamp/layoffs/etc).
    Leaves true negatives as-is.
    """
    t = (title or "").lower()
    if sentiment == "Negative" and any(term in t for term in SOFT_NEGATIVE_TERMS):
        return "Neutral"
    return sentiment


def classify_article_level(title: str, body: str = "") -> str:
    """
    HARD: contains any HARD_KEYWORDS
    MEDIUM: contains any MEDIUM_KEYWORDS (and no HARD)
    SOFT: otherwise
    """
    t = f"{title or ''} {body or ''}".lower()
    if any(k in t for k in HARD_KEYWORDS):
        return "HARD"
    if any(k in t for k in MEDIUM_KEYWORDS):
        return "MEDIUM"
    return "SOFT"


def is_relevant(company: str, title: str, url: str, source: str) -> bool:
    """
    Keep only articles clearly about the target company to prevent pollution
    (e.g., random 'GST buzz' pieces).
    Rules:
      - title contains the full company phrase, OR
      - title contains any strong token from company name (>=3 chars), OR
      - url host or path contains a strong token (handles tickers/brands in URLs)
    """
    if not company:
        return True

    full = company.strip().lower()
    title_l = (title or "").lower()
    if full and full in title_l:
        return True

    tokens = [tok for tok in re.split(r"[^a-z0-9]+", full) if len(tok) >= 3]
    if tokens and any(tok in title_l for tok in tokens):
        return True

    try:
        u = urlparse(url or "")
        host_path = (u.netloc + u.path).lower()
        if tokens and any(tok in host_path for tok in tokens):
            return True
    except Exception:
        pass

    return False


# =========================
# LLM summary
# =========================

def ask_agent(company_name: str, rows: list[dict]) -> str:
    if not rows:
        return f"No articles found to analyze for {company_name}."

    news_text = "\n".join(
        f"- {row.get('date', '')} · {row.get('source', '')} — {row.get('title', '')} ({row.get('sentiment', '')})\n  {row.get('url', '')}"
        for row in rows
    )

    prompt = f"""
    You are a due diligence analyst. Analyze recent news about '{company_name}'.
    Identify **red flags** (fraud, AML/sanctions evasion, fines/penalties, settlements, lawsuits/litigation, indictments/arrests, bankruptcy/insolvency).
    Treat **regulatory scrutiny/probes/inquiries** as **medium risk**, not scandals, unless tied to enforcement or fraud.
    Treat **restructuring/layoffs/revamps/client offboarding** as **operational** unless combined with the above.
    
    Articles:
    {news_text}

    Do a web search to verify the articles and look for other potential risks about the company.
    Return a concise **markdown summary** grouped by issue type.
    """
    response = agent.invoke({"messages": [{"role": "user", "content": prompt.strip()}]})
    return response["messages"][-1].content


# =========================
# Fetch + normalize
# =========================

def fetch_news(company: str):
    """
    Retrieves news from NewsAPI, GNews, and Finnhub.
    Returns a unified list of dicts and an optional error message.
    Applies relevance filtering and aggressive de-duplication.
    """
    tools = [newsapi_tool, gnews_tool, finnhub_tool]
    articles_all: list[dict] = []
    errors = []

    for tool_fn in tools:
        try:
            result = tool_fn(company)
            if "No articles found" in result:
                continue

            articles = result.split("-" * 40)
            for article in articles:
                if not article.strip():
                    continue
                lines = [ln for ln in article.strip().split("\n") if ln.strip()]
                record = {}
                for line in lines:
                    if line.startswith("Title:"):
                        record["title"] = line.replace("Title:", "").strip()
                    elif line.startswith("Source:"):
                        record["source"] = line.replace("Source:", "").strip()
                    elif line.startswith("Date:"):
                        record["date_raw"] = line.replace("Date:", "").strip()
                        dt = parse_to_utc(record["date_raw"])
                        record["date"] = dt.date().isoformat() if dt else ""
                    elif line.startswith("URL:"):
                        record["url"] = line.replace("URL:", "").strip()
                    elif line.startswith("Content:"):
                        record["content"] = line.replace("Content:", "").strip()

                if not record:
                    continue

                # Relevance gate
                if not is_relevant(company, record.get("title", ""), record.get("url", ""), record.get("source", "")):
                    continue

                # Sentiment
                base_sent = finbert_sentiment(record.get("title", ""))

                record["sentiment"] = adjust_sentiment(record.get("title", ""), base_sent)

                level = classify_article_level(record.get("title", ""))

                articles_all.append({
                    "title": record.get("title", ""),
                    "source": record.get("source", ""),
                    "content": record.get("content", ""),
                    "date": record.get("date", ""),
                    "date_raw": record.get("date_raw", ""),
                    "url": record.get("url", ""),
                    "sentiment": record.get("sentiment", "Neutral"),
                    "level": level,
                })

        except Exception as e:
            errors.append(f"{tool_fn.name} failed: {str(e)}")

    print(f"Total articles fetched: {len(articles_all)}")

    if not articles_all:
        return [], "No articles found from any source." + (f" Errors: {errors}" if errors else "")

    # Deduplicate articles
    articles_all = deduplicate_articles(articles_all)

    print(f"Articles after deduplication: {len(articles_all)}")

    return articles_all, None


# =========================
# Visualization + timeline
# =========================
def make_sentiment_pie(rows):
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for row in rows:
        sent = str(row.get("sentiment", "Neutral")).capitalize()
        if sent not in counts:
            sent = "Neutral"
        counts[sent] += 1

    labels = [f"{k} ({v})" for k, v in counts.items() if v > 0]
    values = [v for v in counts.values() if v > 0]
    colors_map = {
        "Positive": "#4CAF50",
        "Neutral": "#9E9E9E",
        "Negative": "#F44336",
    }
    colors = [colors_map[k] for k in counts if counts[k] > 0]

    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white")
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.0f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10, "color": "black"},
    )

    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_color("white")
        autotext.set_weight("bold")

    ax.axis("equal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_scandal_timeline(company: str, rows):
    """
    Only include **HARD** items (true scandals: fraud, fines, lawsuits, settlements, etc.)
    Excludes MEDIUM (probes/scrutiny) and SOFT (revamps/layoffs) from the scandal list.
    """
    scandal_rows = [r for r in rows if r.get("level") == "HARD" and r.get("sentiment") == "Negative"]

    if not scandal_rows:
        return f"No notable enforcement/litigation red flags found for **{company}** in recent articles."

    scandal_rows.sort(
        key=lambda x: parse_to_utc(x.get("date_raw")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True
    )

    lines = []
    for row in scandal_rows:
        d = row.get("date") or ""
        lines.append(f"- **{d}** — [{row.get('title', '')}]({row.get('url', '')}) · _{row.get('source', '')}_")
    return "\n".join(lines)


# =========================
# Risk scoring (tiered)
# =========================

def compute_risk(summary_md: str, rows: list[dict]) -> str:
    """
    Tiered risk:
      - HARD events weigh heavily.
      - MEDIUM (probes/scrutiny) weigh moderately.
      - SOFT negatives mostly affect sentiment, not risk.
      - Recent (≤ 90 days) items boost their respective weights.
      - Dedup prevents echo-chamber spikes.
    """
    now = datetime.now(timezone.utc)
    unique_rows = deduplicate_articles(rows)
    total = max(len(unique_rows), 1)

    neg = 0
    hard_hits = 0
    med_hits = 0
    recent_hard = 0
    recent_med = 0

    for r in unique_rows:
        sent = r.get("sentiment", "Neutral")
        level = r.get("level", "SOFT")
        if sent == "Negative":
            neg += 1
        dt = parse_to_utc(r.get("date_raw")) or parse_to_utc(r.get("date"))

        if level == "HARD":
            hard_hits += 1
            if dt and (now - dt).days <= 90:
                recent_hard += 1
        elif level == "MEDIUM":
            med_hits += 1
            if dt and (now - dt).days <= 90:
                recent_med += 1

    neg_ratio = neg / total

    text = (summary_md or "").lower()
    sum_hard = sum(k in text for k in HARD_KEYWORDS)
    sum_med = sum(k in text for k in MEDIUM_KEYWORDS)
    sum_hard = min(sum_hard, 3)
    sum_med = min(sum_med, 3)

    # Weighted score
    score = (
            30 * neg_ratio +  # general negative
            30 * min(hard_hits, 4) +  # hard red flags
            15 * min(recent_hard, 3) +  # very recent hard
            10 * min(med_hits, 4) +  # medium red flags
            5 * min(recent_med, 3) +  # recent medium
            5 * (sum_hard + 0.5 * sum_med)
    )
    score = max(0, min(100, round(score)))

    if score < 40:
        label = "🟢 Low Risk"
    elif score < 70:
        label = "🟡 Medium Risk"
    else:
        label = "🔴 High Risk"

    return f"**Risk Score:** **{score}/100** → {label}"


# =========================
# Tags
# =========================

def extract_tags(summary: str):
    tags = []
    tag_map = {
        "#AML": ["aml", "anti-money laundering", "money laundering", "sanction", "sanctions"],
        "#Fraud": ["fraud", "scam", "embezzlement", "bribery", "corruption"],
        "#ESG": ["esg", "environment", "sustain", "governance", "social"],
        "#Reputation": ["reputation", "public image", "backlash"],
        "#Compliance": ["regulator", "compliance", "fine", "penalty", "settlement"],
        "#Litigation": ["lawsuit", "litigation", "class action", "investigation", "probe"],
    }
    text = (summary or "").lower()
    for tag, kws in tag_map.items():
        if any(k in text for k in kws):
            tags.append(tag)
    if not tags:
        tags = ["#GeneralRisk"]
    return " ".join(f"`{t}`" for t in tags)


# =========================
# Export
# =========================

def export_markdown(company: str, summary: str, rows, timeline: str, risk: str, tags: str):
    today = date.today().isoformat()
    summary_lines = [
        f"# Company Risk Report: {company}",
        f"_Generated: {today}_",
        "",
        "## 📑 Summary",
        summary or "*No summary available.*",
        "",
        "## 📊 Sentiment Snapshot",
        f"- Positive: {sum(1 for row in rows if row['sentiment'] == 'Positive')}",
        f"- Neutral: {sum(1 for row in rows if row['sentiment'] == 'Neutral')}",
        f"- Negative: {sum(1 for row in rows if row['sentiment'] == 'Negative')}",
        "",
        "### Recent Articles",
    ]
    if rows:
        for row in rows:
            summary_lines.append(
                f"- **{row.get('date', '')}** · _{row.get('source', '')}_ — "
                f"[{row.get('title', '')}]({row.get('url', '')}) · **{row.get('sentiment', '')}**"
            )
    else:
        summary_lines.append("_No recent articles._")

    summary_lines += [
        "",
        "## 📰 Scandal Timeline",
        timeline or "_No items._",
        "",
        "## ⚠️ Risk Assessment",
        risk or "_N/A_",
        "",
        "## 🏷️ Tags",
        tags or "_N/A_",
        "",
    ]

    content = "\n".join(summary_lines)
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
    gr.Markdown(
        "# 🏢 Company Risk Analyzer\nAnalyze potential **red flags** for business partnerships.",
        elem_id="header"
    )

    with gr.Row(elem_id="company-section"):
        company_name = gr.Textbox(label="🔍 Company Name", placeholder="e.g., HSBC, Tesla, Binance...", lines=1)
    start_button = gr.Button("🚀 Start Analysis", variant="primary")

    with gr.Tabs():
        with gr.Tab("📑 Summary"):
            gr.Markdown("📑 Summary of Findings", elem_classes="summary-header")
            summary_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("📊 Sentiment"):
            sentiment_img = gr.Image(type="filepath", label="Sentiment Distribution (Recent News)")
            headlines_tbl = gr.Dataframe(
                headers=["date", "source", "title", "sentiment", "url"],
                interactive=False,
                wrap=True
            )

        with gr.Tab("📰 Scandals"):
            timeline_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("⚠️ Risk"):
            risk_md = gr.Markdown("", elem_id="result-box")
            tags_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("📄 Export"):
            export_btn = gr.Button("Generate Markdown Report")
            export_file = gr.File(label="Download Report (.md)")


    # =========================
    # Orchestration
    # =========================

    def analyze(company: str):
        # 1) Fetch (relevance-filtered + deduped + adjusted sentiments + tiered levels)
        rows, error = fetch_news(company)

        # 2) Summary
        summary = ask_agent(company, rows)

        # 3) Visualization + timeline
        if error:
            pie_path = None
            timeline = f"_News error_: {error}"
        else:
            buf = make_sentiment_pie(rows)
            tmp_dir = tempfile.gettempdir()
            pie_path = os.path.join(tmp_dir, f"{company.replace(' ', '_')}_sentiment.png")
            with open(pie_path, "wb") as f:
                f.write(buf.read())
            timeline = build_scandal_timeline(company, rows)

        # 4) Risk + tags
        risk = compute_risk(summary, rows)
        tags = extract_tags(summary)

        # 5) Table rows
        table = [
            [row.get("date", ""), row.get("source", ""), row.get("title", ""), row.get("sentiment", ""),
             row.get("url", "")]
            for row in rows
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
