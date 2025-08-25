# src/app/gradio_app.py
import os
import io
import re
import json
import datetime
import tempfile
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()

from app.agent import ask_agent

# =========================
# Helpers
# =========================

def _extract_first_json(text: str):
    """
    Extract the FIRST JSON object at the start of the model output.
    Supports both plain and ```json fenced blocks.
    Returns (obj, remaining_markdown) or (None, text) if not found/parsable.
    """
    s = (text or "").lstrip()
    if not s:
        return None, text

    if s.startswith("```"):
        m = re.match(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
        if m:
            json_str = m.group(1)
            rest = s[m.end():].strip()
            try:
                return json.loads(json_str), rest
            except Exception:
                return None, text

    start = s.find("{")
    if start == -1:
        return None, text

    depth = 0
    in_str = False
    esc = False
    end_idx = None
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
    if end_idx:
        json_str = s[start:end_idx]
        rest = s[end_idx:].strip()
        try:
            return json.loads(json_str), rest
        except Exception:
            return None, text
    return None, text


def extract_sources_from_markdown(summary_md: str, max_items: int = 30) -> str:
    """Collect URLs from Markdown, dedupe, render as bullet list."""
    text = summary_md or ""
    urls = []

    for m in re.findall(r"\[[^\]]*\]\((https?://[^)]+)\)", text):
        urls.append(m.strip())
    for m in re.findall(r"(https?://[^\s)]+)", text):
        urls.append(m.strip())

    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
        if len(uniq) >= max_items:
            break

    if not uniq:
        return "_No sources detected._"

    bullets = []
    for u in uniq:
        try:
            domain = urlparse(u).netloc.replace("www.", "")
        except Exception:
            domain = u
        bullets.append(f"- [{domain}]({u})")
    return "\n".join(bullets)

# =========================
# Existing news helpers (kept; we still show recent news)
# =========================

def fetch_news_combined(company: str):
    tools = []  # no external tools; left for future
    articles_all = []
    errors = []

    for tool_fn in tools:
        try:
            result = tool_fn(company)
            if "No articles found" in result:
                continue
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


def build_scandal_timeline(company: str, rows):
    negatives = [r for r in rows if r.get("sentiment") == "Negative"]
    if not negatives:
        return f"No notable negative events found for **{company}** in recent articles."
    lines = []
    for r in sorted(negatives, key=lambda x: x.get("date") or "", reverse=True):
        lines.append(f"- **{r.get('date','')}** — [{r.get('title','')}]({r.get('url','')}) · _{r.get('source','')}_")
    return "\n".join(lines)


def compute_risk(company: str, summary_md: str, rows):
    neg = sum(1 for r in rows if r.get("sentiment") == "Negative")
    total = max(len(rows), 1)
    neg_ratio = neg / total

    if isinstance(summary_md, list):
        summary_md = " ".join(map(str, summary_md))
    elif not isinstance(summary_md, str):
        summary_md = str(summary_md)

    summary_lower = (summary_md or "").lower()
    red_keywords = [
        "aml", "money laundering", "sanction", "fraud", "bribe",
        "regulator", "fine", "penalty", "scandal", "lawsuit", "investigation",
        "whistleblower", "corruption", "sanctions", "non-compliance"
    ]
    kw_hits = sum(k in summary_lower for k in red_keywords)

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

def _fmt_profile_md(profile: dict) -> str:
    if not isinstance(profile, dict):
        return "_Not available_"
    return "\n".join([
        f"- **Industry:** {profile.get('industry','') or 'Not available'}",
        f"- **Size:** {profile.get('size','') or 'Not available'}",
        f"- **HQ Location:** {profile.get('hq_location','') or 'Not available'}",
        f"- **Founded:** {profile.get('founded_year','') or 'Not available'}",
        f"- **Description:** {profile.get('description','') or 'Not available'}",
    ])

def _fmt_financials_md(fin: dict) -> str:
    if not isinstance(fin, dict):
        return "_Not available_"
    return "\n".join([
        f"- **Revenue:** {fin.get('revenue','') or 'Not available'}",
        f"- **Profitability:** {fin.get('profitability','') or 'Not available'}",
        f"- **Growth:** {fin.get('growth','') or 'Not available'}",
        f"- **Notes:** {fin.get('notes','') or 'Not available'}",
    ])

def _build_controversies_rows(data, rows, leadership):
    """
    Create controversies table rows:
    1) Prefer JSON 'controversies'
    2) Else, derive from legal_and_regulatory + negative news
    3) Try to tag a leader if their name appears in a headline
    """
    out = []

    # 1) JSON controversies
    if isinstance(data, dict):
        j = data.get("controversies", [])
        if j:
            for c in j:
                out.append([
                    (c.get("date") or ""),
                    (c.get("entity") or ""),
                    (c.get("entity_type") or ""),
                    (c.get("title") or ""),
                    (c.get("summary") or ""),
                    (c.get("source") or ""),
                ])

    # 2) Fallbacks
    if not out and isinstance(data, dict):
        # legal_and_regulatory
        for x in (data.get("legal_and_regulatory") or []):
            out.append([
                (x.get("date") or ""),
                "Company",
                "Company",
                (x.get("issue") or ""),
                (x.get("status") or ""),
                (x.get("source") or "")
            ])

    # negative news from rows
    if not out:
        leader_names = [l.get("name","") for l in (leadership or []) if isinstance(l, dict)]
        for r in rows:
            if r.get("sentiment") == "Negative":
                title = r.get("title","") or ""
                entity = "Company"
                entity_type = "Company"
                title_lower = title.lower()
                for name in leader_names:
                    if name and name.lower() in title_lower:
                        entity = name
                        entity_type = "Leader"
                        break
                out.append([
                    r.get("date",""),
                    entity,
                    entity_type,
                    title,
                    f"{r.get('source','')} (neg. mention)",
                    r.get("url","")
                ])

    # Deduplicate (date+title+source)
    seen = set()
    dedup = []
    for row in out:
        key = (row[0], row[3], row[5])
        if key not in seen:
            seen.add(key)
            dedup.append(row)
    return dedup

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
            summary_hdr = gr.Markdown("📑 Summary of Findings", elem_classes="summary-header")
            summary_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("🏷️ Company Profile"):
            profile_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("💹 Financials"):
            financials_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("👤 Leadership & Ownership"):
            leadership_tbl = gr.Dataframe(headers=["name", "role", "notes"], interactive=False)

        with gr.Tab("⚖️ Legal & Regulatory"):
            legal_tbl = gr.Dataframe(headers=["date", "issue", "status", "source"], interactive=False)

        with gr.Tab("🤝 Partnerships & Clients"):
            partnerships_tbl = gr.Dataframe(headers=["name", "type", "notes"], interactive=False)

        # 🚨 NEW: Controversies tab (replaces Sentiment)
        with gr.Tab("🚨 Controversies"):
            controversies_tbl = gr.Dataframe(
                headers=["date", "entity", "entity_type", "title", "summary", "source"],
                interactive=False
            )

        # Keep Scandals tab, now includes recent news table below the timeline
        with gr.Tab("📰 Scandals"):
            timeline_md = gr.Markdown("", elem_id="result-box")
            headlines_tbl = gr.Dataframe(headers=["date", "source", "title", "sentiment", "url"], interactive=False)

        with gr.Tab("🔗 Sources"):
            sources_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("⚠️ Risk"):
            risk_md = gr.Markdown("", elem_id="result-box")
            tags_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("📄 Export"):
            export_btn = gr.Button("Generate Markdown Report")
            export_file = gr.File(label="Download Report (.md)")

    # -------- logic
    def analyze(company: str):
        rows, err = fetch_news_combined(company)

        full_output = ask_agent(company, rows)

        data, markdown_summary = _extract_first_json(full_output)

        profile = data.get("company_profile", {}) if isinstance(data, dict) else {}
        fin = data.get("financials", {}) if isinstance(data, dict) else {}
        leadership = data.get("leadership_and_ownership", []) if isinstance(data, dict) else []
        legal = data.get("legal_and_regulatory", []) if isinstance(data, dict) else []
        partners = data.get("partnerships_and_clients", []) if isinstance(data, dict) else []
        json_recent_news = data.get("recent_news", []) if isinstance(data, dict) else []

        # Recent news table: prefer JSON; else rows
        if json_recent_news:
            headlines = [
                [r.get("date",""), r.get("source",""), r.get("title",""), r.get("sentiment",""), r.get("url","")]
                for r in json_recent_news
            ]
        else:
            headlines = [[r.get("date",""), r.get("source",""), r.get("title",""), r.get("sentiment",""), r.get("url","")] for r in rows]

        # Controversies table
        controversies_rows = _build_controversies_rows(data, rows, leadership)

        # Scandal timeline
        if err:
            timeline = f"_News error_: {err}"
        else:
            timeline = build_scandal_timeline(company, rows)

        # Risk & tags
        risk = compute_risk(company, markdown_summary, rows)
        tags = extract_tags(markdown_summary)

        # Structured markdown blocks
        profile_block = _fmt_profile_md(profile)
        financials_block = _fmt_financials_md(fin)

        # Tables (leadership / legal / partnerships)
        leadership_rows = [[x.get("name",""), x.get("role",""), x.get("notes","")] for x in (leadership or [])]
        legal_rows = [[x.get("date",""), x.get("issue",""), x.get("status",""), x.get("source","")] for x in (legal or [])]
        partner_rows = [[x.get("name",""), x.get("type",""), x.get("notes","")] for x in (partners or [])]

        # Sources: prefer JSON "sources"; else extract from markdown
        json_sources = (data or {}).get("sources") if isinstance(data, dict) else None
        if json_sources:
            dedup = []
            seen = set()
            for u in json_sources:
                if isinstance(u, str) and u and u not in seen:
                    seen.add(u); dedup.append(u)
            sources = "\n".join(f"- [{urlparse(u).netloc.replace('www.','')}]({u})" for u in dedup) if dedup else "_No sources detected._"
        else:
            sources = extract_sources_from_markdown(markdown_summary)

        # Return in the order of UI outputs
        return (
            markdown_summary,           # summary_md
            profile_block,              # profile_md
            financials_block,           # financials_md
            leadership_rows,            # leadership_tbl
            legal_rows,                 # legal_tbl
            partner_rows,               # partnerships_tbl
            controversies_rows,         # controversies_tbl
            timeline,                   # timeline_md
            headlines,                  # headlines_tbl
            sources,                    # sources_md
            risk,                       # risk_md
            tags                        # tags_md
        )

    start_button.click(
        fn=analyze,
        inputs=[company_name],
        outputs=[
            summary_md,
            profile_md,
            financials_md,
            leadership_tbl,
            legal_tbl,
            partnerships_tbl,
            controversies_tbl,  # NEW
            timeline_md,
            headlines_tbl,
            sources_md,
            risk_md,
            tags_md
        ],
        show_progress="full"
    )

    def export_markdown(
        company: str,
        summary: str,
        headlines_df,
        profile_block: str,
        financials_block: str,
        leadership_df,
        legal_df,
        partners_df,
        controversies_df,
        timeline_md: str,
        sources_md: str,
        risk_md: str,
        tags_md: str
    ):
        def df_to_rows(df):
            rows = []
            if df is not None and hasattr(df, "itertuples"):
                for row in df.itertuples(index=False):
                    rows.append([*row])
            return rows

        headlines_rows = df_to_rows(headlines_df)
        leadership_rows = df_to_rows(leadership_df)
        legal_rows = df_to_rows(legal_df)
        partners_rows = df_to_rows(partners_df)
        controversies_rows = df_to_rows(controversies_df)

        today = datetime.date.today().isoformat()
        md = [
            f"# Company Risk Report: {company}",
            f"_Generated: {today}_",
            "",
            "## 📑 Summary",
            summary or "*No summary available.*",
            "",
            "## 🏷️ Company Profile",
            profile_block or "_Not available_",
            "",
            "## 💹 Financials",
            financials_block or "_Not available_",
            "",
            "## 👤 Leadership & Ownership",
        ]
        md += ["- " + " — ".join([c for c in r if c]) for r in leadership_rows] or ["_Not available_"]
        md += [
            "",
            "## ⚖️ Legal & Regulatory",
        ]
        md += ["- " + " — ".join([c for c in r if c]) for r in legal_rows] or ["_Not available_"]
        md += [
            "",
            "## 🤝 Partnerships & Clients",
        ]
        md += ["- " + " — ".join([c for c in r if c]) for r in partners_rows] or ["_Not available_"]
        md += [
            "",
            "## 🚨 Controversies",
        ]
        if controversies_rows:
            for r in controversies_rows:
                date, entity, entity_type, title, summary_c, source = (r + [""]*6)[:6]
                bullet = f"- **{date}** — **{entity}** ({entity_type}) — {title}"
                if summary_c:
                    bullet += f" — {summary_c}"
                if source:
                    bullet += f" — [{urlparse(source).netloc.replace('www.','')}]({source})"
                md.append(bullet)
        else:
            md.append("_No controversies found._")

        md += [
            "",
            "## 🗞️ Recent Articles",
        ]
        if headlines_rows:
            md += [f"- **{r[0]}** · _{r[1]}_ — [{r[2]}]({r[4]}) · **{r[3]}**" for r in headlines_rows]
        else:
            md += ["_No recent articles._"]

        md += [
            "",
            "## 📰 Scandal Timeline",
            timeline_md or "_No items._",
            "",
            "## 🔗 Sources",
            sources_md or "_No sources detected._",
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

    export_btn.click(
        fn=export_markdown,
        inputs=[
            company_name,
            summary_md,
            headlines_tbl,
            profile_md,
            financials_md,
            leadership_tbl,
            legal_tbl,
            partnerships_tbl,
            controversies_tbl,  # NEW
            timeline_md,
            sources_md,
            risk_md,
            tags_md
        ],
        outputs=[export_file],
        show_progress=True
    )

def main():
    app.launch()

if __name__ == "__main__":
    main()
