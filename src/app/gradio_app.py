import os
import re
import json
import datetime
import tempfile
from urllib.parse import urlparse

import gradio as gr
import matplotlib.pyplot as plt
from dotenv import load_dotenv

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


from app.agent import ask_agent

# =========================
# Helpers
# =========================

def _domain(u: str) -> str:
    try:
        return urlparse(u).netloc.replace("www.", "")
    except Exception:
        return u or ""

def decision_banner(data: dict) -> str:
    """Show decision only (no justification)."""
    fa = (data or {}).get("final_assessment") or {}
    decision = fa.get("decision", "Insufficient Data")
    emoji = {"Go": "🟢", "Conditional Go": "🟡", "No Go": "🔴", "Insufficient Data": "⚪️"}.get(decision, "⚪️")
    return f"### {emoji} Decision: **{decision}**"

def build_risks_overview_md(data: dict) -> str:
    """
    Count & list risks/weaknesses only.
    (No Legal/Regulatory or Controversies here to avoid duplication.)
    """
    risks = (data or {}).get("risks") or []
    weaknesses = (data or {}).get("weaknesses") or []

    lines = [
        "### Risks Overview",
        f"- **Explicit risks:** {len(risks)}",
        f"- **Weaknesses:** {len(weaknesses)}",
        "",
    ]

    if risks:
        lines.append("#### Risks (model-extracted)")
        for r in risks:
            lines.append(f"- {str(r).strip()}")

    if weaknesses:
        lines.append("")
        lines.append("#### Weaknesses")
        for w in weaknesses:
            lines.append(f"- {str(w).strip()}")

    return "\n".join(lines) if lines else "### Risks Overview\n_Not available._"

def _extract_first_json(text: str):
    """
    Extract the FIRST JSON object at the start of the model output.
    Supports both plain and ```json fenced blocks.
    Returns (obj, remaining_markdown) or (None, text) if not found/parsable.
    """
    s = (text or "").lstrip()
    if not s:
        return None, text


    m = re.match(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if m:
        js = m.group(1); rest = s[m.end():].strip()
        try:
            return json.loads(js), rest
        except Exception:
            return None, text

    start = s.find("{")
    if start == -1:
        return None, text

    depth = 0; in_str = False; esc = False; end_idx = None
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
    if end_idx:
        js = s[start:end_idx]; rest = s[end_idx:].strip()
        try:
            return json.loads(js), rest
        except Exception:
            return None, text
    return None, text

def extract_sources_from_markdown(summary_md: str, max_items: int = 30) -> list[str]:
    """Collect URLs from Markdown, dedupe, return list."""
    text = summary_md or ""
    urls = []
    urls += re.findall(r"\[[^\]]*\]\((https?://[^)]+)\)", text)
    urls += re.findall(r"(https?://[^\s)]+)", text)

    seen, uniq = set(), []
    for u in urls:
        u = u.strip()
        if u and u not in seen:
            seen.add(u); uniq.append(u)
        if len(uniq) >= max_items:
            break
    return uniq

def md_sources_list(urls: list[str]) -> str:
    if not urls:
        return "_No sources detected._"
    return "\n".join(f"- [{_domain(u)}]({u})" for u in urls)

def block_md_list(title: str, lines: list[str]) -> str:
    body = "\n".join(f"- {line}" for line in (lines or [])) if lines else "_Not available._"
    return f"### {title}\n{body}"

def build_profile_md(profile: dict) -> str:
    if not isinstance(profile, dict):
        return block_md_list("Company Profile", [])
    lines = [
        f"**Industry:** {profile.get('industry','') or 'Not available'}",
        f"**Size:** {profile.get('size','') or 'Not available'}",
        f"**HQ Location:** {profile.get('hq_location','') or 'Not available'}",
        f"**Founded:** {profile.get('founded_year','') or 'Not available'}",
        f"**Description:** {profile.get('description','') or 'Not available'}",
    ]
    return block_md_list("Company Profile", lines)

def build_financials_md(fin: dict) -> str:
    if not isinstance(fin, dict):
        return block_md_list("Financials", [])
    lines = [
        f"**Revenue:** {fin.get('revenue','') or 'Not available'}",
        f"**Profitability:** {fin.get('profitability','') or 'Not available'}",
        f"**Growth:** {fin.get('growth','') or 'Not available'}",
        f"**Notes:** {fin.get('notes','') or 'Not available'}",
    ]
    return block_md_list("Financials", lines)

def build_leadership_md(arr: list[dict]) -> str:
    lines = []
    for x in arr or []:
        name = str(x.get("name","")).strip()
        role = str(x.get("role","")).strip()
        notes = str(x.get("notes","")).strip()
        lines.append(f"**{name}** — {role}" + (f" — {notes}" if notes else ""))
    return block_md_list("Leadership & Ownership", lines)

def build_legal_md(arr: list[dict]) -> str:
    lines = []
    for x in arr or []:
        date = str(x.get("date","")).strip()
        issue = str(x.get("issue","")).strip()
        status = str(x.get("status","")).strip()
        src = str(x.get("source","")).strip()
        link = f" — [{_domain(src)}]({src})" if src else ""
        lines.append(f"**{date}** — {issue} — _{status}_{link}")
    return block_md_list("Legal & Regulatory", lines)

def build_partnerships_md(arr: list[dict]) -> str:
    lines = []
    for x in arr or []:
        nm = str(x.get("name","")).strip()
        typ = str(x.get("type","")).strip()
        notes = str(x.get("notes","")).strip()
        lines.append(f"**{nm}**" + (f" ({typ})" if typ else "") + (f" — {notes}" if notes else ""))
    return block_md_list("Partnerships & Clients", lines)

def build_controversies_md(contro_arr: list[dict], legal_arr: list[dict], rows: list[dict], leadership: list[dict]) -> str:
    """
    Prefer JSON 'controversies'; else derive from legal items; else from negative news.
    No text trimming.
    """
    lines = []

    # 1) JSON controversies
    if contro_arr:
        for c in contro_arr:
            date = str(c.get("date","")).strip()
            entity = str(c.get("entity","")).strip()
            etype = str(c.get("entity_type","")).strip()
            title = str(c.get("title","")).strip()
            summ = str(c.get("summary","")).strip()
            src = str(c.get("source","")).strip()
            link = f" — [{_domain(src)}]({src})" if src else ""
            tail = f" — _{etype}_" if etype else ""
            lines.append(
                f"**{date}** — **{entity}**{tail} — {title}"
                + (f" — {summ}" if summ else "")
                + link
            )

    # 2) Fallback: legal/regulatory
    if not lines and legal_arr:
        for x in legal_arr:
            date = str(x.get("date","")).strip()
            issue = str(x.get("issue","")).strip()
            status = str(x.get("status","")).strip()
            src = str(x.get("source","")).strip()
            link = f" — [{_domain(src)}]({src})" if src else ""
            lines.append(f"**{date}** — **Company** — {issue} — _{status}_{link}")

    # 3) Fallback: negative headlines
    if not lines and rows:
        leader_names = [str(l.get("name","")).strip() for l in (leadership or []) if isinstance(l, dict)]
        for r in rows:
            if (str(r.get("sentiment","")).strip().lower()) == "negative":
                title = str(r.get("title","")).strip()
                entity = "Company"
                etype = "Company"
                t_low = title.lower()
                for name in leader_names:
                    if name and name.lower() in t_low:
                        entity = name; etype = "Leader"; break
                src = str(r.get("url","")).strip()
                link = f" — [{_domain(src)}]({src})" if src else ""
                lines.append(f"**{str(r.get('date','')).strip()}** — **{entity}** — {title}{link}")

    return block_md_list("Controversies", lines)

def build_news_md(json_news: list[dict], rows: list[dict]) -> str:
    items = json_news if json_news else rows
    lines = []
    for r in items or []:
        date = str(r.get("date","")).strip()
        title = str(r.get("title","")).strip()
        src_name = _domain(str(r.get("url","")).strip()) if r.get("url") else str(r.get("source","")).strip()
        url = str(r.get("url","")).strip()
        link = f"[{src_name}]({url})" if url else (src_name or "")
        lines.append(f"**{date}** — {title}" + (f" — {link}" if link else ""))
    return block_md_list("Recent Articles", lines)

def sources_from_both(data: dict, markdown_summary: str) -> list[str]:
    urls = []
    if isinstance(data, dict):
        urls += [u for u in (data.get("sources") or []) if isinstance(u, str)]
    if not urls:
        text = markdown_summary or ""
        urls += re.findall(r"\[[^\]]*\]\((https?://[^)]+)\)", text)
        urls += re.findall(r"(https?://[^\s)]+)", text)
    # de-dup
    seen, dedup = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u); dedup.append(u)
    return dedup

# Stub: you can wire your real news tools later if you want
def fetch_news_combined(company: str):
    return [], None


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
        company_name = gr.Textbox(label="🔍 Company Name", placeholder="e.g., HSBC, Yamaha, Binance...", lines=1)
    start_button = gr.Button("🚀 Start Analysis", variant="primary")

    with gr.Tabs():
        with gr.Tab("📑 Summary"):
            summary_hdr = gr.Markdown("📑 Summary of Findings", elem_classes="summary-header")
            decision_md = gr.Markdown("", elem_id="result-box")
            summary_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("🏷️ Company Profile"):
            profile_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("💹 Financials"):
            financials_md = gr.Markdown("", elem_id="result-box")


        with gr.Tab("👤 Leadership & Ownership"):
            leadership_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("⚖️ Legal & Regulatory"):
            legal_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("🤝 Partnerships & Clients"):
            partnerships_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("🚨 Controversies"):
            controversies_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("🗞️ News"):
            news_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("🔗 Sources"):
            sources_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("⚠️ Risks Overview"):
            risks_overview_md = gr.Markdown("", elem_id="result-box")

        with gr.Tab("📄 Export"):
            export_btn = gr.Button("Generate Markdown Report")
            export_file = gr.File(label="Download Report (.md)")


    # =========================
    # Orchestration
    # =========================

    def analyze(company: str):
        rows, _ = fetch_news_combined(company)  # usually []

        full_output = ask_agent(company, rows)
        data, markdown_summary = _extract_first_json(full_output)
        data = data if isinstance(data, dict) else {}

        # Decision banner (top of Summary)
        decision_block = decision_banner(data)

        # Risks Overview (counts + lists; no duplicates with other tabs)
        risks_overview_block = build_risks_overview_md(data)

        # Structured sections from JSON
        profile = data.get("company_profile", {}) or {}
        fin = data.get("financials", {}) or {}
        leadership = data.get("leadership_and_ownership", []) or []
        legal = data.get("legal_and_regulatory", []) or []
        partners = data.get("partnerships_and_clients", []) or []
        json_news = data.get("recent_news", []) or []
        controversies = data.get("controversies", []) or []

        # Markdown sections (no truncation)
        profile_block = build_profile_md(profile)
        financials_block = build_financials_md(fin)
        leadership_block = build_leadership_md(leadership)
        legal_block = build_legal_md(legal)
        partnerships_block = build_partnerships_md(partners)
        controversies_block = build_controversies_md(controversies, legal, rows, leadership)
        news_block = build_news_md(json_news, rows)

        # Sources
        src_urls = sources_from_both(data, markdown_summary)
        sources_block = md_sources_list(src_urls)

        # Return in UI order
        return (
            markdown_summary,      # summary_md
            decision_block,        # decision_md
            profile_block,
            financials_block,
            leadership_block,
            legal_block,
            partnerships_block,
            controversies_block,
            news_block,
            sources_block,
            risks_overview_block   # risks_overview_md
        )


    start_button.click(
        fn=analyze,
        inputs=[company_name],
        outputs=[
            summary_md,
            decision_md,
            profile_md,
            financials_md,
            leadership_md,
            legal_md,
            partnerships_md,
            controversies_md,
            news_md,
            sources_md,
            risks_overview_md
        ],
        show_progress="full"
    )

    def export_markdown(
        company: str,
        summary_block: str,
        decision_block: str,
        profile_block: str,
        financials_block: str,
        leadership_block: str,
        legal_block: str,
        partnerships_block: str,
        controversies_block: str,
        news_block: str,
        sources_block: str,
        risks_overview_block: str,
    ):
        today = datetime.date.today().isoformat()
        md = [
            f"# Company Risk Report: {company}",
            f"_Generated: {today}_",
            "",
            "## ✅ Decision",
            decision_block or "_No decision available._",
            "",
            "## 📑 Summary",
            summary_block or "*No summary available.*",
            "",
            profile_block or "## 🏷️ Company Profile\n_Not available._",
            "",
            financials_block or "## 💹 Financials\n_Not available._",
            "",
            leadership_block or "## 👤 Leadership & Ownership\n_Not available._",
            "",
            legal_block or "## ⚖️ Legal & Regulatory\n_Not available._",
            "",
            partnerships_block or "## 🤝 Partnerships & Clients\n_Not available._",
            "",
            controversies_block or "## 🚨 Controversies\n_Not available._",
            "",
            news_block or "## 🗞️ News\n_Not available._",
            "",
            "## 🔗 Sources",
            sources_block or "_No sources detected._",
            "",
            "## ⚠️ Risks Overview",
            risks_overview_block or "_Not available._",
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
            decision_md,
            profile_md,
            financials_md,
            leadership_md,
            legal_md,
            partnerships_md,
            controversies_md,
            news_md,
            sources_md,
            risks_overview_md,
        ],
        outputs=[export_file],
        show_progress=True
    )


def main():
    app.launch()


if __name__ == "__main__":
    main()
