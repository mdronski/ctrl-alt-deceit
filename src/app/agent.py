# src/app/agent.py
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# --- Force AI Studio API-key auth; disable ADC/GCloud envs ---
API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDUQ-EdjWtLZzvHwcFmEt0skdlFE-VRSwk"
os.environ["GOOGLE_API_KEY"] = API_KEY
for var in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_AUTH_TOKEN", "GOOGLE_CLOUD_PROJECT"):
    os.environ.pop(var, None)

# -------- System + Developer prompts (global) --------
SYSTEM_PROMPT = """You are a due-diligence analyst for HSBC.
Your role is to evaluate potential business partners.
Always act conservatively and prioritize risk awareness.
Use only credible, verifiable information. Never fabricate details.
If information is missing or uncertain, explicitly state “Not available” or “Unconfirmed + reason”.

Scope of the review (ALWAYS cover each item):
1) Company activity: products, markets, clients, licensing, compliance history.
2) Leadership/owners: backgrounds, other ventures, investments, board seats, conflicts of interest.
3) Legal/regulatory: investigations, fines, sanctions, lawsuits, settlements.
4) Reputation & ESG: safety/privacy, labor, environmental, community impact, misinformation/content issues.
5) Sensitive ties: defense/dual-use, surveillance, spyware, government contracts, lobbying/political spend.

Rules:
- Separate “Company activity” from “Leader personal activity” and label each clearly.
- If you suspect a notable controversy exists but cannot verify it now, mark it “Unconfirmed” and add a suggested query in research_gaps.
- Prefer concise bullet points; avoid speculation or editorializing.
"""


DEVELOPER_PROMPT = """Return output in two parts, in this exact order:

1) A SINGLE JSON object with these fields (keep them even if empty; extra fields are allowed):

{
  "company_profile": {
    "industry": "",
    "size": "",
    "hq_location": "",
    "founded_year": "",
    "description": ""
  },
  "financials": {
    "revenue": "",
    "profitability": "",
    "growth": "",
    "notes": ""
  },
  "leadership_and_ownership": [
    { "name": "", "role": "", "notes": "" }
  ],
  "reputation": {
    "overall": "",
    "positives": [],
    "negatives": []
  },
  "legal_and_regulatory": [
    { "issue": "", "status": "", "date": "", "source": "" }
  ],
  "partnerships_and_clients": [
    { "name": "", "type": "", "notes": "" }
  ],
  "recent_news": [
    { "date": "", "title": "", "source": "", "url": "", "sentiment": "" }
  ],
  "controversies": [
    {
      "date": "",
      "title": "",
      "entity": "",
      "entity_type": "Company|Leader",
      "summary": "",
      "source": "",
      "linkage": "Direct|Indirect",
      "materiality": "Low|Medium|High",
      "status": "Allegation|Ongoing|Confirmed|Resolved|Unconfirmed"
    }
  ],
  "strengths": [],
  "weaknesses": [],
  "risks": [],
  "final_assessment": {
    "decision": "Go|Conditional Go|No Go|Insufficient Data",
    "justification": "",
    "confidence": "Low|Medium|High",
    "key_factors": []
  },
  "sources": [],
  "research_gaps": [
    { "query": "", "why": "" }
  ],
  "aliases": {
    "people": [],
    "brands": []
  },
  "timeframe": ""
}

Required behavior:
- Keep values concise and factual. If unknown, use "" or [] (do NOT invent).
- For any legal/controversy item without a verifiable URL, set "status":"Unconfirmed", leave "source":"", and add a related entry in research_gaps with a concrete query to run later.
- In controversies:
  - Set "entity_type" to "Leader" for personal ventures (e.g., defense/dual-use investments) and "Company" for corporate actions.
  - Set "linkage" to "Indirect" when it’s a leader’s personal activity, and "Direct" for company actions.
  - Provide a reasoned "materiality" (Low/Medium/High) based on potential impact on HSBC counterpart risk.

2) A Markdown summary for a human reader:
- Use short sections and bullets.
- Explicitly split “Company activity” vs “Leader personal activity”.
- Include inline links next to claims when available.
- End with a top-level "Sources" section listing up to 20 canonical URLs as markdown links (deduplicated).

Do not include any prose before the JSON. Do not wrap the JSON in extra text. Ensure valid JSON.
When evidence is insufficient, prefer "Insufficient Data" over "No Go" and add targeted research_gaps with specific queries.
"""

# Single model instance (no tools)
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=os.environ["GOOGLE_API_KEY"],
)

def _coerce_content_to_text(content) -> str:
    """Some providers return content as a list of parts; normalize to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("content"), str):
                    parts.append(p["content"])
                else:
                    parts.append(str(p))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)

def _build_user_prompt(company_name: str, rows: list[dict]) -> str:
    base = f"""
Gather all publicly available and credible information about the target company "{company_name}" to assess whether HSBC should consider them a reliable business partner.

Include the following categories:
- Basic facts (industry, size, location, year founded).
- Financial performance and growth trends.
- Leadership and ownership details with any other ventures, investments, and board seats.
- Reputation (reviews, press, social media sentiment).
- Legal or regulatory issues (lawsuits, sanctions, investigations).
- Partnerships and major clients.
- Recent news and controversies.

Leadership controversy probe (ALWAYS run this sweep):
- Check for defense/dual-use or military-related ventures or investments (e.g., Helsing, Anduril, Palantir, NSO Group).
- Check for surveillance/spyware, privacy controversies, or extremist/violent content links.
- Check for lobbying/political spending, major philanthropic vehicles, and conflicts of interest.
- Consider alternate spellings and languages (e.g., defence vs defense; diacritics).
- If you suspect a relevant issue (e.g., a leader investing in a defense AI startup) but cannot verify it now, mark the item as "Unconfirmed" and add a "research_gaps" entry with concrete queries (e.g., "<Leader Name> investment Helsing site:reuters.com OR site:ft.com").

Summarize findings into strengths, weaknesses, risks, and a final assessment for HSBC.

Prepare the output as per the required JSON schema, followed by a Markdown summary that ends with a Sources section.
""".strip()

    if rows:
        news_text = "\n".join(
            f"- {r.get('date','Not available')} · {r.get('source','Not available')} — "
            f"{r.get('title','Not available')} ({r.get('sentiment','Not available')})\n  "
            f"{r.get('url','Not available')}"
            for r in rows
        )
        base += f"\n\nProvided recent articles (may be incomplete):\n{news_text}"
    else:
        base += "\n\nNo recent articles were provided."

    return base

def ask_agent(company_name: str, rows: list[dict]) -> str:
    """Prompt-only analysis; no tools are used."""
    user_prompt = _build_user_prompt(company_name, rows or [])
    resp = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "name": "developer", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    return _coerce_content_to_text(resp.content)