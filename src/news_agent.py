"""Daily 10:00 AM Almaty (UTC+5) AI News Intelligence & Credibility Agent.

Responsibilities:
1. Fetch daily AI infrastructure, circular financing, data-center credit, and tracked-company news
   via RSS feeds (or optional NewsAPI) when run on schedule (.github/workflows/daily_news_agent.yml
   at 05:00 UTC = 10:00 Asia/Almaty) or manually via `python -m src.news_agent`.
2. Classify every source into a strict 3-tier credibility rubric:
   - Tier 1 (Verified Institutional / Primary): SEC, company IR, Reuters, Bloomberg, WSJ, FT, J.P. Morgan
   - Tier 2 (Trade / Tech Press — Verify): The Information, CNBC, Data Center Dynamics, TechCrunch
   - Tier 3 (Unverified / Rumor / Tabloid — Flagged): anonymous blogs, social rumors, unverified outlets
3. Filter for topic relevance, score financial sentiment (-1.0 to +1.0), and produce an analytical
   summary of how each credible item shifts company contagion risk (separate from pure quant numbers).
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

ALMATY_TZ = timezone(timedelta(hours=5))
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FEED_PATH = BASE_DIR / "data" / "news_feed.json"

TIER1_DOMAINS = {
    "sec.gov", "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "jpmorganmarkets.com", "jpmorgan.com", "openai.com", "anthropic.com",
    "nvidianews.nvidia.com", "investor.nvidia.com", "investors.broadcom.com",
    "newsroom.amd.com", "microsoft.com", "aboutamazon.com", "abc.xyz",
    "investor.atmeta.com", "investors.coreweave.com", "investor.oracle.com",
}

TIER2_DOMAINS = {
    "cnbc.com", "theinformation.com", "datacenterdynamics.com",
    "techcrunch.com", "semianalysis.com", "barrons.com", "marketwatch.com",
    "geekwire.com", "scmp.com",
}

RUMOR_KEYWORDS = {
    "rumor", "rumour", "unconfirmed", "anonymous post", "leaked tweet",
    "reddit", "speculation", "gossip", "insider claims without evidence",
    "tabloid", "benzinga", "seekingalpha", "zerohedge",
}

TOPIC_KEYWORDS = {
    "circular financing", "vendor financing", "data center", "gpu", "ai capex",
    "credit spread", "high yield", "moratorium", "backstop", "guarantee",
    "spv", "neocloud", "ipo", "s-1", "debt", "bond", "lease", "compute",
    "chip", "datacenter", "power", "antitrust", "default", "downgrade",
}

ENTITY_ALIASES = {
    "C001": ["nvidia", "nvda"],
    "C002": ["microsoft", "msft", "azure"],
    "C003": ["oracle", "orcl", "oci"],
    "C004": ["amd", "advanced micro devices"],
    "C005": ["coreweave", "crwv"],
    "C006": ["openai", "chatgpt"],
    "C007": ["amazon", "amzn", "aws"],
    "C008": ["alphabet", "google", "googl"],
    "C009": ["meta", "facebook"],
    "C010": ["anthropic", "claude"],
    "C011": ["broadcom", "avgo", "hock tan"],
    "C012": ["sb energy", "softbank"],
    "C013": ["fluidstack"],
    "C014": ["applied digital", "apld"],
    "C015": ["galaxy digital", "glxy", "helios"],
    "C016": ["blackstone", "bx"],
}

NEGATIVE_LEXICON = {
    "moratorium": -0.6, "default": -0.9, "distressed": -0.8, "widened": -0.45,
    "selloff": -0.5, "tumbled": -0.55, "plunge": -0.65, "crash": -0.8,
    "bubble": -0.5, "fear": -0.4, "delay": -0.45, "delays": -0.45,
    "ban": -0.7, "backlash": -0.5, "crowding out": -0.35, "downgrade": -0.75,
    "risk": -0.25, "nervous": -0.35, "wary": -0.35, "reluctant": -0.4,
    "loss": -0.45, "losing": -0.4, "investigation": -0.55, "subpoena": -0.6,
    "cutback": -0.6, "pause": -0.45, "pullback": -0.4,
}

POSITIVE_LEXICON = {
    "approved": 0.45, "rallied": 0.45, "strong demand": 0.5, "validated": 0.35,
    "insulated": 0.35, "relief": 0.3, "stabilize": 0.35, "outperforming": 0.4,
    "surpassed": 0.4, "tighter": 0.35, "growth": 0.25, "upgrade": 0.6,
}


def classify_source_credibility(url: str, source_name: str, text: str) -> tuple[str, str, bool]:
    """Return (tier_label, badge, include_in_base_model)."""
    host = urllib.parse.urlparse(url or "").netloc.lower().replace("www.", "")
    combined = f"{source_name} {text}".lower()

    if any(k in combined for k in RUMOR_KEYWORDS):
        return "Tier 3 — Unverified / Rumor", "⚠️ RUMOR / UNVERIFIED", False
    if any(host.endswith(d) for d in TIER1_DOMAINS) or any(
        s in source_name.lower() for s in ("j.p. morgan", "wall street journal", "wsj", "bloomberg", "reuters", "financial times", "sec")
    ):
        return "Tier 1 — Verified Primary / Institutional", "✅ VERIFIED", True
    if any(host.endswith(d) for d in TIER2_DOMAINS):
        return "Tier 2 — Trade / Financial Press", "ℹ️ TRADE PRESS", True
    return "Tier 3 — Unverified / Rumor", "⚠️ UNVERIFIED SOURCE", False


def score_financial_sentiment(text: str) -> float:
    """Deterministic financial sentiment scorer in [-1.0, +1.0] (transparent, zero-key)."""
    low = text.lower()
    score = 0.0
    hits = 0
    for term, weight in NEGATIVE_LEXICON.items():
        if term in low:
            score += weight
            hits += 1
    for term, weight in POSITIVE_LEXICON.items():
        if term in low:
            score += weight
            hits += 1
    if hits == 0:
        return 0.0
    return round(max(-1.0, min(1.0, score / max(1.0, hits ** 0.6))), 2)


def detect_entities(text: str) -> list[str]:
    low = text.lower()
    found = []
    for cid, aliases in ENTITY_ALIASES.items():
        if any(re.search(rf"\b{re.escape(a)}\b", low) for a in aliases):
            found.append(cid)
    return found


def is_topic_relevant(text: str) -> bool:
    low = text.lower()
    has_topic = any(k in low for k in TOPIC_KEYWORDS)
    has_entity = len(detect_entities(low)) > 0
    return has_topic and has_entity


def fetch_daily_rss_news(queries: list[str] | None = None, max_per_query: int = 8) -> list[dict]:
    """Fetch fresh headlines from Google News RSS (runs live in GitHub Actions at 10:00 Almaty)."""
    if queries is None:
        queries = [
            "AI circular financing NVIDIA OpenAI Anthropic Broadcom",
            "data center credit spread CoreWeave hyperscaler debt",
            "data center moratorium Texas Ohio Georgia AI capex",
        ]
    items = []
    seen_titles = set()
    now_almaty = datetime.now(ALMATY_TZ).isoformat()

    for q in queries:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=en-US&gl=US&ceid=US:en"
        try:
            resp = requests.get(url, timeout=12, headers={"User-Agent": "CircularAlphaNewsAgent/2.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception:
            continue

        for entry in root.findall(".//item")[:max_per_query]:
            title = (entry.findtext("title") or "").strip()
            link = (entry.findtext("link") or "").strip()
            pub_date = (entry.findtext("pubDate") or "").strip()
            source_el = entry.find("source")
            source_name = (source_el.text or "Unknown").strip() if source_el is not None else "Unknown"
            source_url = source_el.attrib.get("url", link) if source_el is not None else link

            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            if not is_topic_relevant(title):
                continue

            tier, badge, include_base = classify_source_credibility(source_url, source_name, title)
            sentiment = score_financial_sentiment(title)
            entities = detect_entities(title)
            items.append({
                "published_date": pub_date[:16] if pub_date else now_almaty[:10],
                "fetched_at_almaty": now_almaty,
                "title": title,
                "source_name": source_name,
                "source_url": link,
                "credibility_tier": tier,
                "credibility_badge": badge,
                "include_in_base_forecast": include_base,
                "sentiment_score": sentiment,
                "affected_company_ids": entities,
                "analytical_takeaway": (
                    f"Auto-classified ({tier}). Sentiment {sentiment:+.2f}. "
                    + ("Excluded from base shock model due to unverified/rumor source." if not include_base else "Included in news-adjusted contagion stress layer.")
                ),
            })
    return items


def load_news_feed(path: Path = DEFAULT_FEED_PATH) -> dict:
    if not path.exists():
        return {"last_run_almaty": "Never", "schedule": "Daily 10:00 Asia/Almaty (05:00 UTC)", "items": []}
    return json.loads(path.read_text(encoding="utf-8"))


def get_company_sentiment_map(feed: dict, include_unverified: bool = False) -> dict[str, float]:
    """Aggregate average news sentiment per company_id from the daily feed."""
    buckets: dict[str, list[float]] = {}
    for item in feed.get("items", []):
        if not include_unverified and not item.get("include_in_base_forecast", False):
            continue
        score = float(item.get("sentiment_score", 0.0))
        for cid in item.get("affected_company_ids", []):
            buckets.setdefault(str(cid), []).append(score)
    return {cid: round(sum(vals) / len(vals), 3) for cid, vals in buckets.items() if vals}


def run_daily_update(path: Path = DEFAULT_FEED_PATH) -> dict:
    """Run the 10:00 Almaty daily news scan and merge new verified items into data/news_feed.json."""
    existing = load_news_feed(path)
    fetched = fetch_daily_rss_news()
    existing_titles = {it["title"] for it in existing.get("items", [])}
    merged = [it for it in fetched if it["title"] not in existing_titles] + existing.get("items", [])
    payload = {
        "last_run_almaty": datetime.now(ALMATY_TZ).strftime("%Y-%m-%d %H:%M +05 (Asia/Almaty)"),
        "schedule": "Daily 10:00 Asia/Almaty (05:00 UTC) via GitHub Actions",
        "items": merged[:60],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run_daily_update()
    print(f"News Agent updated at {result['last_run_almaty']}; total items={len(result['items'])}")
