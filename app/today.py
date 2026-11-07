"""The Today desk: present-day facts for modern topics, kept apart from the gurus' own texts.

Two kinds of material, with different rules:

- TODAY passages (Wikipedia article leads, CC BY-SA 4.0). Stored as `Chunk`s with persona_id "today" and
  source_type "background", ids `today:<slug>:<n>`. A guru may cite one with [src:...] for a plain fact about
  the present; his judgement about that fact stays [inferred]. Sentences that name parties or office-holders
  are removed before storage (no living people, no party politics).
- Headlines (Google News RSS, free, no key; last 60 days, newest first). Title, outlet, date and link only. Shown in
  the UI as "In the news"; never sent to the LLM, because headlines name living politicians.

Everything is cached under data/today/, so the demo runs offline. `python -m scripts.build_today` refreshes it;
the API also refreshes stale headlines in the background.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import get_settings
from .rag.text import overlap
from .schemas import Chunk

log = logging.getLogger(__name__)

UA = {"User-Agent": "Shastrartha/0.1 (Tekathon education demo; https://github.com/)"}
WIKI_API = "https://en.wikipedia.org/w/api.php"
GNEWS_RSS = "https://news.google.com/rss/search"
NEWS_TTL_S = 6 * 3600
NEWS_SPACING_S = 1.5

# Sentences naming parties or the holders of political office are dropped from TODAY passages.
_POLITICAL = re.compile(
    r"\b(BJP|Bharatiya Janata|Indian National Congress|Congress party|Aam Aadmi|AAP|DMK|AIADMK|TMC|Trinamool|"
    r"Shiv Sena|Samajwadi|BSP|CPI\(M\)|NDA government|UPA government|Republican Party|Democratic Party|"
    r"(Prime|Chief|Defence|Finance|Home|External Affairs) Minister [A-Z][a-z]+|President [A-Z][a-z]+ [A-Z][a-z]+|"
    r"ABVP|NSUI|RSS-affiliated|Modi|Gandhi|Kejriwal|Amit Shah|Yogi Adityanath|Owaisi|Mamata|Trump|Biden|Xi Jinping|"
    r"Putin|Netanyahu|Zelensky|Erdogan|Khamenei|Macron|Starmer|Carney|Poroshenko|"
    r"(Senator|Sen\.|Reps?\.|Governor|Chancellor|Premier) [A-Z][a-z]+)\b"
)


def _dir() -> Path:
    d = get_settings().data_dir / "today"
    (d / "news").mkdir(parents=True, exist_ok=True)
    return d


def sources() -> dict[str, dict]:
    path = _dir() / "sources.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:48]


# ---------------------------------------------------------------- Wikipedia (citable background)

def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s*\([^()]*(pronounced|listen|lit\.)[^()]*\)", "", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text) if len(s.split()) >= 5]


def fetch_wiki(title: str, max_passages: int = 5) -> list[Chunk]:
    """Lead section of one article as up to `max_passages` short passages (3 sentences each)."""
    r = requests.get(WIKI_API, headers=UA, timeout=30, params={
        "action": "query", "format": "json", "redirects": 1, "prop": "extracts|info", "inprop": "url",
        "exintro": 1, "explaintext": 1, "titles": title})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"Wikipedia has no article '{title}'")
    kept = [s for s in _sentences(page["extract"]) if not _POLITICAL.search(s)]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = []
    for i in range(0, min(len(kept), max_passages * 3), 3):
        out.append(Chunk(
            chunk_id=f"today:{_slug(page['title'])}:{i // 3 + 1:02d}", source_id=f"wikipedia_{_slug(page['title'])}",
            persona_id="today", work=f"Wikipedia: {page['title']}", edition=f"Wikipedia, retrieved {today}",
            section="Lead section", language="en", license="CC BY-SA 4.0", url=page.get("fullurl", ""),
            text=" ".join(kept[i:i + 3]), source_type="background",
        ))
    return out


WB_API = "https://api.worldbank.org/v2/country/{country}/indicator/{code}"
WB_LABELS = {  # plain wording for indicators whose official names read badly in a sentence
    "GC.TAX.TOTL.GD.ZS": "Central government tax revenue",
    "GC.XPN.TOTL.GD.ZS": "Central government expense",
    "NE.TRD.GNFS.ZS": "Trade in goods and services",
    "MS.MIL.MPRT.KD": "Arms imports (SIPRI trend-indicator value)",
    "IT.NET.USER.ZS": "Share of people using the Internet",
    "SL.AGR.EMPL.ZS": "Share of workers employed in agriculture",
    "SI.POV.DDAY": "Share of people living on under $3.00 a day (2021 PPP)",
}


def _fmt(v: float, unit: str) -> str:
    if unit.startswith("%"):
        return f"{v:.1f}%" if v < 100 else f"{v:.0f}%"
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f} billion"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f} million"
    return f"{v:,.1f}"


def fetch_indicator(code: str, country: str = "IND") -> Chunk:
    """One World Bank indicator as a citable passage: latest value, and the value about ten years earlier."""
    for attempt in range(3):  # the World Bank API is free but sometimes slow
        try:
            r = requests.get(WB_API.format(country=country, code=code), headers=UA, timeout=45,
                             params={"format": "json", "date": "2000:2030", "per_page": 60})
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)
    body = r.json()
    rows = [x for x in (body[1] if len(body) > 1 and body[1] else []) if x.get("value") is not None]
    if not rows:
        raise ValueError(f"World Bank has no data for {code}")
    rows.sort(key=lambda x: int(x["date"]), reverse=True)
    latest = rows[0]
    name = latest["indicator"]["value"]
    unit = "%" if "%" in name else ""
    label = WB_LABELS.get(code) or re.sub(r"\s*\((% of [^)]*|current US\$|modeled ILO estimate)\)", "", name).strip()
    where = latest["country"]["value"]
    text = f"According to the World Bank, {where}'s {label[0].lower() + label[1:]} was {_fmt(latest['value'], unit)}"
    text += f" of {re.search(r'% of ([^)]*)', name).group(1)}" if "% of" in name and not label.startswith("Share") else ""
    text += f" in {latest['date']}."
    past = next((x for x in rows if int(latest["date"]) - int(x["date"]) >= 9), None)
    if past:
        text += f" In {past['date']} it was {_fmt(past['value'], unit)}."
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Chunk(
        chunk_id=f"today:wb_{_slug(code)}:01", source_id=f"worldbank_{_slug(code)}", persona_id="today",
        work=f"World Bank: {name}", edition=f"World Development Indicators, retrieved {today}", section=f"{where}, {latest['date']}",
        language="en", license="CC BY 4.0", url=f"https://data.worldbank.org/indicator/{code}?locations={country[:2]}",
        text=text, source_type="background",
    )


def load_passages() -> list[Chunk]:
    path = _dir() / "wiki.jsonl"
    if not path.exists():
        return []
    return [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def passages_for(topic_id: str) -> list[Chunk]:
    from .knowledge import get_knowledge

    cfg = sources().get(topic_id, {})
    if topic_id.startswith("news-"):  # generated from a headline: background titles are stored with the topic
        from .headline import load_generated

        cfg = {"wiki": load_generated().get(topic_id, {}).get("wiki", [])}
    titles = {_slug(t) for t in cfg.get("wiki", [])} | {f"wb_{_slug(c)}" for c in cfg.get("data", [])}
    k = get_knowledge()
    return [c for c in k.today if c.chunk_id.split(":")[1] in titles]


def today_hits(topic, query: str, k: int = 2):
    """The `k` TODAY passages that best fit this topic and the current argument, as retrieval hits."""
    from .rag.retrieve import Hit

    pool = passages_for(topic.id)
    if not pool:
        return []
    ctx = f"{topic.question} {topic.context}"
    def score(c: Chunk) -> float:
        figure = 0.35 if c.chunk_id.startswith("today:wb_") else 0.0  # a hard number beats a paragraph
        return 1.2 * overlap(ctx, c.text) + overlap(query, c.text) + figure

    figures = sorted((c for c in pool if c.chunk_id.startswith("today:wb_")), key=score, reverse=True)[:1]
    rest = sorted((c for c in pool if c not in figures), key=score, reverse=True)
    return [Hit(c, 1.0, 90 + i) for i, c in enumerate((figures + rest)[:k])]


# ---------------------------------------------------------------- headlines (UI only)

_news_lock = threading.Lock()
_news_last = [0.0]
_refreshing: set[str] = set()


def fetch_headlines(query: str, n: int = 6, must: list[str] | None = None, not_: list[str] | None = None) -> list[dict]:
    """Recent headlines (last 60 days) from Google News RSS, newest first. Headlines naming parties or
    office-holders are skipped so the screen stays politically neutral."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    with _news_lock:  # one request at a time, spaced, to be a polite client
        wait = NEWS_SPACING_S - (time.monotonic() - _news_last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(GNEWS_RSS, timeout=30, headers={"User-Agent": "Mozilla/5.0 (Shastrartha demo)"},
                             params={"q": f"{query} when:60d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
        finally:
            _news_last[0] = time.monotonic()
    if r.status_code != 200:
        raise ValueError(f"Google News {r.status_code}")
    items = []
    for it in ET.fromstring(r.content).findall("./channel/item"):
        outlet = it.findtext("source") or ""
        title = re.sub(r"\s+", " ", it.findtext("title") or "").strip()
        title = title.removesuffix(f" - {outlet}").strip() if outlet else title
        low = title.lower()
        if not title or _POLITICAL.search(title):
            continue
        if must and not any(m in low for m in must):  # off-topic match (e.g. school "lateral entry")
            continue
        if not_ and any(x in low for x in not_):
            continue
        try:
            when = parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        items.append((when, {"title": title, "url": it.findtext("link") or "", "outlet": outlet, "date": when.strftime("%Y-%m-%d")}))
    items.sort(key=lambda x: x[0], reverse=True)
    seen, out = set(), []
    for _, h in items:
        key = h["title"].lower()[:60]
        if key not in seen:
            seen.add(key)
            out.append(h)
        if len(out) >= 20:
            break
    return out[:n]


def keep_relevant(question: str, scope: str, items: list[dict], n: int = 6) -> list[dict]:
    """One small LLM call: which headlines actually report on this debate question? Keyword-only if no LLM."""
    if not items:
        return items
    from .agents.llm import LLMUnavailable, gemini_pool

    s = get_settings()
    if s.llm_provider != "gemini" or not s.gemini_api_keys:
        return items[:n]
    listing = "\n".join(f"{i}. {h['title']} ({h['outlet']})" for i, h in enumerate(items))
    where = "India's public policy" if scope == "india" else "world affairs"
    try:
        raw = gemini_pool().generate(
            "You filter news headlines for a student debate app. Return JSON only.",
            f"DEBATE QUESTION ({where}): {question}\nHEADLINES:\n{listing}\n\n"
            'Return {"keep": [indices]} of headlines that report news or analysis directly relevant to this question'
            + (" in India" if scope == "india" else "") + ". Exclude sport, entertainment, other countries' unrelated "
            "politics, and loose keyword matches. Best first, at most " + str(n) + ".",
            200, temperature=0)
        keep = [i for i in json.loads(raw[raw.find("{"): raw.rfind("}") + 1]).get("keep", []) if isinstance(i, int) and 0 <= i < len(items)]
        return [items[i] for i in dict.fromkeys(keep)][:n]
    except (LLMUnavailable, ValueError, json.JSONDecodeError) as exc:
        log.info("Headline relevance check skipped: %s", exc)
        return items[:n]


def _news_path(topic_id: str) -> Path:
    return _dir() / "news" / f"{topic_id}.json"


def cached_news(topic_id: str) -> dict:
    p = _news_path(topic_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"fetched_at": None, "items": []}


def refresh_news(topic_id: str) -> dict:
    cfg = sources().get(topic_id, {})
    q = cfg.get("news")
    if not q:
        return cached_news(topic_id)
    from .knowledge import get_knowledge

    topic = get_knowledge().topics.get(topic_id)
    items = fetch_headlines(q, n=20, must=cfg.get("must"), not_=cfg.get("not"))
    if topic is not None:
        items = keep_relevant(topic.question, topic.scope or "world", items)
    data = {"fetched_at": time.time(), "query": q, "items": items}
    old = cached_news(topic_id)
    # Keep the old list when the new answer is empty, but record the attempt so it is not retried every request.
    if not items and old["items"]:
        data["items"] = old["items"]
    _news_path(topic_id).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def news(topic_id: str) -> dict:
    """Cached headlines now; refresh in the background when older than 6 hours."""
    data = cached_news(topic_id)
    stale = not data["fetched_at"] or time.time() - data["fetched_at"] > NEWS_TTL_S
    if stale and topic_id in sources() and topic_id not in _refreshing:
        _refreshing.add(topic_id)

        def run() -> None:
            try:
                refresh_news(topic_id)
            except Exception as exc:  # noqa: BLE001 - headlines are optional
                log.info("Headline refresh for %s skipped: %s", topic_id, exc)
            finally:
                _refreshing.discard(topic_id)

        threading.Thread(target=run, name=f"news-{topic_id}", daemon=True).start()
    return data


def record(topic) -> dict:
    """Everything the UI shows under a modern topic's briefing."""
    refs, figures = [], []
    for c in passages_for(topic.id):
        if c.chunk_id.startswith("today:wb_"):
            figures.append({"chunk_id": c.chunk_id, "text": c.text, "source": c.work, "url": c.url, "license": c.license})
        elif all(r["url"] != c.url for r in refs):
            refs.append({"title": c.work.removeprefix("Wikipedia: "), "url": c.url, "license": c.license, "retrieved": c.edition})
    if topic.id.startswith("news-"):  # generated from a headline: show that headline as the news
        from .headline import load_generated

        h = load_generated().get(topic.id, {}).get("headline")
        n = {"items": [h] if h else [], "fetched_at": None}
    else:
        n = news(topic.id)
    return {"topic_id": topic.id, "briefing": topic.context, "references": refs, "figures": figures,
            "passages": [c.chunk_id for c in passages_for(topic.id)],
            "headlines": n["items"], "headlines_fetched_at": n["fetched_at"],
            "headline_note": "Headlines via Google News: titles and links only. Gurus do not see them."}
