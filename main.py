"""
Global Econ Sentiment Tracker - backend

Pulls headlines (plus the short standfirst/summary FT publishes with each one)
from several Financial Times RSS feeds, scores sentiment with FinBERT
(ProsusAI/finbert), and writes everything to data.json for the frontend.

Install deps:
    pip install feedparser transformers torch
"""

import json
import re
import sys
from datetime import datetime, timezone
from html import unescape

import feedparser
from transformers import pipeline

# Section name -> RSS URL. If one of these feeds is down or the URL is wrong,
# the script skips it and carries on with the others.
FEEDS = {
    "Top stories": "https://www.ft.com/rss/home/international",
    "Markets": "https://www.ft.com/markets?format=rss",
    "Global economy": "https://www.ft.com/global-economy?format=rss",
}

MAX_SUMMARY_CHARS = 280

# Loaded ONCE at import time - loading the model is slow, scoring is fast.
sentiment_pipeline = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    top_k=None,  # return all 3 class scores, not just the winner
)


def clean_summary(raw: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """RSS summaries often contain HTML tags/entities - turn them into plain text."""
    text = re.sub(r"<[^>]+>", " ", raw)      # strip HTML tags
    text = unescape(text)                     # &amp; -> &, &#8217; -> ', etc.
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"  # cut at a word boundary
    return text


def fetch_all_entries(feeds: dict[str, str]) -> list[tuple[str, dict]]:
    """Fetch every feed; return a flat list of (section_name, entry) pairs."""
    collected = []
    for section, url in feeds.items():
        feed = feedparser.parse(url)
        if not feed.entries:
            reason = feed.get("bozo_exception", "feed was empty")
            print(f"Skipping '{section}': no entries ({reason})")
            continue
        print(f"Fetched {len(feed.entries)} entries from '{section}'")
        for entry in feed.entries:
            collected.append((section, entry))
    return collected


def dedupe(pairs: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """The same story can appear in several sections - keep the first copy."""
    seen_titles = set()
    unique = []
    for section, entry in pairs:
        title = entry.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique.append((section, entry))
    return unique


def score_headlines(pairs: list[tuple[str, dict]]) -> list[dict]:
    """Run FinBERT on each headline + its gist together, in one batched call."""
    titles = [entry.get("title", "").strip() for _, entry in pairs]
    summaries = [clean_summary(entry.get("summary", "")) for _, entry in pairs]

    # Score on "headline. gist" so FinBERT gets the fuller context, not just
    # a short headline fragment. Falls back to the headline alone if a feed
    # entry has no summary.
    texts_to_score = [
        f"{title}. {summary}" if summary else title
        for title, summary in zip(titles, summaries)
    ]
    raw_results = sentiment_pipeline(texts_to_score, truncation=True)

    processed = []
    for (section, entry), title, summary, class_scores in zip(pairs, titles, summaries, raw_results):
        score_by_label: dict[str, float] = {
            str(item["label"]): float(item["score"]) for item in class_scores
        }
        top_label = max(score_by_label, key=score_by_label.get)
        signed_score = score_by_label["positive"] - score_by_label["negative"]

        processed.append({
            "headline": title,
            "summary": summary,
            "section": section,
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "sentiment": top_label.capitalize(),
            "confidence": round(score_by_label[top_label], 3),
            "signed_score": round(signed_score, 3),
        })
    return processed


def print_summary(processed: list[dict]) -> None:
    print(f"\nProcessed {len(processed)} headlines\n")
    for item in processed[:5]:
        print(f"Headline: {item['headline']}")
        if item["summary"]:
            print(f"Gist: {item['summary']}")
        print(
            f"Sentiment: {item['sentiment']} (confidence {item['confidence']}) "
            f"| Signed score: {item['signed_score']}"
        )
        print("-" * 50)

    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for item in processed:
        counts[item["sentiment"]] += 1
    avg_score = sum(item["signed_score"] for item in processed) / len(processed)

    print("\nSummary")
    print(f"Average signed score: {round(avg_score, 3)}")
    print(
        f"Positive: {counts['Positive']} | "
        f"Neutral: {counts['Neutral']} | "
        f"Negative: {counts['Negative']}"
    )


def main():
    pairs = dedupe(fetch_all_entries(FEEDS))

    if not pairs:
        # Exit with an error code so an automated run (GitHub Actions) fails
        # loudly instead of overwriting good data.json with an empty one.
        print("No headlines fetched from any feed - leaving data.json untouched.")
        sys.exit(1)

    processed = score_headlines(pairs)
    print_summary(processed)

    output = {
        "sources": FEEDS,
        "model": "ProsusAI/finbert",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "headline_count": len(processed),
        "headlines": processed,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("\nSaved all structured data to 'data.json'!")


if __name__ == "__main__":
    main()
