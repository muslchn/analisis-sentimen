"""Scrape Indonesian Google Play reviews for sentiment analysis.

The labeling strategy uses a compact Indonesian sentiment lexicon on the
scraped review text. Star ratings are still preserved as metadata, but the
model target is derived from review wording so the classifier learns text
sentiment rather than rating leakage.

This keeps the dataset reproducible while avoiding manually authored or
open-source datasets.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

from google_play_scraper import Sort, reviews


APPS = [
    "com.gojek.app",
    "com.tokopedia.tkpd",
    "com.shopee.id",
    "com.grabtaxi.passenger",
    "com.traveloka.android",
    "id.dana",
    "ovo.id",
    "com.bukalapak.android",
]

FIELDNAMES = [
    "review_id",
    "app_id",
    "app_name",
    "content",
    "score",
    "thumbs_up_count",
    "review_created_at",
    "sentiment",
]

APP_NAMES = {
    "com.gojek.app": "Gojek",
    "com.tokopedia.tkpd": "Tokopedia",
    "com.shopee.id": "Shopee",
    "com.grabtaxi.passenger": "Grab",
    "com.traveloka.android": "Traveloka",
    "id.dana": "DANA",
    "ovo.id": "OVO",
    "com.bukalapak.android": "Bukalapak",
}

POSITIVE_TERMS = {
    "bagus",
    "baik",
    "mantap",
    "mudah",
    "cepat",
    "lancar",
    "suka",
    "puas",
    "membantu",
    "terbaik",
    "top",
    "keren",
    "recommended",
    "rekomendasi",
    "murah",
    "praktis",
    "aman",
    "nyaman",
    "ramah",
    "jelas",
    "bermanfaat",
    "hebat",
    "oke",
    "ok",
    "good",
    "great",
    "nice",
    "love",
    "excellent",
    "worth",
    "stabil",
}

NEGATIVE_TERMS = {
    "buruk",
    "jelek",
    "parah",
    "susah",
    "sulit",
    "lambat",
    "lemot",
    "error",
    "gagal",
    "kecewa",
    "rugi",
    "mahal",
    "ribet",
    "bohong",
    "tipu",
    "penipuan",
    "hilang",
    "hang",
    "crash",
    "bug",
    "macet",
    "lama",
    "buruknya",
    "payah",
    "zonk",
    "sampah",
    "mengecewakan",
    "aneh",
    "blokir",
    "tidak",
    "nggak",
    "ga",
    "gak",
    "kurang",
    "bad",
    "worst",
    "problem",
}


@dataclass(frozen=True)
class ScrapeConfig:
    target: int
    per_request: int
    sleep_seconds: float
    output: Path


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r"[\r\n\t]+", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_from_text(text: str) -> str:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())
    positive_score = sum(token in POSITIVE_TERMS for token in tokens)
    negative_score = sum(token in NEGATIVE_TERMS for token in tokens)

    if positive_score > negative_score:
        return "positif"
    if negative_score > positive_score:
        return "negatif"
    return "netral"


def scrape_app_reviews(app_id: str, config: ScrapeConfig) -> list[dict[str, str]]:
    """Fetch reviews for one app across all star ratings for class coverage."""
    rows: list[dict[str, str]] = []
    per_score_target = max(1, config.target // (len(APPS) * 5))

    for score in [1, 2, 3, 4, 5]:
        continuation_token = None
        collected_for_score = 0

        while collected_for_score < per_score_target:
            batch, continuation_token = reviews(
                app_id,
                lang="id",
                country="id",
                sort=Sort.NEWEST,
                count=min(config.per_request, per_score_target - collected_for_score),
                filter_score_with=score,
                continuation_token=continuation_token,
            )

            if not batch:
                break

            for item in batch:
                content = clean_text(item.get("content", ""))
                if len(content) < 4:
                    continue

                rows.append(
                    {
                        "review_id": str(item.get("reviewId", "")),
                        "app_id": app_id,
                        "app_name": APP_NAMES.get(app_id, app_id),
                        "content": content,
                        "score": str(item.get("score", score)),
                        "thumbs_up_count": str(item.get("thumbsUpCount", 0)),
                        "review_created_at": str(item.get("at", "")),
                        "sentiment": label_from_text(content),
                    }
                )
                collected_for_score += 1

            if continuation_token is None:
                break

            time.sleep(config.sleep_seconds)

    return rows


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique_rows: list[dict[str, str]] = []

    for row in rows:
        key = (row["review_id"], row["content"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    return unique_rows


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Google Play reviews.")
    parser.add_argument("--target", type=int, default=12000, help="Target number of rows.")
    parser.add_argument("--per-request", type=int, default=200, help="Reviews per request.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between requests.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("google_play_reviews_sentiment.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScrapeConfig(
        target=args.target,
        per_request=args.per_request,
        sleep_seconds=args.sleep,
        output=args.output,
    )

    all_rows: list[dict[str, str]] = []
    for app_id in APPS:
        app_rows = scrape_app_reviews(app_id, config)
        all_rows.extend(app_rows)
        print(f"{app_id}: {len(app_rows)} rows")

    unique_rows = deduplicate(all_rows)
    write_csv(unique_rows[: config.target], config.output)
    print(f"Saved {min(len(unique_rows), config.target)} unique rows to {config.output}")


if __name__ == "__main__":
    main()
