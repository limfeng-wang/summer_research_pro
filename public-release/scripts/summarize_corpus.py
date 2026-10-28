"""Summarize the archived manuscript sample, keeping inclusion rules explicit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/publication_dataset/main/posts.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/corpus_summary"))
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Private post-level input is not distributed. Supply --input or use scripts/reproduce_aggregate_results.py for public checks.")
    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    primary = [row for row in rows if row["analysis_flags"]["manuscript_primary"]]
    country_counts = Counter(row["country"] for row in primary)
    domains = Counter()
    keywords = Counter()
    keyword_posts = defaultdict(set)
    for row in primary:
        units = [unit for unit in row["units"] if unit["quality_flags"]["judge_accepted"]]
        domains.update((row["country"], domain) for domain in {unit["domain"] for unit in units})
        for unit in units:
            concept = unit.get("normalized_concept_en", "")
            if concept:
                key = (row["country"], unit["domain"], concept)
                keywords[key] += 1
                keyword_posts[key].add(row["post_id"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "domain_post_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["country", "domain", "posts", "denominator_posts", "percentage"])
        for (country, domain), count in sorted(domains.items()):
            writer.writerow([country, domain, count, country_counts[country], 100 * count / country_counts[country]])
    with (args.out_dir / "original_keyword_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["country", "domain", "normalized_concept_en", "unit_count", "post_count"])
        for key, count in sorted(keywords.items()):
            writer.writerow([*key, count, len(keyword_posts[key])])
    summary = {
        "posts": len(rows),
        "countries": dict(Counter(row["country"] for row in rows)),
        "primary_posts": len(primary),
        "primary_countries": dict(country_counts),
        "primary_verdict_accepted_units": sum(
            unit["quality_flags"]["judge_accepted"] for row in primary for unit in row["units"]
        ),
        "strict_primary_posts": sum(row["analysis_flags"]["strict_primary"] for row in rows),
        "unit_rule": "Archived manuscript rule: judge_verdict=accept; see quality flags for stricter alternatives.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
