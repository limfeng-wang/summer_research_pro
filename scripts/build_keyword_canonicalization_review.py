#!/usr/bin/env python3
"""Build blinded keyword-canonicalization review materials.

The script reads the country keyword-statistics workbook, retrieves candidate
duplicate label pairs within each CSM dimension, and writes:

1. a blinded reviewer workbook without country/count/rank fields
2. a non-blinded audit inventory for the coordinator
3. a JSON manifest documenting thresholds and package availability

Sentence embeddings are optional because production environments differ. When
`sentence_transformers` is available, pass `--embedding-model`; otherwise the
script still builds lexical candidates and records that embeddings were absent.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


DEFAULT_INPUT = Path("data/keyword_statistics_CHI_JPN_KOR.xlsx")
DEFAULT_OUT_DIR = Path("outputs/keyword_canonicalization")

MODIFIER_PATTERNS = {
    "severity": r"\b(mild|moderate|severe|intense|unbearable|extreme|sharp|acute)\b",
    "temporality": r"\b(night|nocturnal|persistent|recurrent|chronic|acute|sudden|intermittent)\b",
    "cause_trigger": r"\b(wisdom|cold|hot|heat|extraction|post-extraction|bruxism|clenching|caries|cavity|trauma)\b",
    "body_site": r"\b(gum|gingival|jaw|tooth|teeth|face|facial|orofacial|throat|ear|head)\b",
    "treatment": r"\b(ibuprofen|antibiotic|analgesic|medication|dentist|dental visit|extraction|root canal)\b",
    "consequence": r"\b(sleep|eating|dietary|work|study|impairment|disturbance|restriction)\b",
    "uncertainty": r"\b(possible|suspected|unspecified|unclear|unknown)\b",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "or",
    "of",
    "the",
    "to",
    "in",
    "with",
    "due",
    "related",
    "relatedness",
}


@dataclass(frozen=True)
class Thresholds:
    char_tfidf: float = 0.82
    token_jaccard: float = 0.80
    embedding: float = 0.86
    mixed_char: float = 0.65
    mixed_embedding: float = 0.82
    max_pairs_per_dimension: int = 600
    neighbors_per_label: int = 25
    cluster_review_min_total_count: int = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--char-threshold", type=float, default=Thresholds.char_tfidf)
    parser.add_argument("--token-threshold", type=float, default=Thresholds.token_jaccard)
    parser.add_argument("--embedding-threshold", type=float, default=Thresholds.embedding)
    parser.add_argument("--mixed-char-threshold", type=float, default=Thresholds.mixed_char)
    parser.add_argument("--mixed-embedding-threshold", type=float, default=Thresholds.mixed_embedding)
    parser.add_argument("--max-pairs-per-dimension", type=int, default=Thresholds.max_pairs_per_dimension)
    parser.add_argument("--neighbors-per-label", type=int, default=Thresholds.neighbors_per_label)
    parser.add_argument("--cluster-review-min-total-count", type=int, default=Thresholds.cluster_review_min_total_count)
    return parser.parse_args()


def normalize_label(label: object) -> str:
    text = str(label or "").casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[/_(),;:]+", " ", text)
    text = re.sub(r"[^a-z0-9+\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(label: str) -> set[str]:
    return {tok for tok in normalize_label(label).replace("-", " ").split() if tok and tok not in STOPWORDS}


def token_jaccard(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def modifier_flags(label: str) -> str:
    norm = normalize_label(label)
    flags = [name for name, pattern in MODIFIER_PATTERNS.items() if re.search(pattern, norm)]
    return ";".join(flags)


def read_keyword_workbook(path: Path) -> pd.DataFrame:
    rows = []
    for sheet_name in pd.ExcelFile(path).sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        country = sheet_name.split("_")[-1]
        body = raw.iloc[3:].copy()
        body.columns = ["rank", "dimension", "keyword", "main_count", "post_count"]
        body = body.dropna(subset=["dimension", "keyword"])
        for _, row in body.iterrows():
            rows.append(
                {
                    "source_sheet": sheet_name,
                    "country": country,
                    "rank": row["rank"],
                    "dimension": str(row["dimension"]).strip(),
                    "keyword": str(row["keyword"]).strip(),
                    "main_count": int(pd.to_numeric(row["main_count"], errors="coerce") or 0),
                    "post_count": int(pd.to_numeric(row["post_count"], errors="coerce") or 0),
                }
            )
    df = pd.DataFrame(rows)
    return df[df["keyword"].ne("") & df["dimension"].ne("")]


def build_inventory(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["dimension", "keyword", "country"], as_index=False)
        .agg(main_count=("main_count", "sum"), post_count=("post_count", "sum"))
    )
    pivot = grouped.pivot_table(
        index=["dimension", "keyword"],
        columns="country",
        values="main_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for country in ["CHI", "JPN", "KOR"]:
        if country not in pivot.columns:
            pivot[country] = 0
    pivot["total_main_count"] = pivot[["CHI", "JPN", "KOR"]].sum(axis=1)
    pivot["normalized_keyword"] = pivot["keyword"].map(normalize_label)
    pivot["modifier_flags"] = pivot["keyword"].map(modifier_flags)
    return pivot[
        [
            "dimension",
            "keyword",
            "normalized_keyword",
            "modifier_flags",
            "CHI",
            "JPN",
            "KOR",
            "total_main_count",
        ]
    ].sort_values(["dimension", "total_main_count", "keyword"], ascending=[True, False, True])


def load_embeddings(labels: list[str], model_name: str) -> tuple[np.ndarray | None, str]:
    if not model_name:
        return None, "not_requested"
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, "sentence_transformers_not_installed"

    model = SentenceTransformer(model_name)
    embeddings = model.encode(labels, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(embeddings), model_name


def candidate_reason(
    char_score: float,
    token_score: float,
    embedding_score: float | None,
    thresholds: Thresholds,
) -> str | None:
    reasons = []
    if char_score >= thresholds.char_tfidf:
        reasons.append("char_tfidf")
    if token_score >= thresholds.token_jaccard:
        reasons.append("token_jaccard")
    if embedding_score is not None and embedding_score >= thresholds.embedding:
        reasons.append("embedding")
    if (
        embedding_score is not None
        and char_score >= thresholds.mixed_char
        and embedding_score >= thresholds.mixed_embedding
    ):
        reasons.append("mixed_char_embedding")
    return ";".join(reasons) if reasons else None


def add_pair(candidate_pairs: dict[tuple[int, int], dict[str, float | None]], i: int, j: int, **scores: float | None) -> None:
    if i == j:
        return
    key = (i, j) if i < j else (j, i)
    current = candidate_pairs.setdefault(key, {"char_score": None, "embedding_score": None})
    for name, score in scores.items():
        if score is not None:
            current[name] = score


def add_top_neighbors(
    candidate_pairs: dict[tuple[int, int], dict[str, float | None]],
    matrix,
    *,
    score_name: str,
    neighbors_per_label: int,
) -> None:
    n_labels = matrix.shape[0]
    if n_labels < 2:
        return
    n_neighbors = min(neighbors_per_label + 1, n_labels)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)
    for i, row in enumerate(indices):
        for distance, j in zip(distances[i], row):
            if i == j:
                continue
            add_pair(candidate_pairs, i, int(j), **{score_name: 1.0 - float(distance)})


def add_token_candidates(
    candidate_pairs: dict[tuple[int, int], dict[str, float | None]],
    labels: list[str],
    *,
    token_threshold: float,
) -> None:
    token_index: dict[str, list[int]] = defaultdict(list)
    label_tokens = [token_set(label) for label in labels]
    for i, tokens in enumerate(label_tokens):
        for token in tokens:
            token_index[token].append(i)

    for i, tokens in enumerate(label_tokens):
        possible: set[int] = set()
        for token in tokens:
            possible.update(token_index[token])
        for j in possible:
            if j <= i:
                continue
            union = tokens | label_tokens[j]
            if not union:
                continue
            score = len(tokens & label_tokens[j]) / len(union)
            if score >= token_threshold:
                add_pair(candidate_pairs, i, j)


def iter_dimension_candidates(
    dimension: str,
    labels: list[str],
    thresholds: Thresholds,
    embeddings: np.ndarray | None,
) -> Iterable[dict[str, object]]:
    if len(labels) < 2:
        return
    normalized = [normalize_label(label) for label in labels]
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False).fit_transform(normalized)
    candidate_pairs: dict[tuple[int, int], dict[str, float | None]] = {}
    add_top_neighbors(
        candidate_pairs,
        tfidf,
        score_name="char_score",
        neighbors_per_label=thresholds.neighbors_per_label,
    )
    add_token_candidates(candidate_pairs, labels, token_threshold=thresholds.token_jaccard)
    if embeddings is not None:
        add_top_neighbors(
            candidate_pairs,
            embeddings,
            score_name="embedding_score",
            neighbors_per_label=thresholds.neighbors_per_label,
        )

    candidates = []
    for (i, j), retrieved_scores in candidate_pairs.items():
        char_score = retrieved_scores.get("char_score")
        if char_score is None:
            char_score = float(cosine_similarity(tfidf[i], tfidf[j])[0, 0])
        token_score = token_jaccard(labels[i], labels[j])
        embedding_score = retrieved_scores.get("embedding_score")
        if embedding_score is None and embeddings is not None:
            embedding_score = float(cosine_similarity(embeddings[i : i + 1], embeddings[j : j + 1])[0, 0])
        reason = candidate_reason(char_score, token_score, embedding_score, thresholds)
        if not reason:
            continue
        score_for_sort = max(
            char_score,
            token_score,
            embedding_score if embedding_score is not None and not math.isnan(embedding_score) else 0.0,
        )
        candidates.append(
            {
                "candidate_id": "",
                "csm_dimension": dimension,
                "label_a": labels[i],
                "label_b": labels[j],
                "normalized_label_a": normalized[i],
                "normalized_label_b": normalized[j],
                "char_ngram_tfidf_cosine": round(char_score, 4),
                "token_jaccard": round(token_score, 4),
                "sentence_embedding_cosine": "" if embedding_score is None else round(embedding_score, 4),
                "retrieval_reason": reason,
                "modifier_flags_a": modifier_flags(labels[i]),
                "modifier_flags_b": modifier_flags(labels[j]),
                "reviewer1_decision": "",
                "reviewer2_decision": "",
                "consensus_decision": "",
                "canonical_label_if_merge": "",
                "merge_rule_invoked": "",
                "review_notes": "",
                "_sort_score": score_for_sort,
            }
        )
    candidates.sort(key=lambda row: (-float(row["_sort_score"]), row["label_a"], row["label_b"]))
    for row in candidates[: thresholds.max_pairs_per_dimension]:
        row.pop("_sort_score")
        yield row


def build_candidate_pairs(inventory: pd.DataFrame, thresholds: Thresholds, embedding_model: str) -> tuple[pd.DataFrame, str]:
    all_rows = []
    embedding_status = "not_requested"
    next_id = 1
    for dimension, dim_df in inventory.groupby("dimension", sort=True):
        labels = sorted(dim_df["keyword"].astype(str).unique(), key=str.casefold)
        embeddings, status = load_embeddings(labels, embedding_model)
        if embedding_model:
            embedding_status = status
        for row in iter_dimension_candidates(dimension, labels, thresholds, embeddings):
            row["candidate_id"] = f"KC{next_id:06d}"
            next_id += 1
            all_rows.append(row)
    return pd.DataFrame(all_rows), embedding_status


def protocol_rows() -> pd.DataFrame:
    lines = [
        "Reviewers are blinded to country, frequency, and rank.",
        "The cluster workbook is the primary review file; the pair workbook is a trace of how clusters were retrieved.",
        "Review each cluster as a proposed label family, splitting labels that preserve meaningful modifiers.",
        "The cluster workbook is capped by a predefined impact rule, not an exhaustive list of all possible label variants.",
        "Unreviewed labels remain separate unless later selected under a predefined supplemental review rule.",
        "Merge only spelling, punctuation, singular/plural, word-order, synonym, or translation-equivalent variants with no meaning change.",
        "Do not merge when severity, temporality, cause/trigger, body site, treatment, consequence, uncertainty, or experiencer/attribution changes.",
        "Similarity scores retrieved candidate pairs only; they are not merge decisions.",
        "Disagreements should be resolved by consensus or a third reviewer.",
    ]
    return pd.DataFrame({"protocol_instruction": lines})


def build_cluster_review(candidates: pd.DataFrame, inventory: pd.DataFrame, min_total_count: int) -> pd.DataFrame:
    counts = {
        (str(row.dimension), str(row.keyword)): int(row.total_main_count)
        for row in inventory.itertuples(index=False)
    }
    rows = []
    cluster_number = 1
    for dimension, dim_candidates in candidates.groupby("csm_dimension", sort=True):
        adjacency: dict[str, set[str]] = defaultdict(set)
        pair_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in dim_candidates.itertuples(index=False):
            label_a = str(row.label_a)
            label_b = str(row.label_b)
            max_count = max(counts.get((dimension, label_a), 0), counts.get((dimension, label_b), 0))
            if max_count < min_total_count:
                continue
            adjacency[label_a].add(label_b)
            adjacency[label_b].add(label_a)
            pair_ids[tuple(sorted((label_a, label_b)))].append(str(row.candidate_id))

        seen: set[str] = set()
        for start in sorted(adjacency, key=str.casefold):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            labels = []
            while stack:
                label = stack.pop()
                labels.append(label)
                for neighbor in adjacency[label]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            labels = sorted(labels, key=str.casefold)
            max_hidden_count = max(counts.get((dimension, label), 0) for label in labels)
            rows.append(
                {
                    "_max_hidden_count": max_hidden_count,
                    "cluster_id": f"KCL{cluster_number:05d}",
                    "csm_dimension": dimension,
                    "labels_in_proposed_family": "\n".join(labels),
                    "label_count": len(labels),
                    "modifier_flags_present": ";".join(
                        sorted({flag for label in labels for flag in modifier_flags(label).split(";") if flag})
                    ),
                    "source_candidate_ids": ";".join(
                        sorted(
                            {
                                candidate_id
                                for i, label_a in enumerate(labels)
                                for label_b in labels[i + 1 :]
                                for candidate_id in pair_ids.get(tuple(sorted((label_a, label_b))), [])
                            }
                        )
                    ),
                    "reviewer1_decision": "",
                    "reviewer1_canonical_labels": "",
                    "reviewer2_decision": "",
                    "reviewer2_canonical_labels": "",
                    "consensus_decision": "",
                    "consensus_canonical_labels": "",
                    "split_out_labels": "",
                    "merge_rule_invoked": "",
                    "review_notes": "",
                }
            )
            cluster_number += 1
    review = pd.DataFrame(rows)
    if review.empty:
        return review.drop(columns=["_max_hidden_count"], errors="ignore")
    review = review.sort_values(["_max_hidden_count", "csm_dimension", "label_count"], ascending=[False, True, False])
    return review.drop(columns=["_max_hidden_count"])


def write_outputs(
    out_dir: Path,
    inventory: pd.DataFrame,
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    blinded_path = out_dir / "keyword_pair_candidates_blinded.xlsx"
    cluster_path = out_dir / "keyword_cluster_candidates_blinded.xlsx"
    audit_path = out_dir / "keyword_label_inventory_audit.xlsx"
    manifest_path = out_dir / "keyword_canonicalization_manifest.json"
    mapping_template_path = out_dir / "canonical_keyword_mapping_template.xlsx"

    with pd.ExcelWriter(blinded_path, engine="openpyxl") as writer:
        protocol_rows().to_excel(writer, sheet_name="protocol", index=False)
        candidates.to_excel(writer, sheet_name="candidate_pairs_blinded", index=False)

    with pd.ExcelWriter(cluster_path, engine="openpyxl") as writer:
        protocol_rows().to_excel(writer, sheet_name="protocol", index=False)
        clusters.to_excel(writer, sheet_name="cluster_review_blinded", index=False)

    inventory.to_excel(audit_path, sheet_name="label_inventory_nonblinded", index=False)

    template = inventory[["dimension", "keyword"]].copy()
    template["fine_canonical_keyword"] = ""
    template["parent_canonical_keyword"] = ""
    template["merge_rule_invoked"] = ""
    template["consensus_source"] = ""
    template["notes"] = ""
    template.to_excel(mapping_template_path, sheet_name="canonical_mapping", index=False)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = Thresholds(
        char_tfidf=args.char_threshold,
        token_jaccard=args.token_threshold,
        embedding=args.embedding_threshold,
        mixed_char=args.mixed_char_threshold,
        mixed_embedding=args.mixed_embedding_threshold,
        max_pairs_per_dimension=args.max_pairs_per_dimension,
        neighbors_per_label=args.neighbors_per_label,
        cluster_review_min_total_count=args.cluster_review_min_total_count,
    )
    df = read_keyword_workbook(args.input)
    inventory = build_inventory(df)
    candidates, embedding_status = build_candidate_pairs(inventory, thresholds, args.embedding_model)
    clusters = build_cluster_review(candidates, inventory, thresholds.cluster_review_min_total_count)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "out_dir": str(args.out_dir),
        "dimensions": sorted(inventory["dimension"].unique().tolist()),
        "unique_dimension_keyword_labels": int(len(inventory)),
        "candidate_pairs": int(len(candidates)),
        "cluster_review_rows": int(len(clusters)),
        "thresholds": asdict(thresholds),
        "embedding_model_requested": args.embedding_model or None,
        "embedding_status": embedding_status,
        "review_blinding": {
            "country_hidden": True,
            "frequency_hidden": True,
            "rank_hidden": True,
        },
        "final_decision_layer": "two independent human reviewers plus consensus",
    }
    write_outputs(args.out_dir, inventory, candidates, clusters, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
