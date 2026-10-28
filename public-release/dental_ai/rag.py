"""Gold-set retrieval for few-shot CSM extraction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from dental_ai.goldset import assert_ready_for_rag_seed, load_csm_gold_json
from dental_ai.model_config import DEFAULT_MODELS_ROOT, ModelStackConfig
from dental_ai.schemas import ExtractionResult, SourcePost


TOKEN_RE = re.compile(r"[\w一-龥ぁ-んァ-ン가-힣]+", re.UNICODE)


@dataclass(frozen=True)
class RetrievedExample:
    """Retrieved gold example plus retrieval score."""

    result: ExtractionResult
    score: float
    rank: int


class PairReranker(Protocol):
    """Scores query/example text pairs."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one score per passage."""


@dataclass(frozen=True)
class RerankerConfig:
    """Runtime settings for pairwise reranking."""

    backend: str = "transformers"
    batch_size: int = 1
    max_length: int = 512
    use_fp16: bool = True
    device: str = "auto"


class TransformersSequenceReranker:
    """Direct HF sequence-classification reranker.

    This avoids the FlagEmbedding `compute_score` path that can hang on some
    Windows/WSL + small-GPU setups while keeping the same BGE reranker weights.
    """

    def __init__(self, model_path: str | Path, config: RerankerConfig):
        self.model_path = str(model_path)
        self.config = config
        self._tokenizer = None
        self._model = None
        self._device = None

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        self._ensure_loaded()
        scores: list[float] = []
        for start in range(0, len(passages), self.config.batch_size):
            batch = passages[start : start + self.config.batch_size]
            inputs = self._tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.no_grad():
                logits = self._model(**inputs).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)
            scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
        return scores

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device

        dtype = torch.float16 if self.config.use_fp16 and self._device == "cuda" else None
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path, **kwargs)
        self._model.to(self._device)
        self._model.eval()


class FlagEmbeddingPairReranker:
    """FlagEmbedding reranker kept as an explicit optional backend."""

    def __init__(self, model_path: str | Path, config: RerankerConfig):
        self.model_path = str(model_path)
        self.config = config
        self._reranker = None

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._reranker is None:
            from FlagEmbedding import FlagReranker

            devices = None if self.config.device == "auto" else [self.config.device]
            kwargs = {"use_fp16": self.config.use_fp16}
            if devices is not None:
                kwargs["devices"] = devices
            self._reranker = FlagReranker(self.model_path, **kwargs)
        scores = []
        for passage in passages:
            value = self._reranker.compute_score(
                [[query, passage]],
                batch_size=1,
                max_length=self.config.max_length,
                normalize=True,
            )
            if isinstance(value, list):
                value = value[0]
            scores.append(float(value))
        return scores


class GoldRAGRetriever:
    """Retrieve similar adjudicated CSM examples from the few-shot gold set."""

    def __init__(
        self,
        gold: Iterable[ExtractionResult],
        *,
        embedding_model_path: str | Path | None = None,
        reranker_model_path: str | Path | None = None,
        use_embeddings: bool = True,
        use_reranker: bool = False,
        reranker_config: RerankerConfig | None = None,
    ):
        self.gold = list(gold)
        assert_ready_for_rag_seed(self.gold, min_primary_posts=1, min_proxy_posts=1)
        self.embedding_model_path = Path(embedding_model_path) if embedding_model_path else None
        self.reranker_model_path = Path(reranker_model_path) if reranker_model_path else None
        self.use_embeddings = use_embeddings
        self.use_reranker = use_reranker
        self.reranker_config = reranker_config or RerankerConfig()
        self._texts = [_result_text(result) for result in self.gold]
        self._tokens = [_tokenize(text) for text in self._texts]
        self._embedder = None
        self._gold_embeddings = None
        self._reranker = None

    @classmethod
    def from_config(
        cls,
        stack: ModelStackConfig,
        *,
        models_root: str | Path = DEFAULT_MODELS_ROOT,
        use_embeddings: bool = True,
        use_reranker: bool | None = None,
    ) -> GoldRAGRetriever:
        gold_path = stack.paths.get("rag_gold", "")
        if not gold_path:
            raise ValueError("Model stack config must define paths.rag_gold for full HF stage")
        retriever_path = stack.spec("retriever").local_path(models_root)
        if use_reranker is None:
            use_reranker = bool(stack.runtime.get("use_reranker", False))
        reranker_path = stack.spec("reranker").local_path(models_root) if use_reranker else None
        reranker_config = RerankerConfig(
            backend=str(stack.runtime.get("reranker_backend", "transformers")),
            batch_size=int(stack.runtime.get("reranker_batch_size", 1)),
            max_length=int(stack.runtime.get("reranker_max_length", 512)),
            use_fp16=bool(stack.runtime.get("reranker_use_fp16", True)),
            device=str(stack.runtime.get("reranker_device", "auto")),
        )
        return cls(
            load_csm_gold_json(gold_path),
            embedding_model_path=retriever_path,
            reranker_model_path=reranker_path,
            use_embeddings=use_embeddings,
            use_reranker=use_reranker,
            reranker_config=reranker_config,
        )

    def retrieve(self, post: SourcePost, *, k: int) -> list[ExtractionResult]:
        return [item.result for item in self.retrieve_with_scores(post, k=k)]

    def retrieve_with_scores(self, post: SourcePost, *, k: int) -> list[RetrievedExample]:
        """Return ranked gold examples, preferring same language/country ties."""

        if k <= 0:
            return []
        scores = self._embedding_scores(post) if self.use_embeddings else None
        if scores is None:
            scores = self._lexical_scores(post)

        ranked = sorted(
            enumerate(scores),
            key=lambda item: (
                item[1],
                int(self.gold[item[0]].language == post.language),
                int(self.gold[item[0]].country == post.country),
            ),
            reverse=True,
        )[: max(k * 3, k)]
        reranked = self._rerank(post, ranked)[:k]
        return [
            RetrievedExample(result=self.gold[index], score=float(score), rank=rank)
            for rank, (index, score) in enumerate(reranked, start=1)
        ]

    def _embedding_scores(self, post: SourcePost) -> list[float] | None:
        if not self.embedding_model_path or not self.embedding_model_path.exists():
            return None
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(str(self.embedding_model_path), local_files_only=True)
            if self._gold_embeddings is None:
                self._gold_embeddings = self._embedder.encode(
                    self._texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            query = self._embedder.encode([post.combined_source_text], normalize_embeddings=True, show_progress_bar=False)[0]
            return [float(sum(a * b for a, b in zip(query, vector))) for vector in self._gold_embeddings]
        except Exception:
            return None

    def _lexical_scores(self, post: SourcePost) -> list[float]:
        query_tokens = _tokenize(post.combined_source_text)
        query_set = set(query_tokens)
        scores = []
        for result, tokens in zip(self.gold, self._tokens):
            token_set = set(tokens)
            overlap = len(query_set & token_set)
            union = len(query_set | token_set) or 1
            language_bonus = 0.15 if result.language == post.language else 0.0
            country_bonus = 0.05 if result.country == post.country else 0.0
            scores.append(overlap / union + language_bonus + country_bonus)
        return scores

    def _rerank(self, post: SourcePost, ranked: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if not self.use_reranker or not self.reranker_model_path or not self.reranker_model_path.exists():
            return ranked
        try:
            if self._reranker is None:
                self._reranker = build_pair_reranker(self.reranker_model_path, self.reranker_config)
            passages = [self._texts[index] for index, _ in ranked]
            scores = self._reranker.score(post.combined_source_text, passages)
            return sorted(
                [(index, float(score)) for (index, _), score in zip(ranked, scores)],
                key=lambda item: item[1],
                reverse=True,
            )
        except Exception:
            return ranked


def build_pair_reranker(model_path: str | Path, config: RerankerConfig) -> PairReranker:
    backend = config.backend.lower()
    if backend in {"transformers", "hf", "auto"}:
        return TransformersSequenceReranker(model_path, config)
    if backend in {"flagembedding", "flag"}:
        return FlagEmbeddingPairReranker(model_path, config)
    raise ValueError(f"Unsupported reranker backend: {config.backend}")


def retrieval_trace_rows(post: SourcePost, retrieved: list[RetrievedExample]) -> list[dict[str, object]]:
    """Build auditable retrieval trace rows for JSONL output."""

    return [
        {
            "post_id": post.post_id,
            "rank": item.rank,
            "score": item.score if math.isfinite(item.score) else None,
            "gold_post_id": item.result.post_id,
            "gold_country": item.result.country.value,
            "gold_language": item.result.language.value,
            "gold_experiencer_label": item.result.experiencer_label.value if item.result.experiencer_label else "",
            "gold_content_function": item.result.content_function.value if item.result.content_function else "",
        }
        for item in retrieved
    ]


def _result_text(result: ExtractionResult) -> str:
    return result.combined_source_text


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


__all__ = [
    "FlagEmbeddingPairReranker",
    "GoldRAGRetriever",
    "PairReranker",
    "RetrievedExample",
    "RerankerConfig",
    "TransformersSequenceReranker",
    "build_pair_reranker",
    "retrieval_trace_rows",
]
