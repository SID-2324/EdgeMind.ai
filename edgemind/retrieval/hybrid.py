"""Stage 5B: lexical + local dense retrieval with optional cross-encoder reranking.

No cloud services, LLM inference, or Layer 3 integration. This module accepts
embedding/reranking adapters so algorithm tests never download external models.
"""
from __future__ import annotations

import math
from typing import Protocol, Sequence

from ..contracts import Evidence
from .bm25 import rank_bm25


class TextEmbedder(Protocol):
    def embed(self, documents: list[str]): ...


class TextReranker(Protocol):
    def rerank(self, query: str, documents: list[str]): ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity with defensive checks and zero-vector handling."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError('Embedding vectors must be nonempty and have matching dimensions')
    aa = sum(float(x) * float(x) for x in a)
    bb = sum(float(x) * float(x) for x in b)
    if not all(math.isfinite(float(x)) for x in a) or not all(math.isfinite(float(x)) for x in b):
        raise ValueError('Embedding vectors contain non-finite values')
    if not aa or not bb:
        return 0.0
    return sum(float(x) * float(y) for x, y in zip(a, b)) / math.sqrt(aa * bb)


def fuse_rankings(lexical_ids: list[str], semantic_ids: list[str],
                  rrf_k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. IDs must be unique within each input ranking."""
    if type(rrf_k) is not int or rrf_k <= 0:
        raise ValueError('rrf_k must be a positive integer')
    if len(lexical_ids) != len(set(lexical_ids)) or len(semantic_ids) != len(set(semantic_ids)):
        raise ValueError('Ranking source IDs must be unique')
    fused: dict[str, float] = {}
    for rank_list in (lexical_ids, semantic_ids):
        for rank, source_id in enumerate(rank_list, 1):
            fused[source_id] = fused.get(source_id, 0.0) + 1 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda row: (-row[1], row[0]))


def rank_hybrid(query: str, passages: list[Evidence], embedder: TextEmbedder,
                *, top_k: int = 3, candidate_k: int = 10,
                reranker: TextReranker | None = None,
                min_semantic_similarity: float = 0.0) -> tuple[list[Evidence], dict]:
    """Retrieve via BM25 and cosine embedding search, then optionally rerank.

    This small prototype re-embeds every chunk on each query. Production systems
    should cache/precompute document vectors and index them once.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError('top_k must be positive')
    if type(candidate_k) is not int or candidate_k < top_k:
        raise ValueError('candidate_k must be an integer >= top_k')
    if not -1.0 <= min_semantic_similarity <= 1.0:
        raise ValueError('min_semantic_similarity must lie between -1 and 1')
    if not passages or not query.strip():
        return [], {'bm25_hits': 0, 'semantic_hits': 0, 'fused_candidates': 0,
                    'reranked': False, 'semantic_model_used': False}
    ids = [p['source_id'] for p in passages]
    if len(ids) != len(set(ids)):
        raise ValueError('Passage source IDs must be unique')

    lexical = rank_bm25(query, passages, top_k=len(passages))
    lexical_ids = [item['source_id'] for _, item in lexical]

    # Separate encoding of query and passages is explicit for easier benchmarking.
    qvec = list(embedder.embed([query]))
    pvec = list(embedder.embed([p['text'] for p in passages]))
    if len(qvec) != 1 or len(pvec) != len(passages):
        raise ValueError('Embedding backend returned an unexpected number of vectors')
    sims = [(cosine(qvec[0], vector), p['source_id']) for p, vector in zip(passages, pvec)]
    semantic_ids = [source_id for score, source_id in
                    sorted(sims, key=lambda item: (-item[0], item[1]))
                    if score >= min_semantic_similarity]

    ranking = fuse_rankings(lexical_ids, semantic_ids)
    pool = ranking[:candidate_k]
    lookup = {p['source_id']: p for p in passages}
    if reranker is not None and pool:
        batch = [lookup[source_id]['text'] for source_id, _ in pool]
        scores = [float(s) for s in reranker.rerank(query, batch)]
        if len(scores) != len(pool) or not all(math.isfinite(s) for s in scores):
            raise ValueError('Reranker returned invalid relevance scores')
        rescored = [(score, source_id) for score, (source_id, _) in zip(scores, pool)]
        rescored.sort(key=lambda item: (-item[0], item[1]))
        result_ids = [source_id for _, source_id in rescored[:top_k]]
    else:
        result_ids = [source_id for source_id, _ in pool[:top_k]]

    evidence = [lookup[source_id] for source_id in result_ids]
    diagnostics = {
        'bm25_hits': len(lexical_ids),
        'semantic_hits': len(semantic_ids),
        'fused_candidates': len(pool),
        'reranked': reranker is not None,
        'semantic_model_used': True,
        'top_semantic_similarity': round(max((s for s, _ in sims), default=0.0), 4),
    }
    return evidence, diagnostics
