"""Small in-memory BM25 implementation, no LLM/API or external dependency.

Designed to teach and test keyword retrieval. Not a semantic embedding model.
"""
from __future__ import annotations
from collections import Counter
import math
import re

from ..contracts import Evidence

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)
_STOPWORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'do', 'does', 'for',
    'from', 'how', 'i', 'in', 'is', 'it', 'of', 'on', 'or', 'our', 'the', 'their',
    'these', 'this', 'to', 'was', 'what', 'when', 'where', 'which', 'who', 'with',
})


def tokenize(text: str) -> list[str]:
    """Lightweight English keyword tokenization. Not multilingual stemming."""
    return [word.lower() for word in _TOKEN_PATTERN.findall(text) if word.lower() not in _STOPWORDS]


def rank_bm25(query: str, passages: list[Evidence], top_k: int = 3,
              k1: float = 1.5, b: float = 0.75) -> list[tuple[float, Evidence]]:
    """Return positive-scoring ranked passages, highest first; deterministic ties."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if not passages:
        return []
    query_terms = set(tokenize(query))
    if not query_terms:
        return []
    tokenized = [tokenize(passage['text']) for passage in passages]
    lengths = [len(tokens) for tokens in tokenized]
    avg_len = sum(lengths) / len(lengths)
    if avg_len == 0:
        return []
    term_counts = [Counter(tokens) for tokens in tokenized]
    n = len(passages)
    document_frequency = Counter()
    for freq in term_counts:
        for token in freq:
            if token in query_terms:
                document_frequency[token] += 1
    scored: list[tuple[float, Evidence]] = []
    for doc, freq, length in zip(passages, term_counts, lengths):
        score = 0.0
        for term in query_terms:
            tf = freq.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_len))
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: (-item[0], item[1]['source_id']))
    return scored[:top_k]
