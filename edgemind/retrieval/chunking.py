"""Passage chunking; keeps stable source identifiers for citations."""
from __future__ import annotations
from ..contracts import Evidence


def chunk_documents(documents: list[Evidence], chunk_words: int = 120,
                    overlap_words: int = 20) -> list[Evidence]:
    """Split long source texts into overlapping word chunks.

    Keep source IDs unique so later answer verification can trace a passage back
    to a specific input. This is a word-window baseline, not semantic chunking.
    """
    if type(chunk_words) is not int or not 1 <= chunk_words <= 2000:
        raise ValueError('chunk_words must be 1..2000')
    if type(overlap_words) is not int or not 0 <= overlap_words < chunk_words:
        raise ValueError('overlap_words must be >= 0 and less than chunk_words')
    if not isinstance(documents, list):
        raise ValueError('documents must be a list')
    passages: list[Evidence] = []
    for doc in documents:
        source_id = doc.get('source_id')
        text = doc.get('text')
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError('every document needs a nonempty source_id')
        if not isinstance(text, str):
            raise ValueError('every document needs text as a string')
        words = text.split()
        if not words:
            continue
        step = chunk_words - overlap_words
        for chunk_number, start in enumerate(range(0, len(words), step), 1):
            part = words[start:start + chunk_words]
            if not part:
                break
            passages.append({
                'source_id': f'{source_id}#chunk-{chunk_number:03d}',
                'text': ' '.join(part),
            })
            if start + chunk_words >= len(words):
                break
    return passages
