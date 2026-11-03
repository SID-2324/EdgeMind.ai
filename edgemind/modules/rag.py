"""Stage 5B RAG: BM25 baseline and opt-in local semantic hybrid search.

The default remains BM25 to preserve all previous scenarios. Hybrid mode is
explicit; the optional cross-encoder can be toggled separately. All retrieval
is independent of Layer 3 / generative LLMs.
"""
from ..state import EdgeState
from ..retrieval.bm25 import rank_bm25
from ..retrieval.chunking import chunk_documents
from ..retrieval.hybrid import rank_hybrid


def retrieve(state: EdgeState, *, embedder=None, reranker=None) -> dict:
    settings = state.get('rag_settings', {})
    top_k = settings.get('top_k', 3)
    chunk_words = settings.get('chunk_words', 120)
    overlap_words = settings.get('overlap_words', 20)
    mode = settings.get('mode', 'bm25')
    enable_reranker = settings.get('enable_reranker', False)
    candidate_k = settings.get('candidate_k', max(10, top_k))
    if mode not in ('bm25', 'hybrid'):
        raise ValueError('RAG mode must be bm25 or hybrid')
    if enable_reranker and mode != 'hybrid':
        raise ValueError('Reranker is only supported in hybrid mode')
    docs = state.get('documents', [])
    passages = chunk_documents(docs, chunk_words=chunk_words, overlap_words=overlap_words)
    if mode == 'bm25':
        matches = rank_bm25(state['query'], passages, top_k=top_k)
        evidence = [passage for _, passage in matches]
        diagnostics = {
            'method': 'bm25',
            'input_documents': len(docs),
            'indexed_passages': len(passages),
            'retrieved_passages': len(evidence),
            'scores': [round(score, 6) for score, _ in matches],
        }
        message = (f'RAG: BM25 ranked {len(evidence)} passages from '
                   f'{len(passages)} chunks ({len(docs)} source documents); '
                   'no semantic embeddings or LLM used')
    else:
        if passages:
            if embedder is None:
                from ..retrieval.local_models import get_embedder
                embedder = get_embedder()
            if enable_reranker and reranker is None:
                from ..retrieval.local_models import get_reranker
                reranker = get_reranker()
            evidence, extra = rank_hybrid(
                state['query'], passages, embedder, top_k=top_k,
                candidate_k=candidate_k,
                reranker=reranker if enable_reranker else None,
                min_semantic_similarity=settings.get('min_semantic_similarity', 0.0),
            )
        else:
            evidence, extra = [], {'bm25_hits': 0, 'semantic_hits': 0,
                                   'fused_candidates': 0, 'reranked': False,
                                   'semantic_model_used': False}
        diagnostics = {
            'method': 'hybrid',
            'input_documents': len(docs),
            'indexed_passages': len(passages),
            'retrieved_passages': len(evidence),
            'embedding_model': getattr(embedder, 'model_name', 'injected-adapter') if embedder is not None else None,
            'reranker_requested': enable_reranker,
            **extra,
        }
        message = (f'RAG: HYBRID retrieved {len(evidence)} passages from '
                   f'{len(passages)} chunks; reranking={extra["reranked"]}; '
                   'retrieval models only, no generative LLM used')
    return {
        'evidence': evidence,
        'rag_diagnostics': diagnostics,
        'trace': [message],
    }
