"""Local ONNX retrieval model adapters, deliberately separate from Layer 3.

FASTEMBED models are downloaded on their first use (internet needed once), then
cached by FastEmbed locally. No OpenVINO/Qwen or cloud LLM calls are made here.
"""
from functools import lru_cache

EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
RERANK_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2'


@lru_cache(maxsize=1)
def get_embedder():
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise RuntimeError(
            'Hybrid RAG needs fastembed. Run: python -m pip install -r requirements-rag.txt'
        ) from e
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker():
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as e:
        raise RuntimeError(
            'Cross-encoder reranking needs fastembed. Install requirements-rag.txt'
        ) from e
    return TextCrossEncoder(model_name=RERANK_MODEL)
