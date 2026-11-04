"""Actual cached FastEmbed models; offline loading never downloads weights."""
import os
import unittest
from unittest.mock import patch
import numpy as np

from edgemind.retrieval import local_models
from edgemind.retrieval.hybrid import rank_hybrid, cosine
from edgemind.modules.rag import retrieve

DOCUMENTS = [
    {'source_id': 'capital', 'text': 'Paris is the capital city of France.'},
    {'source_id': 'fruit', 'text': 'Bananas are yellow fruit and grow in warm climates.'},
    {'source_id': 'compute', 'text': 'A graphics processor accelerates parallel computing.'},
]


class CachedRetrievalTests(unittest.TestCase):
    def load(self, factory):
        if os.getenv('EDGEMIND_TEST_RETRIEVAL') != '1':
            self.skipTest('BLOCKED: dedicated cached-model tests not selected')
        with patch.dict(os.environ, {'EDGEMIND_RETRIEVAL_LOCAL_ONLY': '1', 'HF_HUB_OFFLINE': '1'}):
            try:
                return factory()
            except (ValueError, OSError, RuntimeError, ImportError) as exc:
                self.skipTest(f'BLOCKED: offline cached model could not load: {type(exc).__name__}: {exc}')

    def test_actual_embeddings_and_numpy_cosine(self):
        model = self.load(local_models.get_embedder)
        vectors = list(model.embed([item['text'] for item in DOCUMENTS]))
        self.assertEqual(len(vectors), 3)
        self.assertEqual(vectors[0].shape, (384,))
        self.assertTrue(np.isfinite(vectors).all())
        self.assertAlmostEqual(cosine(vectors[0], vectors[0]), 1.0, places=5)
        query_vector = list(model.embed(['Which city is the French capital?']))[0]
        scores = [cosine(query_vector, vector) for vector in vectors]
        self.assertGreater(scores[0], max(scores[1:]))
        self.evidence = {'model': local_models.EMBEDDING_MODEL, 'dimension': 384,
                         'semantic_scores': scores, 'source_ids': [d['source_id'] for d in DOCUMENTS],
                         'local_files_only': True}

    def test_actual_hybrid_retrieval(self):
        model = self.load(local_models.get_embedder)
        evidence, diagnostics = rank_hybrid('Which city is the French capital?', DOCUMENTS,
                                             model, top_k=1, candidate_k=3)
        self.assertEqual(evidence[0]['source_id'], 'capital')
        self.assertTrue(diagnostics['semantic_model_used'])
        self.evidence = diagnostics

    def test_actual_cross_encoder_scores(self):
        model = self.load(local_models.get_reranker)
        scores = list(model.rerank('What is the capital of France?', [d['text'] for d in DOCUMENTS]))
        self.assertEqual(len(scores), 3)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(scores[0]), max(float(x) for x in scores[1:]))
        self.evidence = {'model': local_models.RERANK_MODEL, 'scores': [float(s) for s in scores],
                         'local_files_only': True}

    def test_actual_hybrid_with_cross_encoder(self):
        embedder = self.load(local_models.get_embedder)
        reranker = self.load(local_models.get_reranker)
        state = {'query': 'What is the capital of France?', 'documents': DOCUMENTS,
                 'rag_settings': {'mode': 'hybrid', 'enable_reranker': True, 'top_k': 1, 'candidate_k': 3}}
        result = retrieve(state, embedder=embedder, reranker=reranker)
        self.assertTrue(result['evidence'][0]['source_id'].startswith('capital#chunk-'))
        self.assertTrue(result['rag_diagnostics']['reranked'])
        self.evidence = result['rag_diagnostics']
