"""Stage 5B algorithm tests run offline with deterministic mock embeddings.

The actual FastEmbed models are tested by the manual smoke tests on the user's PC.
"""
import unittest
from edgemind.retrieval.hybrid import cosine, fuse_rankings, rank_hybrid
from edgemind.modules.rag import retrieve


DOCUMENTS = [
    {'source_id': 'edge', 'text': 'A local inference engine runs when the network is unavailable.'},
    {'source_id': 'color', 'text': 'The ocean is blue and sandy beaches are warm.'},
    {'source_id': 'compute', 'text': 'Graphics processors accelerate neural network inference.'},
]


class FakeEmbedder:
    model_name = 'fake-test-embeddings'

    def embed(self, documents):
        for text in documents:
            s = text.lower()
            if 'graphics' in s or 'gpu' in s or 'neural' in s:
                yield [0., 1., 0.]
            elif 'network' in s or 'internet' in s or 'wifi' in s or 'offline' in s:
                yield [1., 0., 0.]
            elif 'ocean' in s or 'beaches' in s:
                yield [0., 0., 1.]
            else:
                yield [0.1, 0.1, 0.1]


class FakeReranker:
    def rerank(self, query, documents):
        for text in documents:
            yield 100. if 'local' in text.lower() else 0.


class TestStage5B(unittest.TestCase):
    def test_cosine_same_and_orthogonal(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0)

    def test_cosine_invalid_vectors(self):
        with self.assertRaises(ValueError):
            cosine([1], [1, 2])
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)

    def test_rrf_fuses_overlapping_rankings(self):
        results = fuse_rankings(['a', 'b'], ['b', 'a'])
        self.assertEqual([source_id for source_id, _ in results], ['a', 'b'])
        self.assertAlmostEqual(results[0][1], results[1][1])

    def test_rrf_rejects_duplicate_source_ids(self):
        with self.assertRaises(ValueError):
            fuse_rankings(['a', 'a'], ['b'])

    def test_semantic_finds_paraphrase_that_bm25_misses(self):
        # Neither 'wifi' nor 'offline' is present in the target passage.
        state = {'query': 'Can it still work with wifi offline?', 'documents': DOCUMENTS}
        lexical = retrieve(state)
        self.assertEqual(lexical['evidence'], [])
        state['rag_settings'] = {'mode': 'hybrid', 'top_k': 1}
        hybrid = retrieve(state, embedder=FakeEmbedder())
        self.assertEqual(hybrid['evidence'][0]['source_id'], 'edge#chunk-001')
        self.assertEqual(hybrid['rag_diagnostics']['method'], 'hybrid')
        self.assertTrue(hybrid['rag_diagnostics']['semantic_model_used'])

    def test_hybrid_keeps_source_ids(self):
        evidence, details = rank_hybrid('GPU', DOCUMENTS, FakeEmbedder(), top_k=2)
        self.assertEqual(evidence[0]['source_id'], 'compute')
        self.assertEqual(len(evidence), 2)
        self.assertEqual(details['reranked'], False)

    def test_reranker_changes_order(self):
        # Both candidates semantically tied; reranker emphasizes 'local'.
        evidence, details = rank_hybrid('GPU', DOCUMENTS, FakeEmbedder(),
                                        top_k=1, reranker=FakeReranker())
        self.assertEqual(evidence[0]['source_id'], 'edge')
        self.assertTrue(details['reranked'])

    def test_reject_invalid_reranker(self):
        class BrokenReranker:
            def rerank(self, query, documents):
                return [1.]  # Doesn't match number of candidate passages.
        with self.assertRaises(ValueError):
            rank_hybrid('GPU', DOCUMENTS, FakeEmbedder(), reranker=BrokenReranker())

    def test_hybrid_empty_documents_no_model_required(self):
        result = retrieve({'query': 'hi', 'documents': [],
                           'rag_settings': {'mode': 'hybrid'}})
        self.assertEqual(result['evidence'], [])
        self.assertFalse(result['rag_diagnostics']['semantic_model_used'])

    def test_invalid_hybrid_settings(self):
        with self.assertRaises(ValueError):
            retrieve({'query': 'hi', 'rag_settings': {'mode': 'unknown'}})
        with self.assertRaises(ValueError):
            retrieve({'query': 'hi', 'rag_settings': {'mode': 'bm25', 'enable_reranker': True}})
        with self.assertRaises(ValueError):
            rank_hybrid('test', DOCUMENTS, FakeEmbedder(), top_k=4, candidate_k=2)


if __name__ == '__main__':
    unittest.main()
