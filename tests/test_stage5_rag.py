"""Stage 5A independent tests: no LangGraph, no external dependency needed."""
import unittest
from edgemind.retrieval.bm25 import rank_bm25, tokenize
from edgemind.retrieval.chunking import chunk_documents
from edgemind.modules.rag import retrieve


class TestStage5Retrieval(unittest.TestCase):
    def test_tokenize_filters_common_stopwords(self):
        self.assertEqual(tokenize('What is the GPU?'), ['gpu'])

    def test_chunk_documents_preserves_unique_identifiers(self):
        docs = [{'source_id': 'paper-A', 'text': 'one two three four five six seven'}]
        chunks = chunk_documents(docs, chunk_words=4, overlap_words=1)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([c['source_id'] for c in chunks], ['paper-A#chunk-001', 'paper-A#chunk-002'])
        self.assertEqual(chunks[1]['text'], 'four five six seven')

    def test_document_shorter_than_chunk_stays_single(self):
        self.assertEqual(len(chunk_documents([{'source_id': 'a', 'text': 'hello world'}])), 1)

    def test_empty_documents(self):
        self.assertEqual(chunk_documents([]), [])

    def test_reject_invalid_chunk_sizes(self):
        with self.assertRaises(ValueError):
            chunk_documents([], chunk_words=4, overlap_words=4)
        with self.assertRaises(ValueError):
            chunk_documents([], chunk_words=0)

    def test_reject_invalid_documents(self):
        with self.assertRaises(ValueError):
            chunk_documents([{'source_id': '', 'text': 'hello'}])
        with self.assertRaises(ValueError):
            chunk_documents([{'source_id': 'a', 'text': None}])

    def test_bm25_ranks_relevant_document_first(self):
        docs = [
            {'source_id': 'z', 'text': 'GPU graphics hardware accelerates model inference GPU GPU'},
            {'source_id': 'a', 'text': 'Cloud costs involve billing and usage charges'},
            {'source_id': 'b', 'text': 'A CPU is a general processor'},
        ]
        ranked = rank_bm25('GPU inference', docs)
        self.assertEqual(ranked[0][1]['source_id'], 'z')

    def test_no_matching_passages(self):
        self.assertEqual(rank_bm25('banana', [{'source_id': 'a', 'text': 'green gpu'}]), [])

    def test_stable_sorting_for_ties(self):
        docs = [{'source_id': 'z', 'text': 'gpu'}, {'source_id': 'a', 'text': 'gpu'}]
        self.assertEqual([d['source_id'] for _, d in rank_bm25('gpu', docs)], ['a', 'z'])

    def test_top_k_applies(self):
        docs = [{'source_id': str(i), 'text': 'gpu compute'} for i in range(5)]
        self.assertEqual(len(rank_bm25('gpu', docs, top_k=2)), 2)

    def test_retrieve_returns_source_tracking(self):
        state = {'query': 'private document', 'documents': [
            {'source_id': 'privacy.txt', 'text': 'Private documents should remain on-device.'},
            {'source_id': 'hardware.txt', 'text': 'GPU acceleration is fast.'},
        ], 'rag_settings': {'top_k': 2}}
        response = retrieve(state)
        self.assertEqual(response['rag_diagnostics']['method'], 'bm25')
        self.assertEqual(response['evidence'][0]['source_id'], 'privacy.txt#chunk-001')
        self.assertIn('RAG: BM25', response['trace'][0])

    def test_retrieve_no_documents(self):
        response = retrieve({'query': 'hello'})
        self.assertEqual(response['evidence'], [])
        self.assertEqual(response['rag_diagnostics']['input_documents'], 0)

    def test_overlapping_passages_can_be_searched(self):
        docs = [{'source_id': 'long.txt', 'text': 'alpha beta gamma delta epsilon zeta eta theta'}]
        response = retrieve({'query': 'zeta theta', 'documents': docs,
                             'rag_settings': {'top_k': 3, 'chunk_words': 4, 'overlap_words': 1}})
        self.assertGreaterEqual(response['rag_diagnostics']['indexed_passages'], 2)
        self.assertTrue(response['evidence'])


if __name__ == '__main__':
    unittest.main()
