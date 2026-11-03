"""Small manual smoke check using REAL local ONNX text embedding model."""
from edgemind.modules.rag import retrieve

state = {
    'query': 'How can the app work without an internet connection?',
    'documents': [
        {'source_id': 'offline', 'text': 'The application can execute its small model when the network is unavailable.'},
        {'source_id': 'graphics', 'text': 'A graphics processor speeds up machine learning computations.'},
        {'source_id': 'weather', 'text': 'Rain and thunderstorms may arrive later today.'},
    ],
}
for mode in ('bm25', 'hybrid'):
    result = retrieve({**state, 'rag_settings': {'mode': mode, 'top_k': 2}})
    print(f'{mode:>6} -> {[x["source_id"] for x in result["evidence"]]}')
    print('        ', result['rag_diagnostics'])
print('These are retrieved passages only, NOT verified answers.')
