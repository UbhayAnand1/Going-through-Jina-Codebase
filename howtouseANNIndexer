from jina import Flow, DocumentArray

# Define the flow
f = Flow().add(
    name='indexer',
    uses='jinahub+docker://ANNIndexer',
    uses_with={
        'index_file_name': 'vec.idx',
        'dim': 128,
        'index_label': '__label__',
        'metric': 'euclidean',
        'backend': 'annoy',
        'n_trees': 50,
    }
)

# Create the data
data = DocumentArray([
    {"embedding": [0.1, 0.2, 0.3]},
    {"embedding": [0.4, 0.5, 0.6]},
    {"embedding": [0.7, 0.8, 0.9]}
])

# Use the flow to index the data
with f:
    f.index(inputs=data)

# Define the query flow
qf = Flow().add(
    name='searcher',
    uses='jinahub+docker://ANNIndexer',
    uses_with={
        'index_file_name': 'vec.idx',
        'dim': 128,
        'index_label': '__label__',
        'metric': 'euclidean',
        'backend': 'annoy',
        'n_trees': 50,
    }
)

# Use the query flow to search for similar vectors
with qf:
    results = qf.search(inputs=data, return_results=True)

# Print the search results
for result in results:
    print(result)
