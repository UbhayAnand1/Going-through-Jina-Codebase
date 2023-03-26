from jina import Flow, Document
from jina.types.ndarray.generic import NdArray
from annlite import AnnoyIndexer

# Create an AnnoyIndexer with 10-dimensional embeddings and 1000 trees
indexer = AnnoyIndexer(10, 1000)

# Generate 1000 random documents, each with a random 10-dimensional embedding
docs = []
for i in range(1000):
    doc = Document()
    doc.embedding = NdArray(np.random.rand(10))
    docs.append(doc)

# Index the documents using ANNLite
indexer.index(docs)

# Create a Jina flow with a simple indexer and a RESTful gateway
flow = Flow().add(name='my_indexer', uses='jinahub+annlite_indexer_config.yml')
flow.add(name='my_gateway', uses='restful')

# Start the flow
with flow:
    # Index the documents using the ANNLite indexer
    flow.index(inputs=docs)

    # Query the index for the nearest neighbors of a new document
    query_doc = Document()
    query_doc.embedding = NdArray(np.random.rand(10))
    response = flow.search(inputs=[query_doc], return_results=True)

    # Print the results
    print(response[0].data.docs[0].matches)
