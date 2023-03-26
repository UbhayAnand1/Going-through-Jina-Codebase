from jina import Executor, DocumentArray
from annlite import AnnoyIndex

class ANNLiteExecutor(Executor):
    def __init__(self, index_path: str = 'annlite_index', n_trees: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.index_path = index_path
        self.n_trees = n_trees
        self.index = None

    def build(self, data: DocumentArray, *args, **kwargs):
        self.index = AnnoyIndex(data[0].embedding.shape[0])
        for idx, doc in enumerate(data):
            self.index.add_item(idx, doc.embedding.tolist())
        self.index.build(self.n_trees)

    def search(self, queryset: DocumentArray, *args, **kwargs):
        for query in queryset:
            idxs, distances = self.index.get_nns_by_vector(query.embedding.tolist(), 10, include_distances=True)
            query.matches = DocumentArray([Document(embedding=data[idx], scores={'euclidean': distance}) for idx, distance in zip(idxs, distances)])
