from jina.executors.decorators import single
from jina.executors.indexers.vector import BaseVectorIndexer

import annlite


@single
class ANNLiteExecutor(BaseVectorIndexer):
    """
    ANN Indexer powered by ANNLite library.

    For more information about ANNLite please refer to:
    https://github.com/jina-ai/ann-learn/blob/master/docs/source/annlite.rst

    :param metric: the distance metric to use, defaults to 'euclidean'
    :param n_jobs: the number of jobs to use for indexing, defaults to 1
    :param index_params: additional parameters for the index, defaults to None
    :param n_neighbors: number of nearest neighbors to retrieve, defaults to 10
    """

    def __init__(
        self,
        metric: str = 'euclidean',
        n_jobs: int = 1,
        index_params: dict = None,
        n_neighbors: int = 10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.n_jobs = n_jobs
        self.index_params = index_params
        self.n_neighbors = n_neighbors
        self.indexer = None

    def post_init(self):
        self.indexer = annlite.AnnoyIndex(metric=self.metric, n_jobs=self.n_jobs, **self.index_params)

    def add(self, keys, vectors, *args, **kwargs):
        for key, vector in zip(keys, vectors):
            self.indexer.add_item(key, vector)

        self.indexer.build(self.n_neighbors)

    def query(self, vectors, top_k, *args, **kwargs):
        results = []

        for vector in vectors:
            neighbors, distances = self.indexer.get_nns_by_vector(vector, top_k, search_k=-1, include_distances=True)

            results.append(list(zip(neighbors, distances)))

        return results
