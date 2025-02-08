import numpy
import sklearn.neighbors

from .distance import compute_distance, metrics as pd

from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Optional
import psutil

import numpy

class BaseANN(object):
    """Base class/interface for Approximate Nearest Neighbors (ANN) algorithms used in benchmarking."""

    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        pass

    def get_memory_usage(self) -> Optional[float]:
        """Returns the current memory usage of this ANN algorithm instance in kilobytes.

        Returns:
            float: The current memory usage in kilobytes (for backwards compatibility), or None if
                this information is not available.
        """

        return psutil.Process().memory_info().rss / 1024

    def fit(self, X: numpy.array) -> None:
        """Fits the ANN algorithm to the provided data. 

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            X (numpy.array): The data to fit the algorithm to.
        """
        pass

    def query(self, q: numpy.array, n: int) -> numpy.array:
        """Performs a query on the algorithm to find the nearest neighbors. 

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            q (numpy.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.

        Returns:
            numpy.array: An array of indices representing the nearest neighbors.
        """
        return []  # array of candidate indices

    def batch_query(self, X: numpy.array, n: int) -> None:
        """Performs multiple queries at once and lets the algorithm figure out how to handle it.

        The default implementation uses a ThreadPool to parallelize query processing.

        Args:
            X (numpy.array): An array of vectors to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return for each query.
        Returns: 
            None: self.get_batch_results() is responsible for retrieving batch result
        """
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self) -> numpy.array:
        """Retrieves the results of a batch query (from .batch_query()).

        Returns:
            numpy.array: An array of nearest neighbor results for each query in the batch.
        """
        return self.res

    def get_additional(self) -> Dict[str, Any]:
        """Returns additional attributes to be stored with the result.

        Returns:
            dict: A dictionary of additional attributes.
        """
        return {}

    def __str__(self) -> str:
        return self.name

class BruteForce(BaseANN):
    def __init__(self, metric):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self._metric = metric
        self.name = "BruteForce()"

    def fit(self, X):
        metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[self._metric]
        self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm="brute", metric=metric)
        self._nbrs.fit(X)

    def query(self, v, n):
        return list(self._nbrs.kneighbors([v], return_distance=False, n_neighbors=n)[0])

    def query_with_distances(self, v, n):
        (distances, positions) = self._nbrs.kneighbors([v], return_distance=True, n_neighbors=n)
        return zip(list(positions[0]), list(distances[0]))


class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""

    def __init__(self, metric, precision=numpy.float32):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForceBLAS doesn't support metric %s" % metric)
        elif metric == "hamming" and precision != numpy.bool_:
            raise NotImplementedError(
                "BruteForceBLAS doesn't support precision" " %s with Hamming distances" % precision
            )
        self._metric = metric
        self._precision = precision
        self.name = "BruteForceBLAS()"

    def fit(self, X):
        """Initialize the search index."""
        if self._metric == "angular":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            # normalize index vectors to unit length
            X /= numpy.sqrt(lens)[..., numpy.newaxis]
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == "hamming":
            # Regarding bitvectors as vectors in l_2 is faster for blas
            X = X.astype(numpy.float32)
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=numpy.float32)
            self.lengths = numpy.ascontiguousarray(lens, dtype=numpy.float32)
        elif self._metric == "euclidean":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"

    def query(self, v, n):
        return [index for index, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        """Find indices of `n` most similar vectors from the index to query
        vector `v`."""

        # use same precision for query as for index
        v = numpy.ascontiguousarray(v, dtype=self.index.dtype)

        # HACK we ignore query length as that's a constant
        # not affecting the final ordering
        if self._metric == "angular":
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)  # noqa
            dists = -numpy.dot(self.index, v)
        elif self._metric == "euclidean":
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab  # noqa
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == "hamming":
            # Just compute hamming distance using euclidean distance
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"
        # partition-sort by distance, get `n` closest
        nearest_indices = numpy.argpartition(dists, n)[:n]
        indices = [idx for idx in nearest_indices if pd[self._metric].distance_valid(dists[idx])]

        def fix(index):
            ep = self.index[index]
            ev = v
            return (index, pd[self._metric].distance(ep, ev))

        return map(fix, indices)