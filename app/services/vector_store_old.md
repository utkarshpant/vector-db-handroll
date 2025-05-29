```py
import numpy as np
from collections import defaultdict


class VectorStore:
    def __init__(self, num_partitions=5):
        self.vector_data = {}
        self.index = defaultdict(list)
        self.num_partitions = num_partitions

    def add_vector(self, key, vector):
        """
        Add a `vector` with a given `key` to the vector store.
        @param key: Unique identifier for the vector.
        @param vector: The vector to be stored, should be a numpy array.
        """

        self.vector_data[key] = vector
        self.update_index(key, vector)

    def get_vector(self, key):
        """
        Retrieve a vector by its key.
        @param key: Unique identifier for the vector.
        @return: The vector associated with the key, or None if not found.
        """
        return self.vector_data.get(key, None)

    def _get_partition_key(self, vector):
        """
        Quantize each dimension to get the partition key for the vector.
        """

        normalized_vector = vector / \
            np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
        quantized = np.floor((normalized_vector + 1)/2 *
                             self.num_partitions).astype(int)

        quantized = np.clip(quantized, 0, self.num_partitions - 1)

        # return tuples of indices for the top-K dimensions
        k = min(3, len(vector))
        top_dims = np.argsort(np.abs(vector))[-k:]
        return tuple((dim, quantized[dim]) for dim in sorted(top_dims))

    def update_index(self, key, vector):
        """
        Update the index with the new vector.
        @param key: Unique identifier for the vector.
        @param vector: The vector to be indexed, should be a numpy array.
        """

        # get partition key for the vector
        partition_key = self._get_partition_key(vector)

        self.index[partition_key].append(key)

        # also add to neighboring partitions for better recall of similar vectors
        for i in range(len(partition_key)):
            dim, val = partition_key[i]

            if (val > 0):
                neighbor_key = list(partition_key)
                neighbor_key[i] = (dim, val - 1)
                self.index[tuple(neighbor_key)].append(key)

            if (val < self.num_partitions - 1):
                neighbor_key = list(partition_key)
                neighbor_key[i] = (dim, val + 1)
                self.index[tuple(neighbor_key)].append(key)

    def find_similar_vectors(self, vector, limit=5):
        """
        Find the most similar vectors to the given vector.
        @param vector: The vector to compare against.
        @param limit: The maximum number of similar vectors to return.
        @return: A list of tuples (key, similarity) sorted by similarity.
        """
        if (len(self.vector_data) == 0):
            return []
        
        # Get the partition key for the input vector
        partition_key = self._get_partition_key(vector)

        candidate_keys = set()
        if partition_key in self.index:
            candidate_keys.update(self.index[partition_key])

        # If still not enough candidates, use all vectors
        if len(candidate_keys) < limit:
            candidate_keys = set(self.vector_data.keys())

        similarities = []
        for key in candidate_keys:
            candidate_vector = self.vector_data[key]
            if candidate_vector is not None:
                similarity = np.dot(vector, candidate_vector) / \
                    (np.linalg.norm(vector) * np.linalg.norm(candidate_vector))
                similarities.append((key, similarity))
        
        # Sort by similarity and return the top `limit` results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
```