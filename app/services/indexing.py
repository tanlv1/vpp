import faiss
import numpy as np


def create_faiss_index(feature_vectors: np.ndarray, index_path: str):
    """
    Create a Faiss index from the extracted feature vectors and save it to disk.

    :param feature_vectors: Array of feature vectors.
    :param index_path: Path to save the Faiss index.
    """
    dimension = feature_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(feature_vectors)
    faiss.write_index(index, index_path)


def load_faiss_index(index_path: str) -> faiss.IndexFlatL2:
    """
    Load a Faiss index from disk.

    :param index_path: Path to the Faiss index file.
    :return: Loaded Faiss index.
    """
    return faiss.read_index(index_path)
