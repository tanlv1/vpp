from app.models.object_metadata import ObjectMetadata
import faiss
import numpy as np
from elasticsearch import elasticsearch
from sqlalchemy.orm import Session

from app.models.object_metadata import ObjectMetadata
from app.services.feature_extraction import extract_features


def search_faiss_index(query_image: np.ndarray, index_path: str, top_k: int = 5) -> np.ndarray:
    """
    Search the Faiss index for the top-K similar images.

    :param query_image: The query image as a numpy array.
    :param index_path: Path to the Faiss index file.
    :param top_k: Number of top results to return.
    :return: Indices of the top-K similar images.
    """
    # Load the Faiss index
    index = faiss.read_index(index_path)

    # Extract features from the query image
    features = extract_features([query_image])

    # Search the Faiss index
    distances, indices = index.search(features, top_k)

    return indices[0]


def search_elasticsearch(query_text: str, index_name: str, top_k: int = 5):
    """
    Search the Elasticsearch index for the top-K matching documents.

    :param query_text: The text query.
    :param index_name: The name of the Elasticsearch index.
    :param top_k: Number of top results to return.
    :return: List of top-K matching documents.
    """
    es = Elasticsearch()

    response = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "content": query_text
                }
            },
            "size": top_k
        }
    )

    return [hit["_source"] for hit in response["hits"]["hits"]]


def combine_search_results(faiss_results, es_results, session: Session):
    """
    Combine results from Faiss and Elasticsearch and filter based on object metadata.

    :param faiss_results: List of indices from Faiss search.
    :param es_results: List of documents from Elasticsearch search.
    :param session: SQLAlchemy session for querying the database.
    :return: Combined and filtered list of results.
    """
    # Example of filtering based on object metadata (pseudo-code)
    filtered_results = []

    for result in faiss_results:
        obj_metadata = session.query(
            ObjectMetadata).filter_by(index=result).first()
        if obj_metadata and obj_metadata.meets_criteria:
            filtered_results.append(obj_metadata)

    # Optionally, combine with Elasticsearch results
    combined_results = filtered_results + es_results

    return combined_results
