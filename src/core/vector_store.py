"""
Vector store interface for the RAG system.
Qdrant backend implementation.
"""

import os
import logging
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add embeddings with metadata to the store"""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Get total number of vectors"""
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store implementation"""

    def __init__(
        self,
        collection_name: str = "course_embeddings",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            collection_name: Name of the Qdrant collection
            url: Qdrant server URL
            api_key: API key for authentication (optional)
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("Qdrant client not installed. Install with: pip install qdrant-client")

        if not url:
            raise ValueError("Qdrant URL is required. Set QDRANT_URL environment variable.")

        self.collection_name = collection_name
        self.vector_size: Optional[int] = None

        self.client = QdrantClient(url=url, api_key=api_key)

        collections = self.client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if not collection_exists:
            logger.info(
                "Qdrant collection '%s' does not exist. "
                "It will be created on first add_embeddings call.",
                collection_name,
            )
        else:
            self.vector_size = self._get_collection_vector_size(collection_name)
            logger.info(
                "Qdrant collection '%s' exists with vector size %s",
                collection_name,
                self.vector_size,
            )

        logger.debug("Initialized Qdrant client (url=%s, collection=%s)", url, collection_name)
        self._ensure_payload_indexes()

    def _get_collection_vector_size(self, collection_name: str) -> Optional[int]:
        """Extract vector size from collection configuration"""
        try:
            collection_info = self.client.get_collection(collection_name)
            params = collection_info.config.params
            vectors_config = getattr(params, "vectors", None) if hasattr(params, "vectors") else None

            if vectors_config is None and isinstance(params, dict):
                vectors_config = params.get("vectors")

            if vectors_config is None:
                return None

            # Handle dict-style config
            if isinstance(vectors_config, dict):
                if "size" in vectors_config:
                    return vectors_config["size"]
                if len(vectors_config) > 0:
                    first_vector = next(iter(vectors_config.values()))
                    if isinstance(first_vector, dict) and "size" in first_vector:
                        return first_vector["size"]
            else:
                # Object-style config
                vector_size = getattr(vectors_config, "size", None)
                if vector_size is not None:
                    return vector_size
                if hasattr(vectors_config, "__dict__"):
                    vectors_dict = vectors_config.__dict__
                    if vectors_dict:
                        first_vector = next(iter(vectors_dict.values()))
                        if hasattr(first_vector, "size"):
                            return first_vector.size

            return None
        except Exception as e:
            logger.warning("Could not determine vector size from collection info: %s", e)
            return None

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for commonly filtered fields"""
        try:
            from qdrant_client.models import PayloadSchemaType

            index_fields = ["language", "data_type", "course_code", "source"]
            created_indexes = []
            existing_indexes = []

            for field_name in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                    created_indexes.append(field_name)
                except Exception as e:
                    error_str = str(e).lower()
                    if "already exists" in error_str or "duplicate" in error_str:
                        existing_indexes.append(field_name)
                    else:
                        logger.warning("Could not create payload index for '%s': %s", field_name, e)
            
            if created_indexes:
                logger.debug(f"Created payload indexes: {', '.join(created_indexes)}")
            if existing_indexes:
                logger.debug(f"Payload indexes already exist: {', '.join(existing_indexes)}")
        except Exception as e:
            logger.warning("Error ensuring payload indexes: %s", e)

    def _ensure_collection(self, vector_size: int) -> None:
        """Ensure collection exists with correct configuration"""
        from qdrant_client.models import Distance, VectorParams

        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if not collection_exists:
            logger.info(
                "Creating Qdrant collection '%s' with vector size %s",
                self.collection_name,
                vector_size,
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            self.vector_size = vector_size
            self._ensure_payload_indexes()
        else:
            # Collection exists - validate or set vector size
            if self.vector_size is None:
                self.vector_size = self._get_collection_vector_size(self.collection_name)

            collection_info = self.client.get_collection(self.collection_name)
            points_count = collection_info.points_count

            if points_count == 0:
                # Empty collection - recreate with correct size
                logger.info(
                    "Collection '%s' exists but is empty. Recreating with vector size %s...",
                    self.collection_name,
                    vector_size,
                )
                try:
                    self.client.delete_collection(self.collection_name)
                except Exception as e:
                    logger.warning("Error deleting collection: %s", e)

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                self.vector_size = vector_size
                self._ensure_payload_indexes()
            elif self.vector_size is not None and self.vector_size != vector_size:
                # Collection has data but vector size mismatch
                raise ValueError(
                    f"Vector size mismatch: collection '{self.collection_name}' has vector size {self.vector_size}, "
                    f"but trying to add vectors of size {vector_size}. "
                    "Please clear the collection or use matching vector size."
                )
            else:
                # Collection exists with matching vector size
                if self.vector_size is None:
                    self.vector_size = vector_size
                logger.debug(
                    "Collection '%s' exists with vector size %s",
                    self.collection_name,
                    self.vector_size,
                )

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add embeddings with metadata to Qdrant"""
        try:
            from qdrant_client.models import PointStruct

            if len(embeddings) == 0:
                logger.warning("No embeddings to add")
                return False

            if len(embeddings.shape) > 1:
                vector_size = embeddings.shape[1]
            else:
                vector_size = len(embeddings)

            self._ensure_collection(vector_size)

            points = []

            for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
                data_type = meta.get("data_type", "course")
                string_id = f"{data_type}_{i}"

                hash_obj = hashlib.sha256(string_id.encode("utf-8"))
                point_id = int.from_bytes(hash_obj.digest()[:8], byteorder="big") % (2**64)

                payload = dict(meta)
                payload["_original_id"] = string_id

                vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
                points.append(point)

            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )

            logger.info("Added %s embeddings to Qdrant", len(embeddings))
            return True

        except Exception as e:
            logger.error("Error adding embeddings to Qdrant: %s", e)
            return False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """Search Qdrant collection"""
        try:
            from qdrant_client.models import Filter

            query_vector = (
                query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            )

            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count == 0:
                logger.error("Qdrant collection is empty!")
                return np.array([]), []

            if len(query_vector) == 0:
                logger.error("Query embedding is empty!")
                return np.array([]), []

            query_filter: Optional[Filter] = None
            if filter_metadata:
                query_filter = self._build_qdrant_filter(filter_metadata)

            query_params: Dict[str, Any] = {
                "collection_name": self.collection_name,
                "query": query_vector,
                "limit": top_k,
                "with_payload": True,
            }
            if query_filter is not None:
                query_params["query_filter"] = query_filter

            results = self.client.query_points(**query_params)

            if results and hasattr(results, "points") and results.points:
                similarities = np.array([point.score for point in results.points])
                indices: List[int] = []

                for point in results.points:
                    point_id = point.id
                    try:
                        original_id = point.payload.get("_original_id") if point.payload else None
                        if original_id:
                            if original_id.startswith("course_") or original_id.startswith("professor_"):
                                idx = int(original_id.split("_")[1])
                            else:
                                idx = int(original_id)
                        else:
                            if isinstance(point_id, str):
                                if point_id.startswith("course_") or point_id.startswith("professor_"):
                                    idx = int(point_id.split("_")[1])
                                else:
                                    idx = int(point_id)
                            else:
                                logger.warning(
                                    "Integer point ID %s without _original_id in payload, cannot determine index",
                                    point_id,
                                )
                                continue
                        indices.append(idx)
                    except (ValueError, IndexError, TypeError) as e:
                        logger.warning("Could not parse point ID: %s, error: %s", point_id, e)
                        continue

                return similarities, indices

            return np.array([]), []

        except Exception as e:
            logger.error("Error in Qdrant vector store search: %s", e)
            return np.array([]), []

    def _build_qdrant_filter(self, filter_metadata: Dict[str, Any]):
        """Convert filter metadata to Qdrant Filter"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Support {"$and": [{k: v}, {k2: v2}, ...]}
        if "$and" in filter_metadata:
            conditions = []
            for condition in filter_metadata["$and"]:
                for key, value in condition.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
            return Filter(must=conditions)

        # Simple flat {k: v, ...}
        conditions = []
        for key, value in filter_metadata.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        return Filter(must=conditions)

    def get_count(self) -> int:
        """Get total number of vectors"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error("Error getting Qdrant collection count: %s", e)
            return 0

    def clear(self) -> bool:
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            if self.vector_size:
                self._ensure_collection(self.vector_size)
            logger.info("Cleared Qdrant collection '%s'", self.collection_name)
            return True
        except Exception as e:
            logger.error("Error clearing Qdrant collection: %s", e)
            return False


def create_vector_store(**kwargs) -> VectorStore:
    """
    Factory function to create Qdrant vector store instance.

    Args:
        **kwargs: Arguments passed to QdrantVectorStore constructor.
                  - url: Qdrant server URL (from QDRANT_URL env var if not provided)
                  - api_key: Qdrant API key for authentication (from QDRANT_API_KEY env var if not provided)
                  - collection_name: Collection name (from QDRANT_COLLECTION_NAME env var if not provided)

    Returns:
        QdrantVectorStore instance
    """
    url = kwargs.get("url") or os.getenv("QDRANT_URL")
    api_key = kwargs.get("api_key") or os.getenv("QDRANT_API_KEY")
    collection_name = (
        kwargs.get("collection_name") or os.getenv("QDRANT_COLLECTION_NAME", "course_embeddings")
    )

    return QdrantVectorStore(
        collection_name=collection_name,
        url=url,
        api_key=api_key,
    )
