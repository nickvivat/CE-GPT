"""
Vector store interface for the RAG system.
ChromaDB backend implementation.
"""

import os
import logging
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
	def search(self, query_embedding: np.ndarray, top_k: int, 
			   filter_metadata: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[int]]:
		"""Search for similar vectors"""
		pass
	
	@abstractmethod
	def get_count(self) -> int:
		"""Get total number of vectors"""
		pass

class ChromaVectorStore(VectorStore):
	"""Chroma-based vector store implementation"""
	
	def __init__(self, persist_directory: str = "./chroma_db"):
		try:
			import chromadb
			from chromadb.config import Settings
		except ImportError:
			raise ImportError("ChromaDB not installed. Install with: pip install chromadb")
		
		self.persist_directory = persist_directory
		self.client = chromadb.PersistentClient(
			path=persist_directory,
			settings=Settings(anonymized_telemetry=False)
		)
		
		# Create or get collection
		self.collection = self.client.get_or_create_collection(
			name="course_embeddings",
			metadata={"hnsw:space": "cosine"}
		)
		
		logger.info(f"Initialized ChromaDB at {persist_directory}")
	
	def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
		"""Add embeddings with metadata to ChromaDB"""
		try:
			# Convert numpy arrays to lists for ChromaDB
			embeddings_list = embeddings.tolist()
			
			# Create IDs for each embedding based on data type
			ids = []
			for i, meta in enumerate(metadata):
				data_type = meta.get('data_type', 'course')
				if data_type == 'professor':
					ids.append(f"professor_{i}")
				else:
					ids.append(f"course_{i}")
			
			# Convert metadata to ChromaDB-compatible format
			chroma_metadata = []
			for meta in metadata:
				chroma_meta = {}
				for key, value in meta.items():
					if isinstance(value, list):
						# Convert lists to comma-separated strings
						chroma_meta[key] = ", ".join(str(v) for v in value)
					else:
						chroma_meta[key] = value
				chroma_metadata.append(chroma_meta)
			
			# Add to collection
			self.collection.add(
				embeddings=embeddings_list,
				metadatas=chroma_metadata,
				ids=ids
			)
			
			logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")
			return True
			
		except Exception as e:
			logger.error(f"Error adding embeddings to ChromaDB: {e}")
			return False
	
	def search(self, query_embedding: np.ndarray, top_k: int, 
			   filter_metadata: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[int]]:
		"""Search ChromaDB collection"""
		try:
			# Convert query embedding to list
			query_embedding_list = query_embedding.tolist()
			
			# Check collection status
			if self.collection.count() == 0:
				logger.error("ChromaDB collection is empty!")
				return np.array([]), []
			
			# Check if query embedding is valid
			if query_embedding.size == 0:
				logger.error("Query embedding is empty!")
				return np.array([]), []
			
			# Perform similarity search
			query_params = {
				"query_embeddings": [query_embedding_list],
				"n_results": top_k
			}
			
			# Only add where clause if filter_metadata is provided
			if filter_metadata is not None:
				query_params["where"] = filter_metadata
			
			results = self.collection.query(**query_params)
			
			if results['ids'] and results['ids'][0]:
				# Extract similarities and indices
				similarities = np.array(results['distances'][0])
				indices = []
				
				for id_str in results['ids'][0]:
					try:
						if id_str.startswith('course_') or id_str.startswith('professor_'):
							idx = int(id_str.split('_')[1])
						else:
							idx = int(id_str)
						indices.append(idx)
					except (ValueError, IndexError):
						logger.warning(f"Could not parse chunk ID: {id_str}")
						continue
				
				return similarities, indices
			else:
				return np.array([]), []
				
		except Exception as e:
			logger.error(f"Error in vector store search: {e}")
			return np.array([]), []
	
	def get_count(self) -> int:
		"""Get total number of vectors"""
		return self.collection.count()

def create_vector_store(store_type: str = "chroma", **kwargs) -> VectorStore:
	"""Factory function to create vector store instances"""
	if store_type.lower() == "chroma":
		return ChromaVectorStore(**kwargs)
	else:
		raise ValueError(f"Only ChromaDB is supported. Got: {store_type}")
