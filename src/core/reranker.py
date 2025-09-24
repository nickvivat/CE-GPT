import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
import logging
from ..utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.models.reranker_model
        self.device = "cpu"
        self.cache = {}  # Simple cache for reranking results
        self.cache_max_size = 50
        self.batch_size = 16  # Optimized batch size for reranking
        
        logger.info(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name, device=self.device)
        logger.info(f"Reranker model loaded on device: {self.device}")
    
    def prepare_query_passage_pairs(self, query: str, passages: List[str]) -> List[List[str]]:
        """Prepare query-passage pairs for reranking"""
        return [[query, passage] for passage in passages]
    
    def rerank(self, query: str, passages: List[str], scores: List[float] = None, 
               top_k: int = None) -> List[Tuple[int, float, str]]:
        """
        Rerank passages based on query relevance with caching and batch processing
        
        Args:
            query: The search query
            passages: List of passages to rerank
            scores: Original similarity scores (optional)
            top_k: Number of top results to return
            
        Returns:
            List of tuples: (original_index, new_score, passage)
        """
        if not passages:
            return []
        
        # Check cache first
        cache_key = f"{query}_{hash(tuple(passages))}"
        if cache_key in self.cache:
            logger.debug("Using cached reranking result")
            return self.cache[cache_key][:top_k] if top_k else self.cache[cache_key]
        
        # Prepare query-passage pairs
        pairs = self.prepare_query_passage_pairs(query, passages)
        
        # Process in batches for better performance
        rerank_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            try:
                batch_scores = self.model.predict(batch_pairs)
                rerank_scores.extend(batch_scores)
            except Exception as e:
                logger.error(f"Error processing reranking batch {i//self.batch_size + 1}: {e}")
                # Fill with zeros for failed batch
                rerank_scores.extend([0.0] * len(batch_pairs))
        
        # Create list of (index, score, passage) tuples
        results = [(i, score, passage) for i, (score, passage) in enumerate(zip(rerank_scores, passages))]
        
        # Sort by reranking score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Cache the result
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = results
        
        # Return top_k results
        return results[:top_k] if top_k else results
    
    def rerank_with_metadata(self, query: str, passages: List[Dict], 
                            top_k: int = None) -> List[Tuple[int, float, Dict]]:
        """
        Rerank passages with metadata
        
        Args:
            query: The search query
            passages: List of passage dictionaries with metadata
            top_k: Number of top results to return
            
        Returns:
            List of tuples: (original_index, new_score, passage_dict)
        """
        if not passages:
            return []
        
        # Use default top_k if not specified
        top_k = top_k or config.search.top_k_rerank
        
        # Extract text content for reranking
        passage_texts = [passage.get('content', '') for passage in passages]
        
        # Rerank
        reranked = self.rerank(query, passage_texts, top_k=top_k)
        
        # Return with metadata
        return [(idx, score, passages[idx]) for idx, score, _ in reranked]
    
    def batch_rerank(self, queries: List[str], passages_list: List[List[str]], 
                     batch_size: int = None) -> List[List[Tuple[int, float, str]]]:
        """
        Batch rerank multiple queries
        
        Args:
            queries: List of queries
            passages_list: List of passage lists for each query
            batch_size: Batch size for processing
            
        Returns:
            List of reranked results for each query
        """
        all_results = []
        
        # Use default batch_size if not specified
        batch_size = batch_size or config.processing.batch_size
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_passages = passages_list[i:i + batch_size]
            
            batch_results = []
            for query, passages in zip(batch_queries, batch_passages):
                result = self.rerank(query, passages)
                batch_results.append(result)
            
            all_results.extend(batch_results)
        
        return all_results
