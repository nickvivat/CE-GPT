#!/usr/bin/env python3
"""
Multilingual RAG System for Computer Engineering Courses
Main system that orchestrates all components
"""

import os
import time
import hashlib
import json
import numpy as np
from typing import List, Dict, Any, Optional

# Core components
from .data_processor import DataProcessor
from .embedder import Embedder
from .reranker import Reranker
from .query import Query
from .vector_store import create_vector_store
from .generator import ResponseGenerator

# Utilities
from ..utils.config import config
from ..utils.logger import get_logger
from ..utils.error_handler import (
    handle_errors, 
    ErrorType, 
    CircuitBreaker
)
from ..utils.performance_monitor import monitor_operation, performance_monitor

logger = get_logger(__name__)

class RAGSystem:
    """Main RAG system for multilingual course search and generation"""
    
    def __init__(self, use_reranker: bool = True, use_query_enhancement: bool = True):
        """Initialize the RAG system"""
        self.use_reranker = use_reranker
        self.use_query_enhancement = use_query_enhancement
        
        # Initialize error handling components
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        
        # Initialize components with error handling
        try:
            self.data_processor = DataProcessor()
            self.embedder = Embedder()
            
            # Initialize reranker with error handling
            if use_reranker:
                try:
                    self.reranker = Reranker()
                    logger.info("Reranker initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize reranker: {e}. Continuing without reranking.")
                    self.reranker = None
            else:
                self.reranker = None
            
            # Initialize query enhancer with error handling
            if use_query_enhancement:
                try:
                    self.query = Query()
                    logger.info("Query enhancer initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize query enhancer: {e}. Continuing without query enhancement.")
                    self.query = None
            else:
                self.query = None
            
            # Initialize vector store
            self.vector_store = create_vector_store()
            
            # Initialize response generator
            try:
                self.response_generator = ResponseGenerator()
                logger.info("Response generator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize response generator: {e}. Continuing without generation.")
                self.response_generator = None
            
            logger.info("RAG system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise e
        
        # Data storage
        self.chunks = None
        self.embeddings = None
        
        # Conversation context
        self.conversation_context = ""
        self.last_query = ""
        self.last_results = []
        
    @monitor_operation("data_loading")
    @handle_errors(ErrorType.DATA_PROCESSING, fallback_value=False)
    def load_and_process_data(self, data_file: str) -> bool:
        """Load and process course data"""
        logger.info("Loading and processing course data...")
        
        # Try to load from cache first
        base_name = os.path.splitext(os.path.basename(data_file))[0]  # Extract base filename without extension
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(data_file)), "processed")  # Go up to data/ then into processed/
        os.makedirs(processed_dir, exist_ok=True)  # Ensure processed directory exists
        cache_file = os.path.join(processed_dir, f"{base_name}_processed.json")
        
        if os.path.exists(cache_file):
            logger.info("Loading preprocessed chunks from cache...")
            self.chunks = self.data_processor.load_processed_chunks(cache_file)
            if self.chunks:
                logger.info(f"Loaded {len(self.chunks)} processed chunks from cache")
                stats = self.data_processor.get_statistics(self.chunks)
                logger.info(f"Data statistics: {stats}")
                return True
        
        # Process data if cache not available
        if not self.data_processor.load_course_data(data_file):
            logger.error("Failed to load course data")
            return False
            
        courses = self.data_processor.load_course_data(data_file)
        self.chunks = self.data_processor.process_courses(courses)
        if not self.chunks:
            logger.error("Failed to process course data")
            return False
            
        # Save to cache
        self.data_processor.save_processed_chunks(self.chunks, cache_file)
        
        stats = self.data_processor.get_statistics(self.chunks)
        logger.info(f"Data statistics: {stats}")
        
        return True
    
    def build_vector_index(self) -> bool:
        """Build vector index from processed chunks"""
        try:
            if not self.chunks:
                logger.error("No chunks available for indexing")
                return False
                
            logger.info("Building vector index...")
            
            # Check if embeddings cache exists and is valid
            cache_dir = config.cache.embeddings_dir
            embeddings_cache_file = os.path.join(cache_dir, "course_embeddings.npy")
            chunks_hash_file = os.path.join(cache_dir, "chunks_hash.txt")
            
            # Calculate hash of current chunks to check if cache is still valid
            chunks_content = "".join([chunk.content + str(chunk.metadata) for chunk in self.chunks])
            current_hash = hashlib.md5(chunks_content.encode()).hexdigest()
            
            # Try to load cached embeddings if they exist and hash matches
            if (os.path.exists(embeddings_cache_file) and 
                os.path.exists(chunks_hash_file) and
                os.path.getsize(embeddings_cache_file) > 0):
                
                try:
                    with open(chunks_hash_file, 'r') as f:
                        cached_hash = f.read().strip()
                    
                    if cached_hash == current_hash:
                        logger.info("Loading embeddings from cache...")
                        self.embeddings = np.load(embeddings_cache_file)
                        logger.info(f"Loaded {len(self.embeddings)} embeddings from cache")
                        
                        # Check if vector store already has data
                        if self.vector_store.get_count() > 0:
                            logger.info("Vector store already contains embeddings, skipping rebuild")
                            return True
                        else:
                            logger.info("Vector store empty, adding cached embeddings...")
                    else:
                        logger.info("Data changed, regenerating embeddings...")
                        self.embeddings = None
                except Exception as e:
                    logger.warning(f"Failed to load cached embeddings: {e}, regenerating...")
                    self.embeddings = None
            else:
                logger.info("No embedding cache found, generating embeddings...")
                self.embeddings = None
            
            # Generate embeddings if not loaded from cache
            if self.embeddings is None:
                chunk_texts = self.data_processor.get_chunk_texts(self.chunks)
                self.embeddings = self.embedder.get_embeddings(chunk_texts)
                if self.embeddings is None or getattr(self.embeddings, "size", 0) == 0:
                    logger.error("Failed to generate embeddings")
                    return False
                
                # Save embeddings to cache
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.save(embeddings_cache_file, self.embeddings)
                    with open(chunks_hash_file, 'w') as f:
                        f.write(current_hash)
                    logger.info(f"Saved {len(self.embeddings)} embeddings to cache")
                except Exception as e:
                    logger.warning(f"Failed to save embeddings cache: {e}")
            
            # Add to vector store
            chunk_metadata = self.data_processor.get_chunk_metadata(self.chunks)
            if not self.vector_store.add_embeddings(self.embeddings, chunk_metadata):
                logger.error("Failed to add embeddings to vector store")
                return False
                
            logger.info("Vector index built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            return False
    
    @monitor_operation("vector_search")
    @handle_errors(ErrorType.VECTOR_SEARCH, fallback_value=[])
    def search(self, query: str, top_k: int = 5, language: str = None, 
               use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Search for relevant courses"""
        # Only skip truly conversational queries, not valid search terms
        conversational_queries = ['hi', 'hello', 'hey', 'สวัสดี', 'how are you', 'how are you doing']
        query_lower = query.lower().strip()
        
        if query_lower in conversational_queries:
            logger.info(f"Query '{query}' classified as conversational, skipping search")
            return []
        
        # Check if it's a legitimate course-related query
        course_keywords = [
            'calculus', 'math', 'mathematics', 'programming', 'coding', 'software', 
            'hardware', 'circuit', 'electronics', 'ai', 'artificial intelligence', 
            'machine learning', 'แคลคูลัส', 'คณิตศาสตร์', 'เขียนโปรแกรม', 'ซอฟต์แวร์',
            'ฮาร์ดแวร์', 'วงจร', 'อิเล็กทรอนิกส์', 'ปัญญาประดิษฐ์', 'การเรียนรู้ของเครื่อง'
        ]
        
        # If the query contains course-related keywords, it's definitely not conversational
        contains_course_keywords = any(keyword in query_lower for keyword in course_keywords)
        
        if contains_course_keywords:
            logger.info(f"Query '{query}' contains course keywords, proceeding with search")
        else:
            logger.info(f"Query '{query}' proceeding with search (may be conversational but allowing search)")
        
        # Use circuit breaker for critical operations
        def _perform_search():
            current_query = query  # Store the original query
            
            # Use Gemma to intelligently classify and enhance the query
            if self.use_query_enhancement and self.query and hasattr(self.query, 'available') and self.query.available:
                enhanced_query = self.query.enhance_query(current_query, self.conversation_context)
                
                # Don't skip based on length - let the search work
                current_query = enhanced_query
            
            self.last_query = current_query
            if self.conversation_context:
                self.conversation_context = f"Last query: {self.last_query}. Previous results: {len(self.last_results)} courses found."
            
            query_embedding = self.embedder.get_single_embedding(current_query)
            
            filter_metadata = None
            if language:
                filter_metadata = {"language": language}
            
            similarities, indices = self.vector_store.search(query_embedding, top_k=top_k, filter_metadata=filter_metadata)
            
            if not similarities.size:
                logger.warning("No results found in vector store")
                return []
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities, indices)):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    result = {
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'similarity_score': float(similarity),
                        'chunk_id': chunk.chunk_id,
                        'original_index': chunk.original_index
                    }
                    results.append(result)
            
            if use_reranking and self.use_reranker and self.reranker:
                logger.info("Applying reranking...")
                reranked_results = self.reranker.rerank_with_metadata(current_query, results)
                results = []
                for _, score, result_dict in reranked_results:
                    result_dict['rerank_score'] = float(score)
                    results.append(result_dict)
            
            logger.info(f"Search completed, returned {len(results)} results")
            return results
        
        # Execute with circuit breaker and retry logic
        try:
            results = self.circuit_breaker.call(_perform_search)
            logger.info(f"Search completed, returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def generate_response(self, query: str, top_k: int = 5, language: str = None, 
                         use_reranking: bool = True, stream: bool = False, search_results: List[Dict[str, Any]] = None) -> str:
        """Generate a contextual response based on retrieved data"""
        try:
            if not self.response_generator:
                logger.warning("Response generator not available, falling back to search results")
                if not search_results:
                    search_results = self.search(query, top_k=top_k, language=language, use_reranking=use_reranking)
                return self.response_generator._format_fallback_response(query, search_results)
            
            if language is None:
                language = self._detect_language(query)
            
            # Use provided search results or perform search if none provided
            if search_results is None:
                search_results = self.search(query, top_k=top_k, language=language, use_reranking=use_reranking)
            
            if not search_results:
                logger.warning(f"No search results found for query: '{query}'")
                return self.response_generator.generate_conversational_response(query)
            
            # Use streaming if requested
            if stream:
                return self.response_generator.generate_response_stream(query, search_results, language)
            else:
                return self.response_generator.generate_response(query, search_results, language)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to search results only
            try:
                if not search_results:
                    search_results = self.search(query, top_k=top_k, language=language, use_reranking=use_reranking)
                return self.response_generator._format_fallback_response(query, search_results, language)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำขอของคุณ" if language == "th" else "Sorry, an error occurred while processing your request."
    
    def generate_response_stream(self, query: str, top_k: int = 5, language: str = None, 
                               use_reranking: bool = True, search_results: List[Dict[str, Any]] = None):
        """Generate a streaming response generator for real-time output"""
        try:
            if not self.response_generator:
                logger.warning("Response generator not available, falling back to search results")
                if not search_results:
                    search_results = self.search(query, top_k=top_k, language=language, use_reranking=use_reranking)
                # Return fallback response as a single chunk
                fallback_response = self.response_generator._format_fallback_response(query, search_results)
                yield fallback_response
                return
            
            if language is None:
                language = self._detect_language(query)
            
            # Use provided search results or perform search if none provided
            if search_results is None:
                search_results = self.search(query, top_k=top_k, language=language, use_reranking=use_reranking)
            
            if not search_results:
                logger.warning(f"No search results found for query: '{query}'")
                conversational_response = self.response_generator.generate_conversational_response(query)
                yield conversational_response
                return
            
            # Get streaming response from generator
            for chunk in self.response_generator.generate_response_stream_generator(query, search_results, language):
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            # Fallback to search results only
            try:
                if not search_results:
                    search_results = self.search(query, top_k=top_k, language=language, use_reranking=use_reranking)
                fallback_response = self.response_generator._format_fallback_response(query, search_results, language)
                yield fallback_response
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                error_msg = "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำขอของคุณ" if language == "th" else "Sorry, an error occurred while processing your request."
                yield error_msg
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        try:
            status = {
                'total_chunks': len(self.chunks) if self.chunks else 0,
                'vector_store_type': self.vector_store.__class__.__name__,
                'vector_store_count': self.vector_store.get_count() if hasattr(self.vector_store, 'get_count') else 0,
                'reranker_enabled': self.use_reranker,
                'query_enhancement_enabled': self.use_query_enhancement,
                'response_generation_enabled': self.response_generator is not None,
                'conversation_context': self.conversation_context,
                'statistics': self.data_processor.get_statistics(self.chunks) if self.chunks else {},
                'circuit_breaker_state': self.circuit_breaker.get_state().value,
                'performance_metrics': performance_monitor.get_operation_stats()
            }
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary."""
        return performance_monitor.get_performance_summary()
    
    def export_performance_data(self, filepath: str):
        """Export performance data to file."""
        performance_monitor.export_metrics(filepath)
    
    def clear_conversation_context(self):
        """Clear conversation context"""
        self.conversation_context = ""
        self.last_query = ""
        self.last_results = []
        logger.info("Conversation context cleared")
    
    def clear_embedding_cache(self):
        """Clear embedding cache to force regeneration"""
        try:
            cache_dir = config.cache.embeddings_dir
            embeddings_cache_file = os.path.join(cache_dir, "course_embeddings.npy")
            chunks_hash_file = os.path.join(cache_dir, "chunks_hash.txt")
            
            if os.path.exists(embeddings_cache_file):
                os.remove(embeddings_cache_file)
                logger.info("Removed embeddings cache file")
            
            if os.path.exists(chunks_hash_file):
                os.remove(chunks_hash_file)
                logger.info("Removed chunks hash file")
            
            if hasattr(self.vector_store, 'clear'):
                self.vector_store.clear()
                logger.info("Cleared vector store")
            
            logger.info("Embedding cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing embedding cache: {e}")
            return False

    def _detect_language(self, text: str) -> str:
        """Detect if text is Thai or English"""
        thai_chars = set('กขคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะาิีึืุูเแโใไๆ')
        text_chars = set(text)
        if text_chars.intersection(thai_chars):
            return "th"
        return "en"
