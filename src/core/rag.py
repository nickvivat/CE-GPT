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
from ..preprocess.data_processor import DataProcessor, DataChunk
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
from ..utils.performance_logger import csv_logger

logger = get_logger(__name__)

def get_chunk_content(chunk) -> str:
    """Get content from chunk, handling both course and professor data"""
    if hasattr(chunk, 'content') and chunk.content:
        return chunk.content
    elif hasattr(chunk, 'metadata') and chunk.metadata.get('data_type') == 'professor':
        # Generate content from professor metadata
        content_parts = []
        
        # Add name
        name = chunk.metadata.get('name', '')
        if name:
            content_parts.append(f"Professor: {name}")
        
        # Add degrees
        degrees = chunk.metadata.get('degrees', [])
        if degrees:
            degrees_text = " ".join(degrees)
            content_parts.append(f"Education: {degrees_text}")
        
        # Add research areas
        research_areas = chunk.metadata.get('research_areas', [])
        if research_areas:
            research_text = ", ".join(research_areas)
            content_parts.append(f"Research Areas: {research_text}")
        
        # Add teaching subjects
        teaching = chunk.metadata.get('teaching_subjects', [])
        if teaching:
            teaching_text = ", ".join(teaching)
            content_parts.append(f"Teaching: {teaching_text}")
        
        # Add textbooks
        textbooks = chunk.metadata.get('textbooks', [])
        if textbooks:
            textbook_text = ", ".join(textbooks)
            content_parts.append(f"Textbooks: {textbook_text}")
        
        return " | ".join(content_parts)
    else:
        return ""

class RAGSystem:
    """Main RAG system for multilingual course search and generation"""
    
    def __init__(self, use_reranker: bool = True, use_query_enhancement: bool = True, auto_load_data: bool = True):
        """Initialize the RAG system"""
        self.use_reranker = use_reranker
        self.use_query_enhancement = use_query_enhancement
        self.auto_load_data = auto_load_data
        
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
        
        # Auto-load data if enabled
        if self.auto_load_data:
            self._auto_load_data()
    
    def _auto_load_data(self):
        """Auto-load both course and professor data"""
        try:
            logger.info("Auto-loading course and professor data...")
            
            # Define data sources
            data_sources = [
                {'file_path': 'data/raw/course_detail.json', 'data_type': 'course'},
                {'file_path': 'data/raw/professor_detail.json', 'data_type': 'professor'}
            ]
            
            # Load multiple data sources
            success = self.load_multiple_data_sources(data_sources)
            
            if success:
                # Build vector index
                self.build_vector_index()
                logger.info("Auto-loading completed successfully")
            else:
                logger.warning("Auto-loading failed, but system will continue")
                
        except Exception as e:
            logger.error(f"Error during auto-loading: {e}")
            logger.warning("System will continue without auto-loaded data")
        
    @monitor_operation("data_loading")
    @handle_errors(ErrorType.DATA_PROCESSING, fallback_value=False)
    def load_and_process_data(self, data_file: str, data_type: str = "course") -> bool:
        """Load and process data using unified processor"""
        logger.info(f"Loading and processing {data_type} data...")
        
        # Try to load from cache first
        base_name = os.path.splitext(os.path.basename(data_file))[0]
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(data_file)), "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Use consistent naming: course_detail_processed.json and professor_detail_processed.json
        if data_type == "course":
            cache_file = os.path.join(processed_dir, "course_detail_processed.json")
        elif data_type == "professor":
            cache_file = os.path.join(processed_dir, "professor_detail_processed.json")
        else:
            cache_file = os.path.join(processed_dir, f"{base_name}_{data_type}_processed.json")
        
        if os.path.exists(cache_file):
            logger.info("Loading preprocessed chunks from cache...")
            self.chunks = self.data_processor.load_processed_chunks(cache_file)
            if self.chunks:
                logger.info(f"Loaded {len(self.chunks)} processed chunks from cache")
                stats = self.data_processor.get_statistics(self.chunks)
                logger.info(f"Data statistics: {stats}")
                return True
        
        # Process data if cache not available
        self.chunks = self.data_processor.process_file(data_file, data_type)
        if not self.chunks:
            logger.error(f"Failed to process {data_type} data")
            return False
            
        # Save to cache
        self.data_processor.save_processed_chunks(self.chunks, cache_file)
        
        stats = self.data_processor.get_statistics(self.chunks)
        logger.info(f"Data statistics: {stats}")
        
        return True
    
    def load_multiple_data_sources(self, data_sources: List[Dict[str, str]]) -> bool:
        """Load multiple data sources of different types with caching"""
        logger.info(f"Loading {len(data_sources)} data sources...")
        
        all_chunks = []
        
        for source in data_sources:
            file_path = source.get('file_path')
            data_type = source.get('data_type', 'course')
            
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            # Try to load from cache first
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            processed_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # Use consistent naming: course_detail_processed.json and professor_detail_processed.json
            if data_type == "course":
                cache_file = os.path.join(processed_dir, "course_detail_processed.json")
            elif data_type == "professor":
                cache_file = os.path.join(processed_dir, "professor_detail_processed.json")
            else:
                cache_file = os.path.join(processed_dir, f"{base_name}_processed.json")
            
            chunks = []
            if os.path.exists(cache_file):
                logger.info(f"Loading preprocessed {data_type} chunks from cache...")
                chunks = self.data_processor.load_processed_chunks(cache_file)
                if chunks:
                    logger.info(f"Loaded {len(chunks)} processed {data_type} chunks from cache")
            
            # Process data if cache not available
            if not chunks:
                logger.info(f"Processing {data_type} data from {file_path}...")
                chunks = self.data_processor.process_file(file_path, data_type)
                if chunks:
                    # Save to cache
                    self.data_processor.save_processed_chunks(chunks, cache_file)
                    logger.info(f"Saved {len(chunks)} processed {data_type} chunks to cache")
                else:
                    logger.warning(f"Failed to process {data_type} data from {file_path}")
                    continue
            
            all_chunks.extend(chunks)
            logger.info(f"Added {len(chunks)} {data_type} chunks to combined dataset")
        
        if not all_chunks:
            logger.error("No data loaded from any source")
            return False
        
        self.chunks = all_chunks
        stats = self.data_processor.get_statistics(self.chunks)
        logger.info(f"Combined data statistics: {stats}")
        
        return True
    
    def find_links_between_data_types(self, source_type: str, target_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """Find links between different data types"""
        if not self.chunks:
            logger.error("No data loaded")
            return {}
        
        return self.data_processor.find_links(self.chunks, source_type, target_type)
    
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
            chunks_content = "".join([chunk.content for chunk in self.chunks])
            current_hash = hashlib.md5(chunks_content.encode()).hexdigest()
            
            # Check if data has changed by comparing with cached hash
            data_changed = True
            if (os.path.exists(chunks_hash_file)):
                try:
                    with open(chunks_hash_file, 'r') as f:
                        cached_hash = f.read().strip()
                    
                    if cached_hash == current_hash:
                        data_changed = False
                        logger.info("Data has not changed, checking existing cache...")
                    else:
                        logger.info("Data has changed, will regenerate embeddings...")
                except Exception as e:
                    logger.warning(f"Failed to read cached hash: {e}, will regenerate...")
            
            # If data hasn't changed, check if we can use existing cache
            if not data_changed:
                # Check if vector store already has data
                if self.vector_store.get_count() > 0:
                    logger.info("Vector store already contains embeddings and data unchanged, skipping rebuild")
                    return True
                
                # Try to load cached embeddings if they exist
                if (os.path.exists(embeddings_cache_file) and 
                    os.path.getsize(embeddings_cache_file) > 0):
                    try:
                        logger.info("Loading embeddings from cache...")
                        self.embeddings = np.load(embeddings_cache_file)
                        logger.info(f"Loaded {len(self.embeddings)} embeddings from cache")
                    except Exception as e:
                        logger.warning(f"Failed to load cached embeddings: {e}, regenerating...")
                        self.embeddings = None
                else:
                    logger.info("No embedding cache found, generating embeddings...")
                    self.embeddings = None
            else:
                logger.info("Data changed, regenerating embeddings...")
                self.embeddings = None
            
            # Generate embeddings if not loaded from cache
            if self.embeddings is None:
                chunk_texts = [get_chunk_content(chunk) for chunk in self.chunks]
                self.embeddings = self.embedder.get_embeddings(chunk_texts)
                if self.embeddings is None or getattr(self.embeddings, "size", 0) == 0:
                    logger.error("Failed to generate embeddings")
                    return False
                
                # Save embeddings to cache (only when generating new embeddings)
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.save(embeddings_cache_file, self.embeddings)
                    with open(chunks_hash_file, 'w') as f:
                        f.write(current_hash)
                    logger.info(f"Saved {len(self.embeddings)} embeddings to cache")
                except Exception as e:
                    logger.warning(f"Failed to save embeddings cache: {e}")
            
            # Add to vector store only if it's empty
            if self.vector_store.get_count() == 0:
                chunk_metadata = [chunk.metadata for chunk in self.chunks]
                if not self.vector_store.add_embeddings(self.embeddings, chunk_metadata):
                    logger.error("Failed to add embeddings to vector store")
                    return False
                logger.info("Added embeddings to vector store")
            else:
                logger.info("Vector store already contains embeddings, skipping addition")
                
            logger.info("Vector index built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            return False
    
    @monitor_operation("vector_search")
    @handle_errors(ErrorType.VECTOR_SEARCH, fallback_value=[])
    def search(self, query: str, top_k: int = 5, language: str = None, 
               use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Search for relevant courses and professors with detailed performance logging"""
        start_time = time.time()
        query_lower = query.lower().strip()
        logger.info(f"Processing query: '{query}'")
        
        # Detect professor-related queries and increase top_k to ensure professor results are included
        professor_keywords = ['who', 'teach', 'teaches', 'instructor', 'professor', 'อาจารย์', 'สอน']
        is_professor_query = any(keyword in query_lower for keyword in professor_keywords)
        
        # Log all professor data for debugging
        if is_professor_query:
            professor_chunks = [chunk for chunk in self.chunks if getattr(chunk, 'data_type', chunk.metadata.get('data_type', 'unknown')) == 'professor']
        
        # Use circuit breaker for critical operations
        def _perform_search():
            current_query = query  # Store the original query
            enhanced_query = None
            classification = None
            detected_language = None
            
            # Step 1: Query Enhancement with timing
            query_enhancement_start = time.time()
            try:
                if self.use_query_enhancement and self.query and hasattr(self.query, 'available') and self.query.available:
                    enhanced_query = self.query.enhance_query(current_query, self.conversation_context)
                    
                    # Check if query was classified as conversational or external
                    if enhanced_query == current_query:
                        # Query was classified as 'pass' or 'external', return empty results
                        logger.info("Query classified as conversational or external, returning empty results")
                        classification = "pass"
                        return []
                    
                    # Query was enhanced, use the enhanced version
                    current_query = enhanced_query
                    classification = "enhanced"
                
                query_enhancement_duration = time.time() - query_enhancement_start
                csv_logger.log_query_enhancement(
                    query=query,
                    duration=query_enhancement_duration,
                    success=True,
                    original_query=query,
                    enhanced_query=enhanced_query,
                    classification=classification,
                    language=detected_language,
                    model_name=getattr(self.query, 'model_name', 'gemma3:4b-it-qat') if self.query else None
                )
                
            except Exception as e:
                query_enhancement_duration = time.time() - query_enhancement_start
                csv_logger.log_query_enhancement(
                    query=query,
                    duration=query_enhancement_duration,
                    success=False,
                    error_message=str(e),
                    original_query=query,
                    enhanced_query=enhanced_query,
                    classification=classification,
                    language=detected_language,
                    model_name=getattr(self.query, 'model_name', 'gemma3:4b-it-qat') if self.query else None
                )
                logger.error(f"Query enhancement failed: {e}")
            
            # Step 2: Language Detection
            if is_professor_query:
                detected_language = None  # Disable language filtering for professor queries
                logger.info("Professor query detected, disabling language filtering for better cross-language matching")
            else:
                detected_language = language
                if not detected_language:
                    detected_language = self._detect_language(query)
                    logger.info(f"Auto-detected language: {detected_language} for original query: '{query}'")
            
            # Step 3: Embedding and Vector Search with timing
            embedding_search_start = time.time()
            try:
                self.last_query = current_query
                if self.conversation_context:
                    self.conversation_context = f"Last query: {self.last_query}. Previous results: {len(self.last_results)} courses found."
                
                query_embedding = self.embedder.get_single_embedding(current_query)
                
                filter_metadata = None
                if detected_language:
                    filter_metadata = {"language": detected_language}
                    logger.info(f"Applying language filter: {detected_language} (based on original query)")
                else:
                    logger.info("No language filter applied - searching all data")
                
                similarities, indices = self.vector_store.search(query_embedding, top_k=top_k, filter_metadata=filter_metadata)
                
                embedding_search_duration = time.time() - embedding_search_start
                csv_logger.log_embedding_search(
                    query=query,
                    duration=embedding_search_duration,
                    success=True,
                    top_k=top_k,
                    results_count=len(similarities) if similarities.size > 0 else 0,
                    language_filter=detected_language,
                    embedding_model=getattr(self.embedder, 'model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                    vector_store_type=self.vector_store.__class__.__name__
                )
                
            except Exception as e:
                embedding_search_duration = time.time() - embedding_search_start
                csv_logger.log_embedding_search(
                    query=query,
                    duration=embedding_search_duration,
                    success=False,
                    error_message=str(e),
                    top_k=top_k,
                    results_count=0,
                    language_filter=detected_language,
                    embedding_model=getattr(self.embedder, 'model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                    vector_store_type=self.vector_store.__class__.__name__
                )
                logger.error(f"Embedding search failed: {e}")
                return []
            
            if not similarities.size:
                logger.warning("No results found in vector store")
                return []
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities, indices)):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    result = {
                        'content': get_chunk_content(chunk),
                        'metadata': chunk.metadata,
                        'similarity_score': float(similarity),
                        'chunk_id': chunk.chunk_id,
                        'original_index': chunk.original_index,
                        'data_type': chunk.metadata.get('data_type', 'unknown')
                    }
                    results.append(result)
            
            # Step 4: Reranking with timing
            if use_reranking and self.use_reranker and self.reranker:
                reranking_start = time.time()
                try:
                    logger.info("Applying reranking...")
                    input_count = len(results)
                    reranked_results = self.reranker.rerank_with_metadata(current_query, results)
                    results = []
                    for _, score, result_dict in reranked_results:
                        result_dict['rerank_score'] = float(score)
                        results.append(result_dict)
                    
                    reranking_duration = time.time() - reranking_start
                    csv_logger.log_reranking(
                        query=query,
                        duration=reranking_duration,
                        success=True,
                        input_count=input_count,
                        output_count=len(results),
                        reranker_model=getattr(self.reranker, 'model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                    )
                    
                except Exception as e:
                    reranking_duration = time.time() - reranking_start
                    csv_logger.log_reranking(
                        query=query,
                        duration=reranking_duration,
                        success=False,
                        error_message=str(e),
                        input_count=len(results),
                        output_count=0,
                        reranker_model=getattr(self.reranker, 'model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                    )
                    logger.error(f"Reranking failed: {e}")
            
            logger.info(f"Search completed, returned {len(results)} results")
            return results
        
        # Execute with circuit breaker and retry logic
        try:
            results = self.circuit_breaker.call(_perform_search)
            total_duration = time.time() - start_time
            
            # Log overall search performance
            csv_logger.log_overall_rag(
                query=query,
                duration=total_duration,
                success=True,
                total_steps=4,  # query_enhancement, embedding_search, reranking, overall
                total_duration=total_duration
            )
            
            logger.info(f"Search completed, returned {len(results)} results")
            return results
        except Exception as e:
            total_duration = time.time() - start_time
            csv_logger.log_overall_rag(
                query=query,
                duration=total_duration,
                success=False,
                error_message=str(e),
                total_steps=4,
                total_duration=total_duration
            )
            logger.error(f"Search failed: {e}")
            return []
    
    def generate_response(self, query: str, top_k: int = 5, language: str = None, 
                         use_reranking: bool = True, stream: bool = False, search_results: List[Dict[str, Any]] = None) -> str:
        """Generate a contextual response based on retrieved data with performance logging"""
        response_generation_start = time.time()
        
        try:
            if not self.response_generator:
                logger.warning("Response generator not available, falling back to search results")
                if not search_results:
                    search_results = self.search(query, top_k=top_k, language=language, use_reranking=use_reranking)
                
                response_duration = time.time() - response_generation_start
                
                csv_logger.log_response_generation(
                    query=query,
                    duration=response_duration,
                    success=True,
                    response_length=len(response) if response else 0,
                    language=language or self._detect_language(query),
                    model_name="fallback",
                    streaming=stream,
                    context_length=len(search_results) if search_results else 0
                )
                
                return response
            
            if language is None:
                language = self._detect_language(query)
            
            # Use provided search results or perform search if none provided
            if search_results is None:
                search_results = self.search(query, top_k=top_k, language=language, use_reranking=use_reranking)
            
            if not search_results:
                logger.warning(f"No search results found for query: '{query}'")
                response = self.response_generator.generate_conversational_response(query)
                response_duration = time.time() - response_generation_start
                
                csv_logger.log_response_generation(
                    query=query,
                    duration=response_duration,
                    success=True,
                    response_length=len(response) if response else 0,
                    language=language,
                    model_name=getattr(self.response_generator, 'model_name', 'gemma3:4b-it-qat'),
                    streaming=stream,
                    context_length=0
                )
                
                return response
            
            # Use streaming if requested
            if stream:
                response = self.response_generator.generate_response_stream(query, search_results, language)
            else:
                response = self.response_generator.generate_response(query, search_results, language)
            
            response_duration = time.time() - response_generation_start
            csv_logger.log_response_generation(
                query=query,
                duration=response_duration,
                success=True,
                response_length=len(response) if response else 0,
                language=language,
                model_name=getattr(self.response_generator, 'model_name', 'gemma3:4b-it-qat'),
                streaming=stream,
                context_length=len(search_results)
            )
            
            return response
                
        except Exception as e:
            response_duration = time.time() - response_generation_start
            logger.error(f"Error generating response: {e}")
            
            csv_logger.log_response_generation(
                query=query,
                duration=response_duration,
                success=False,
                error_message=str(e),
                response_length=0,
                language=language or self._detect_language(query),
                model_name=getattr(self.response_generator, 'model_name', 'gemma3:4b-it-qat'),
                streaming=stream,
                context_length=len(search_results) if search_results else 0
            )
            
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
    
    def get_csv_performance_summary(self, step: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary from CSV logs"""
        return csv_logger.get_performance_summary(step, hours)
    
    def export_performance_data_csv(self, output_file: str = None, hours: int = 24) -> str:
        """Export all performance data to CSV"""
        return csv_logger.export_data(output_file, hours)
    
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
