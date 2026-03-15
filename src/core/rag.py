#!/usr/bin/env python3
"""
Multilingual RAG System for Computer Engineering Courses
Main system that orchestrates all components
"""

import os
import re
import time
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ..preprocess.data_processor import DataProcessor
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
        
        # Add teaching subjects
        teaching = chunk.metadata.get('teaching_subjects', [])
        if teaching:
            teaching_text = ", ".join(teaching)
            content_parts.append(f"Teaching: {teaching_text}")
        
        return " | ".join(content_parts)
    else:
        return ""

def _bm25_tokenize(text: str) -> List[str]:
    """Tokenize text for BM25: keep 8-digit course codes as single tokens, split rest."""
    if not text or not text.strip():
        return []

    tokens = re.findall(r"\d{8}|[a-zA-Z0-9\u0E00-\u0E7F]+", text)
    return [t.lower() for t in tokens if t]

class RAGSystem:
    """Main RAG system for multilingual course search and generation"""
    
    def __init__(
        self,
        use_reranker: bool = True,
        use_query_enhancement: bool = True,
        auto_load_data: bool = True,
        chat_history_manager = None
    ):
        """Initialize the RAG system"""
        self.use_reranker = use_reranker
        self.use_query_enhancement = use_query_enhancement
        self.auto_load_data = auto_load_data
        self.chat_history_manager = chat_history_manager
        
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        
        try:
            self.data_processor = DataProcessor()
            self.embedder = Embedder()
            
            initialized_components = []
            
            if use_reranker:
                try:
                    self.reranker = Reranker()
                    initialized_components.append("reranker")
                except Exception as e:
                    logger.warning(f"Failed to initialize reranker: {e}. Continuing without reranking.")
                    self.reranker = None
            else:
                self.reranker = None
            
            if use_query_enhancement:
                try:
                    self.query = Query()
                    initialized_components.append("query_enhancer")
                except Exception as e:
                    logger.warning(f"Failed to initialize query enhancer: {e}. Continuing without query enhancement.")
                    self.query = None
            else:
                self.query = None
            
            self.vector_store = create_vector_store()
            initialized_components.append("vector_store")
            
            try:
                self.response_generator = ResponseGenerator()
                initialized_components.append("response_generator")
            except Exception as e:
                logger.warning(f"Failed to initialize response generator: {e}. Continuing without generation.")
                self.response_generator = None
            
            logger.info(f"RAG system initialized: {', '.join(initialized_components)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise e
        
        self.chunks = None
        self.embeddings = None
        self._bm25_index = None  # BM25Okapi instance for hybrid search
        self._bm25_corpus_tokenized = None  # tokenized docs for BM25
        self.conversation_context = ""
        self.last_query = ""
        self.last_results = []
        
        if self.auto_load_data:
            self._auto_load_data()
        
        # Update response generator with chat history manager if available
        if self.response_generator and self.chat_history_manager:
            self.response_generator.chat_history_manager = self.chat_history_manager
    
    def _auto_load_data(self):
        """Auto-load course, professor, curriculum, and study plan data"""
        try:
            data_sources = [
                {'file_path': 'data/raw/course_detail.json', 'data_type': 'course'},
                {'file_path': 'data/raw/professor_detail.json', 'data_type': 'professor'},
                {'file_path': 'data/raw/curriculum.json', 'data_type': 'curriculum'},
                {'file_path': 'data/raw/studyplan.json', 'data_type': 'studyplan'},
            ]
            
            success = self.load_multiple_data_sources(data_sources)
            
            if success:
                self.build_vector_index()
                logger.info("Data loading completed successfully")
            else:
                logger.warning("Data loading failed, but system will continue")
                
        except Exception as e:
            logger.error(f"Error during data loading: {e}")
            logger.warning("System will continue without auto-loaded data")
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from current chunks for hybrid search."""
        if not self.chunks:
            self._bm25_index = None
            self._bm25_corpus_tokenized = None
            return
        try:
            from rank_bm25 import BM25Okapi
            corpus = [get_chunk_content(chunk) for chunk in self.chunks]
            self._bm25_corpus_tokenized = [_bm25_tokenize(doc) for doc in corpus]
            self._bm25_index = BM25Okapi(
                self._bm25_corpus_tokenized,
                k1=config.search.bm25_k1,
                b=config.search.bm25_b
            )
            logger.info(f"Built BM25 index for {len(self.chunks)} chunks (hybrid search)")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}. Hybrid search disabled.")
            self._bm25_index = None
            self._bm25_corpus_tokenized = None
    
    def _chunk_passes_filter(self, chunk_index: int, filter_metadata: Optional[Dict[str, Any]]) -> bool:
        """Check if chunk at index passes the given metadata filter (same logic as Qdrant)."""
        # Index -1 denotes OCR/document points (no chunk); include them in hybrid merge
        if chunk_index == -1:
            return True
        if not filter_metadata or chunk_index < 0 or not self.chunks or chunk_index >= len(self.chunks):
            return not filter_metadata
        meta = self.chunks[chunk_index].metadata
        if "$and" in filter_metadata:
            for condition in filter_metadata["$and"]:
                for key, value in condition.items():
                    if meta.get(key) != value:
                        return False
            return True
        for key, value in filter_metadata.items():
            if meta.get(key) != value:
                return False
        return True
    
    def _hybrid_merge_rrf(
        self,
        vector_indices: List[int],
        vector_similarities: np.ndarray,
        bm25_indices: List[int],
        filter_metadata: Optional[Dict[str, Any]],
        top_k: int,
        rrf_k: Optional[int] = None,
    ) -> Tuple[List[int], Dict[int, float]]:
        """Merge vector and BM25 results using Reciprocal Rank Fusion. Returns (ordered_indices, index_to_similarity)."""
        if rrf_k is None:
            rrf_k = config.search.rrf_k
        rank_v = {idx: (r + 1) for r, idx in enumerate(vector_indices)}
        rank_b = {idx: (r + 1) for r, idx in enumerate(bm25_indices)}
        combined_indices = set(vector_indices) | set(bm25_indices)
        sim_map = {}
        for r, idx in enumerate(vector_indices):
            if r < (vector_similarities.size if hasattr(vector_similarities, 'size') else len(vector_similarities)):
                sim_map[idx] = float(vector_similarities[r])
        rrf_scores = []
        for idx in combined_indices:
            if not self._chunk_passes_filter(idx, filter_metadata):
                continue
            rv = rank_v.get(idx)
            rb = rank_b.get(idx)
            rrf = (1.0 / (rrf_k + rv) if rv else 0.0) + (1.0 / (rrf_k + rb) if rb else 0.0)
            rrf_scores.append((idx, rrf, sim_map.get(idx, rrf)))
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        ordered = [x[0] for x in rrf_scores[:top_k]]
        # Preserve original vector similarity when available for downstream
        index_to_similarity = {x[0]: x[2] for x in rrf_scores[:top_k]}
        return ordered, index_to_similarity
        
    @monitor_operation("data_loading")
    @handle_errors(ErrorType.DATA_PROCESSING, fallback_value=False)
    def load_and_process_data(self, data_file: str, data_type: str = "course") -> bool:
        """Load and process data using unified processor"""
        logger.info(f"Loading and processing {data_type} data...")
        
        base_name = os.path.splitext(os.path.basename(data_file))[0]
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(data_file)), "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        if data_type == "course":
            cache_file = os.path.join(processed_dir, "course_detail_processed.json")
        elif data_type == "professor":
            cache_file = os.path.join(processed_dir, "professor_detail_processed.json")
        else:
            cache_file = os.path.join(processed_dir, f"{base_name}_{data_type}_processed.json")
        
        if os.path.exists(cache_file):
            self.chunks = self.data_processor.load_processed_chunks(cache_file)
            if self.chunks:
                stats = self.data_processor.get_statistics(self.chunks)
                logger.info(f"Loaded {len(self.chunks)} processed chunks from cache - {stats}")
                return True
        
        self.chunks = self.data_processor.process_file(data_file, data_type)
        if not self.chunks:
            logger.error(f"Failed to process {data_type} data")
            return False
            
        self.data_processor.save_processed_chunks(self.chunks, cache_file)
        
        stats = self.data_processor.get_statistics(self.chunks)
        logger.info(f"Data statistics: {stats}")
        
        return True
    
    def load_multiple_data_sources(self, data_sources: List[Dict[str, str]]) -> bool:
        """Load multiple data sources of different types with caching"""
        all_chunks = []
        
        for source in data_sources:
            file_path = source.get('file_path')
            data_type = source.get('data_type', 'course')
            
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            processed_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            if data_type == "course":
                cache_file = os.path.join(processed_dir, "course_detail_processed.json")
            elif data_type == "professor":
                cache_file = os.path.join(processed_dir, "professor_detail_processed.json")
            else:
                cache_file = os.path.join(processed_dir, f"{base_name}_processed.json")
            
            chunks = []
            if os.path.exists(cache_file):
                chunks = self.data_processor.load_processed_chunks(cache_file)
                if chunks:
                    logger.info(f"Loaded {len(chunks)} {data_type} chunks from cache")
            
            if not chunks:
                logger.info(f"Processing {data_type} data from {file_path}...")
                chunks = self.data_processor.process_file(file_path, data_type)
                if chunks:
                    self.data_processor.save_processed_chunks(chunks, cache_file)
                    logger.info(f"Processed and cached {len(chunks)} {data_type} chunks")
                else:
                    logger.warning(f"Failed to process {data_type} data from {file_path}")
                    continue
            
            all_chunks.extend(chunks)
        
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
            
            cache_dir = config.cache.embeddings_dir
            embeddings_cache_file = os.path.join(cache_dir, "course_embeddings.npy")
            chunks_hash_file = os.path.join(cache_dir, "chunks_hash.txt")
            
            chunks_content = "".join([chunk.content for chunk in self.chunks])
            chunks_metadata = "".join([str(chunk.metadata) for chunk in self.chunks])
            current_hash = hashlib.md5((chunks_content + chunks_metadata).encode()).hexdigest()
            
            data_changed = True
            if os.path.exists(chunks_hash_file):
                try:
                    with open(chunks_hash_file, 'r') as f:
                        cached_hash = f.read().strip()
                    
                    if cached_hash == current_hash:
                        data_changed = False
                        logger.debug("Data has not changed, checking existing cache...")
                    else:
                        logger.debug("Data has changed, will regenerate embeddings...")
                except Exception as e:
                    logger.warning(f"Failed to read cached hash: {e}, will regenerate...")
            
            if not data_changed:
                if self.vector_store.get_count() > 0:
                    logger.debug("Vector store already contains embeddings and data unchanged, skipping rebuild")
                    if config.search.use_hybrid_search:
                        self._build_bm25_index()
                    return True
                
                if os.path.exists(embeddings_cache_file) and os.path.getsize(embeddings_cache_file) > 0:
                    try:
                        self.embeddings = np.load(embeddings_cache_file)
                        logger.debug(f"Loaded {len(self.embeddings)} embeddings from cache")
                    except Exception as e:
                        logger.warning(f"Failed to load cached embeddings: {e}, regenerating...")
                        self.embeddings = None
                else:
                    logger.debug("No embedding cache found, generating embeddings...")
                    self.embeddings = None
            else:
                logger.debug("Data changed, regenerating embeddings...")
                self.embeddings = None
            
            if self.embeddings is None:
                chunk_texts = [get_chunk_content(chunk) for chunk in self.chunks]
                self.embeddings = self.embedder.get_embeddings(chunk_texts)
                if self.embeddings is None or getattr(self.embeddings, "size", 0) == 0:
                    logger.error("Failed to generate embeddings")
                    return False
                
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.save(embeddings_cache_file, self.embeddings)
                    with open(chunks_hash_file, 'w') as f:
                        f.write(current_hash)
                    logger.info(f"Saved {len(self.embeddings)} embeddings to cache")
                except Exception as e:
                    logger.warning(f"Failed to save embeddings cache: {e}")
            
            if data_changed and self.vector_store.get_count() > 0:
                logger.info("Data changed, clearing vector store before adding new embeddings")
                self.vector_store.clear()
            
            if self.vector_store.get_count() == 0:
                chunk_metadata = []
                for chunk in self.chunks:
                    meta = dict(chunk.metadata)
                    meta['content'] = get_chunk_content(chunk)
                    chunk_metadata.append(meta)
                if not self.vector_store.add_embeddings(self.embeddings, chunk_metadata):
                    logger.error("Failed to add embeddings to vector store")
                    return False
                logger.info(f"Vector index built: {len(self.embeddings)} embeddings indexed")
            else:
                logger.debug("Vector store already contains embeddings, skipping addition")
            # Build BM25 index for hybrid search when chunks are ready
            if config.search.use_hybrid_search:
                self._build_bm25_index()
            return True
            
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            return False
    
    @monitor_operation("vector_search")
    @handle_errors(ErrorType.VECTOR_SEARCH, fallback_value=[])
    async def search(
        self,
        query: str,
        top_k: int = 10,
        language: str = None,
        use_reranking: bool = True,
        session_id: str = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant courses and professors with detailed performance logging"""
        start_time = time.time()
        query_lower = query.lower().strip()
        logger.info(f"Processing query: '{query}'")
        
        professor_keywords = ['who', 'teach', 'teaches', 'instructor', 'professor', 'อาจารย์', 'สอน', 'who is', 'who are', 'teaching']
        is_professor_query = any(keyword in query_lower for keyword in professor_keywords)
        
        async def _perform_search():
            current_query = query
            enhanced_query = None
            metadata = None
            classification = None
            detected_language = None
            
            # Load chat history BEFORE query enhancement so it can be used for context
            # With 128k context window, we can include more context for better query enhancement
            conversation_context = None
            if session_id and self.chat_history_manager:
                recent_messages = self.chat_history_manager.get_recent_messages(session_id, n=10)
                if recent_messages:
                    # Remove current query if it's the last message
                    messages_to_use = recent_messages
                    if query and recent_messages:
                        last_msg = recent_messages[-1]
                        if last_msg.role == "user" and last_msg.content.strip() == query.strip():
                            messages_to_use = recent_messages[:-1]
                    
                    if messages_to_use:
                        # Extract course codes from all messages
                        all_codes = set()
                        for msg in messages_to_use:
                            codes = self._extract_course_codes_from_query(msg.content)
                            all_codes.update(codes)
                        
                        # Build context: course codes + last 3-4 messages (recent exchange)
                        context_parts = []
                        if all_codes:
                            context_parts.append(f"Course codes mentioned: {', '.join(sorted(all_codes))}")
                            context_parts.append("")
                        
                        # Include last 3-4 messages for better context (recent exchange)
                        recent_exchange = messages_to_use[-4:] if len(messages_to_use) >= 4 else messages_to_use
                        for msg in recent_exchange:
                            role_label = "User" if msg.role == "user" else "Assistant"
                            # Include more content (500 chars) since we have context room
                            content = msg.content[:500] if len(msg.content) > 500 else msg.content
                            if len(msg.content) > 500:
                                content = content + "..."
                            context_parts.append(f"{role_label}: {content}")
                        
                        if context_parts:
                            conversation_context = "\n".join(context_parts)
                            logger.debug(f"Loaded conversation context from session {session_id} for query enhancement ({len(recent_exchange)} messages)")
            
            query_enhancement_start = time.time()
            try:
                if is_professor_query:
                    logger.info("Professor query detected, skipping query enhancement")
                    classification = "professor_skip_enhancement"
                elif self.use_query_enhancement and self.query and hasattr(self.query, 'available') and self.query.available:
                    enhanced_query, metadata = await self.query.enhance_query_async(current_query, conversation_context)
                    
                    if metadata and metadata.get("query_intent") == "abusing":
                        logger.warning("Abusive query detected. Rejecting.")
                        raise ValueError("ABUSIVE_QUERY")
                        
                    if metadata and metadata.get("query_intent") in ["conversational", "external"]:
                        if enhanced_query == current_query:
                            logger.info(f"Query classified as {metadata.get('query_intent')}, returning empty results")
                            classification = metadata.get("query_intent", "pass")
                            return []
                        else:
                            codes_in_original = set(self._extract_course_codes_from_query(current_query))
                            codes_in_enhanced = set(self._extract_course_codes_from_query(enhanced_query))
                            codes_added = codes_in_enhanced - codes_in_original
                            
                            if not codes_added:
                                logger.info(f"Query classified as {metadata.get('query_intent')} with non-course modification, returning empty results")
                                classification = metadata.get("query_intent", "pass")
                                return []
                            logger.info(f"Query classified as {metadata.get('query_intent')} but has course codes appended, proceeding to search")
                            classification = metadata.get("query_intent", "pass")
                            current_query = enhanced_query
                    elif metadata and metadata.get("query_intent") == "course_search" and metadata.get("tags") == ["clear_query"]:
                        logger.info("Query classified as pass (clear query) - searching database without enhancement")
                        classification = "pass"
                        if enhanced_query != current_query:
                            logger.info(f"Updating current_query with course codes for search: {enhanced_query}")
                            current_query = enhanced_query
                    else:
                        current_query = enhanced_query
                        classification = "enhanced"
                else:
                    logger.info("Query enhancement disabled, treating as pass (direct search)")
                    classification = "pass"
                
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
                    enhanced_query=current_query if 'enhanced_query' not in locals() else enhanced_query,
                    classification=classification if 'classification' in locals() else "unknown",
                    language=detected_language,
                    model_name=getattr(self.query, 'model_name', 'gemma3:4b-it-qat') if self.query else None
                )
                logger.error(f"Query enhancement failed: {e}")
                if str(e) == "ABUSIVE_QUERY":
                    raise ValueError("ABUSIVE_QUERY")
            
            detected_language = language
            if not detected_language:
                detected_language = self._detect_language(query)
                logger.info(f"Auto-detected language: {detected_language} for original query: '{query}'")
            
            embedding_search_start = time.time()
            try:
                self.last_query = current_query
                
                # Store conversation context for response generation (already loaded above for query enhancement)
                if conversation_context:
                    self.conversation_context = conversation_context
                else:
                    self.conversation_context = f"Last query: {self.last_query}. Previous results: {len(self.last_results)} courses found."
                
                query_embedding = self.embedder.get_single_embedding(current_query)
                
                filter_metadata = {}
                if detected_language:
                    filter_metadata["language"] = detected_language
                    logger.info(f"Applying language filter: {detected_language} (based on original query)")
                
                # Curriculum/studyplan intent: use primary context (curriculum or studyplan) + course as helper
                primary_type = None
                if metadata and metadata.get("query_intent") in ("curriculum_search", "studyplan_search"):
                    primary_type = metadata.get("query_intent").replace("_search", "")
                    logger.info(f"Primary context from metadata: {primary_type}")
                elif self._is_curriculum_query(query):
                    primary_type = "curriculum"
                    logger.info("Primary context from keyword detection: curriculum (graduation/requirements)")
                elif self._is_studyplan_query(query):
                    primary_type = "studyplan"
                    logger.info("Primary context from keyword detection: studyplan (per-semester plan)")
                
                if metadata and metadata.get("tags"):
                    tags = metadata.get("tags", [])
                    query_intent = metadata.get("query_intent", "course_search")
                    
                    if is_professor_query and query_intent != "professor_search":
                        query_intent = "professor_search"
                        logger.info(f"Overriding query_intent to 'professor_search' based on early detection")
                    
                    if primary_type:
                        logger.info(f"Using {primary_type} as primary context with course as helper")
                    elif query_intent == "professor_search":
                        filter_metadata["data_type"] = "professor"
                        logger.info("Applying professor filter")
                    else:
                        filter_metadata["data_type"] = "course"
                        logger.info("Applying course filter")
                    
                    logger.info(f"Metadata tags available: {tags} (intent: {query_intent})")
                elif is_professor_query:
                    filter_metadata["data_type"] = "professor"
                    logger.info("Applying professor filter based on early detection (no metadata)")
                elif not primary_type:
                    filter_metadata["data_type"] = "course"
                    logger.info("Applying course filter (default)")
                
                if len(filter_metadata) > 1:
                    and_conditions = []
                    for key, value in filter_metadata.items():
                        and_conditions.append({key: value})
                    filter_metadata = {"$and": and_conditions}
                    logger.info(f"Combined filters using $and: {filter_metadata}")
                
                if not filter_metadata:
                    logger.info("No filters applied - searching all data")
                else:
                    logger.info(f"Final filter metadata: {filter_metadata}")
                
                if primary_type:
                    and_conditions_primary = [{"data_type": primary_type}]
                    filter_primary = and_conditions_primary[0]
                    top_k_primary = top_k
                    sim_primary, idx_primary, pay_primary = self.vector_store.search(query_embedding, top_k=top_k_primary, filter_metadata=filter_primary)
                    result_indices = list(idx_primary)
                    result_similarities = (sim_primary.tolist() if hasattr(sim_primary, 'tolist') else list(sim_primary))
                    result_payloads = list(pay_primary)
                    logger.info(f"Primary-only search ({primary_type}): {len(idx_primary)} chunks (no course/professor helper)")
                else:
                    similarities, indices, payloads = self.vector_store.search(query_embedding, top_k=top_k, filter_metadata=filter_metadata if filter_metadata else None)
                    
                    # Hybrid search: merge vector + BM25 with Reciprocal Rank Fusion
                    result_indices = list(indices)
                    result_similarities = similarities.tolist() if hasattr(similarities, 'tolist') else list(similarities)
                    result_payloads = list(payloads) if payloads else [None] * len(result_indices)
                    if config.search.use_hybrid_search and self.chunks and current_query:
                        if self._bm25_index is None:
                            self._build_bm25_index()
                        if self._bm25_index:
                            try:
                                query_tokens = _bm25_tokenize(current_query)
                                if query_tokens:
                                    bm25_scores = self._bm25_index.get_scores(query_tokens)
                                    bm25_top_n = min(
                                        top_k * config.search.bm25_top_k_multiplier, 
                                        len(self.chunks), 
                                        config.search.bm25_max_candidates
                                    )
                                    bm25_ranked = np.argsort(bm25_scores)[::-1][:bm25_top_n]
                                    bm25_indices = [int(i) for i in bm25_ranked if bm25_scores[i] > 0]
                                    ordered_indices, index_to_similarity = self._hybrid_merge_rrf(
                                        list(indices),
                                        similarities,
                                        bm25_indices,
                                        filter_metadata if filter_metadata else None,
                                        top_k,
                                    )
                                    result_indices = ordered_indices
                                    result_similarities = [index_to_similarity.get(i, 0.0) for i in ordered_indices]
                                    # Align payloads with ordered_indices (same order as vector search; multiple -1 map to different payloads)
                                    result_payloads = []
                                    for idx in ordered_indices:
                                        for j, orig_idx in enumerate(indices):
                                            if orig_idx == idx:
                                                result_payloads.append(payloads[j] if j < len(payloads) else None)
                                                break
                                        else:
                                            result_payloads.append(None)
                                    logger.debug(f"Hybrid search merged vector + BM25: {len(ordered_indices)} results")
                            except Exception as e:
                                logger.warning(f"Hybrid merge failed, using vector-only results: {e}")
                
                embedding_search_duration = time.time() - embedding_search_start
                csv_logger.log_embedding_search(
                    query=query,
                    duration=embedding_search_duration,
                    success=True,
                    top_k=top_k,
                    results_count=len(result_indices),
                    language_filter=detected_language,
                    embedding_model=getattr(self.embedder, 'model_name', 'embeddinggemma:latest'),
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
                    embedding_model=getattr(self.embedder, 'model_name', 'embeddinggemma:latest'),
                    vector_store_type=self.vector_store.__class__.__name__
                )
                logger.error(f"Embedding search failed: {e}")
                return []
            
            if not result_indices:
                logger.warning("No results found in vector store")
                return []
            
            results = []
            for i, (idx, similarity) in enumerate(zip(result_indices, result_similarities)):
                payload = result_payloads[i] if i < len(result_payloads) else None
                # Use payload from Qdrant when present (e.g. OCR studyplan points with "text" and data_type)
                if payload is not None and (payload.get("text") is not None or payload.get("content") is not None):
                    content = payload.get("text") or payload.get("content", "")
                    metadata = {k: v for k, v in payload.items() if k != "_original_id"}
                    result = {
                        'content': content,
                        'metadata': metadata,
                        'similarity_score': float(similarity),
                        'chunk_id': payload.get("_original_id", ""),
                        'original_index': -1,
                        'data_type': metadata.get('data_type', 'unknown')
                    }
                    results.append(result)
                elif idx >= 0 and idx < len(self.chunks):
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
            
            # Prioritize exact course code matches over semantic search
            course_codes_in_query = self._extract_course_codes_from_query(current_query)
            exact_matches = []
            unfound_course_codes = []
            suggestions_metadata = {}
            
            if course_codes_in_query:
                exact_matches = self._find_exact_course_code_matches(course_codes_in_query)
                exact_codes_found = {r['metadata'].get('course_code') for r in exact_matches}
                unfound_course_codes = [code for code in course_codes_in_query if code not in exact_codes_found]
                
                if exact_matches:
                    logger.info(f"Found {len(exact_matches)} exact course code match(es) for: {course_codes_in_query}")
                    # Remove semantic results that match the same course codes to avoid duplicates
                    results = [r for r in results if r.get('metadata', {}).get('course_code') not in exact_codes_found]
                    results = exact_matches + results
                else:
                    logger.info(f"No exact matches found for course codes: {course_codes_in_query}")
                    # Boost semantic results that mention the course codes
                    if results:
                        results = self._boost_results_with_course_codes(results, course_codes_in_query)
                
                if unfound_course_codes:
                    for code in unfound_course_codes:
                        similar_codes = self._find_similar_course_codes(code, max_distance=2, max_results=5)
                        if similar_codes:
                            suggestions_metadata[code] = similar_codes
                            logger.info(f"Found {len(similar_codes)} similar course codes for {code}: {similar_codes}")
            
            # Add suggestions metadata before filtering so it's preserved
            if suggestions_metadata or unfound_course_codes:
                suggestions_result = {
                    'content': '',
                    'metadata': {},
                    'similarity_score': 1.0,  # Ensures it passes any threshold
                    'chunk_id': 'suggestions_metadata',
                    'original_index': -1,
                    'data_type': 'metadata',
                    'suggestions': suggestions_metadata,
                    'unfound_course_codes': unfound_course_codes,
                    'bypass_score_filter': True
                }
                results.append(suggestions_result)
            
            # Filter by similarity threshold, but always keep exact matches and metadata results
            similarity_threshold = config.search.similarity_threshold
            if results and similarity_threshold > 0:
                original_count = len(results)
                exact_match_results = [r for r in results if r.get('is_exact_match', False)]
                metadata_results = [r for r in results if r.get('data_type') == 'metadata' or r.get('bypass_score_filter', False)]
                other_results = [r for r in results if not r.get('is_exact_match', False) and r.get('data_type') != 'metadata' and not r.get('bypass_score_filter', False)]
                
                filtered_other = [r for r in other_results if r.get('similarity_score', 0.0) >= similarity_threshold]
                results = exact_match_results + filtered_other + metadata_results
                filtered_count = len(results)
                if original_count != filtered_count:
                    logger.info(f"Pre-filtered by similarity: {original_count} -> {filtered_count} results (threshold: {similarity_threshold:.3f}, kept {len(exact_match_results)} exact matches, {len(metadata_results)} metadata entries)")
            
            results = self._deduplicate_results(results)
            
            if use_reranking and self.use_reranker and self.reranker:
                reranking_start = time.time()
                try:
                    logger.info("Applying reranking...")
                    input_count = len(results)
                    
                    # Separate metadata results and exact matches - they should not be reranked
                    metadata_results = [r for r in results if r.get('data_type') == 'metadata' or r.get('bypass_score_filter', False)]
                    exact_match_results = [r for r in results if r.get('is_exact_match', False)]
                    rerankable_results = [r for r in results if r.get('data_type') != 'metadata' and not r.get('bypass_score_filter', False) and not r.get('is_exact_match', False)]
                    
                    # Rerank only non-metadata results
                    reranked_results = self.reranker.rerank_with_metadata(current_query, rerankable_results, top_k=top_k)
                    results = []
                    for _, score, result_dict in reranked_results:
                        result_dict['rerank_score'] = float(score)
                        # Calculate hybrid score combining similarity and rerank scores
                        similarity = result_dict.get('similarity_score', 0.0)
                        rerank = result_dict.get('rerank_score', 0.0)
                        # Weighted combination: 30% similarity, 70% rerank (rerank is more accurate)
                        hybrid_score = (0.3 * similarity) + (0.7 * rerank)
                        result_dict['hybrid_score'] = float(hybrid_score)
                        results.append(result_dict)
                    
                    # Sort by hybrid score for better accuracy
                    results.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
                    
                    # Add exact matches and metadata results back (always preserve these)
                    # Exact matches first (highest priority), then reranked results, then metadata
                    results = exact_match_results + results + metadata_results
                    
                    reranking_duration = time.time() - reranking_start
                    csv_logger.log_reranking(
                        query=query,
                        duration=reranking_duration,
                        success=True,
                        input_count=input_count,
                        output_count=len(results),
                        reranker_model=getattr(self.reranker, 'model_name', 'BAAI/bge-reranker-v2-m3')
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
                        reranker_model=getattr(self.reranker, 'model_name', 'BAAI/bge-reranker-v2-m3')
                    )
            # Final truncation to strictly respect top_k
            return results[:top_k]
        
        try:
            results = await _perform_search()
            total_duration = time.time() - start_time
            
            csv_logger.log_overall_rag(
                query=query,
                duration=total_duration,
                success=True,
                total_steps=4,  # query_enhancement, embedding_search, reranking, overall
                total_duration=total_duration
            )
            
            logger.info(f"Search completed: {len(results)} results in {total_duration:.2f}s")
            self.last_results = results
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
    
    async def generate_response(
        self,
        query: str,
        top_k: int = 10,
        language: str = None,
        use_reranking: bool = True,
        stream: bool = False,
        search_results: List[Dict[str, Any]] = None,
        session_id: str = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate a contextual response based on retrieved data with performance logging"""
        response_generation_start = time.time()
        
        try:
            if not self.response_generator:
                logger.warning("Response generator not available, falling back to search results")
                if not search_results:
                    search_results = await self.search(
                        query, top_k=top_k, language=language, use_reranking=use_reranking, session_id=session_id
                    )
                
                response_duration = time.time() - response_generation_start
                response = ""
                
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
            
            if search_results is None:
                search_results = await self.search(
                    query, top_k=top_k, language=language, use_reranking=use_reranking, session_id=session_id
                )
            
            if not search_results:
                logger.warning(f"No search results found for query: '{query}'")
                response = ""
                for chunk in self.response_generator.generate_response(query, [], language, session_id=session_id, temperature=temperature):
                    if chunk:
                        response += chunk
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
            
            response = ""
            for chunk in self.response_generator.generate_response(
                query, search_results, language, session_id=session_id, temperature=temperature
            ):
                if chunk:
                    response += chunk
            
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
            
            try:
                if not search_results:
                    search_results = await self.search(
                        query, top_k=top_k, language=language, use_reranking=use_reranking, session_id=session_id
                    )
                if self.response_generator:
                    return self.response_generator._format_fallback_response(query, search_results, language)
                else:
                    # If response_generator is None, return a simple error message
                    return "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำขอของคุณ" if language == "th" else "Sorry, an error occurred while processing your request."
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำขอของคุณ" if language == "th" else "Sorry, an error occurred while processing your request."
    
    async def generate_response_stream(
        self,
        query: str,
        top_k: int = 10,
        language: str = None,
        use_reranking: bool = True,
        search_results: List[Dict[str, Any]] = None,
        session_id: str = None,
        temperature: Optional[float] = None
    ):
        """Generate a streaming response generator for real-time output"""
        try:
            if not self.response_generator:
                logger.warning("Response generator not available, falling back to search results")
                if not search_results:
                    search_results = await self.search(
                        query, top_k=top_k, language=language, use_reranking=use_reranking, session_id=session_id
                    )
                fallback_response = self.response_generator._format_fallback_response(query, search_results) if self.response_generator else ""
                yield fallback_response
                return
            
            if language is None:
                language = self._detect_language(query)
            
            if search_results is None:
                search_results = await self.search(
                    query, top_k=top_k, language=language, use_reranking=use_reranking, session_id=session_id
                )
            
            if not search_results:
                logger.warning(f"No search results found for query: '{query}'")
                conversational_response = ""
                for chunk in self.response_generator.generate_response(
                    query, [], language, session_id=session_id, temperature=temperature
                ):
                    if chunk:
                        conversational_response += chunk
                yield conversational_response
                return
            
            for chunk in self.response_generator.generate_response(
                query, search_results, language, session_id=session_id
            ):
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            try:
                if not search_results:
                    search_results = await self.search(
                        query, top_k=top_k, language=language, use_reranking=use_reranking, session_id=session_id
                    )
                if self.response_generator:
                    fallback_response = self.response_generator._format_fallback_response(query, search_results, language)
                    yield fallback_response
                else:
                    # If response_generator is None, yield a simple error message
                    error_msg = "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำขอของคุณ" if language == "th" else "Sorry, an error occurred while processing your request."
                    yield error_msg
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

    def _extract_course_codes_from_query(self, query: str) -> List[str]:
        """
        Extract course codes (8-digit numbers) from query.
        Returns: List of course codes found
        """
        import re
        course_code_pattern = r'\b\d{8}\b'
        course_codes = re.findall(course_code_pattern, query)
        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in course_codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)
        return unique_codes
    
    def _boost_results_with_course_codes(self, results: List[Dict[str, Any]], course_codes: List[str]) -> List[Dict[str, Any]]:
        """
        Boost similarity scores for results that contain the mentioned course codes.
        This helps prioritize exact matches when users mention specific course codes.
        """
        if not course_codes or not results:
            return results
        
        for result in results:
            metadata = result.get('metadata', {})
            content = result.get('content', '')
            
            # Check if result contains any of the mentioned course codes
            result_course_code = metadata.get('course_code', '')
            content_lower = content.lower()
            
            for code in course_codes:
                if code == result_course_code:
                    # Strong boost for exact course code match in metadata
                    original_score = result.get('similarity_score', 0.0)
                    result['similarity_score'] = min(1.0, original_score + 0.3)
                    logger.debug(f"Boosted result with course code {code}: {original_score:.3f} -> {result['similarity_score']:.3f}")
                    break
                elif code in content_lower:
                    # Moderate boost if course code appears in content
                    original_score = result.get('similarity_score', 0.0)
                    result['similarity_score'] = min(1.0, original_score + 0.15)
                    logger.debug(f"Boosted result mentioning course code {code}: {original_score:.3f} -> {result['similarity_score']:.3f}")
                    break
        
        # Re-sort by boosted similarity score
        results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
        return results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]], similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Remove duplicate or very similar results based on content similarity and course codes.
        This prevents redundant information from appearing multiple times.
        """
        if not results or len(results) <= 1:
            return results
        
        deduplicated = []
        seen_chunks = set()
        seen_course_codes = set()
        
        for result in results:
            chunk_id = result.get('chunk_id', '')
            course_code = result.get('metadata', {}).get('course_code', '')
            content = result.get('content', '')[:100]  # Use first 100 chars for comparison
            
            # Create a unique identifier
            content_hash = hash(content)
            
            # Skip if we've seen this exact chunk
            if chunk_id in seen_chunks:
                continue
            
            # Skip if same course code and very similar content (likely duplicate chunk)
            if course_code and course_code in seen_course_codes:
                # Check if content is very similar to any existing result
                is_duplicate = False
                for existing in deduplicated:
                    existing_content = existing.get('content', '')[:100]
                    if course_code == existing.get('metadata', {}).get('course_code', ''):
                        if content == existing_content:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    continue
            
            # Add to deduplicated list
            deduplicated.append(result)
            seen_chunks.add(chunk_id)
            if course_code:
                seen_course_codes.add(course_code)
        
        if len(deduplicated) < len(results):
            logger.info(f"Deduplicated results: {len(results)} -> {len(deduplicated)}")
        
        return deduplicated
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is Thai or English using ASCII check"""
        if any(ord(c) > 127 for c in text):
            return "th"
        return "en"

    def _is_curriculum_query(self, query: str) -> bool:
        """Detect if the query is about graduation/curriculum requirements (use curriculum as primary context)."""
        q = query.lower().strip()
        keywords = [
            "graduate", "graduation", "complete the program", "finish the course", "degree requirements",
            "จบหลักสูตร", "เรียนจบ", "ต้องทำอย่างไรบ้าง", "จบการศึกษา", "หลักสูตรต้องเรียนอะไร",
            "requirements to graduate", "how to graduate", "what do i need to graduate",
            "หมวดวิชา", "โครงสร้างหลักสูตร", "หน่วยกิต", "general education", "curriculum structure", "credit requirement"
        ]
        return any(k in q for k in keywords)

    def _is_studyplan_query(self, query: str) -> bool:
        """Detect if the query is about study plan per semester (use studyplan as primary context)."""
        q = query.lower().strip()
        keywords = [
            "study plan", "studyplan", "each semester", "per semester", "every semester",
            "แผนการเรียน", "แต่ละเทอม", "เทอม", "ภาคเรียน", "semester plan",
            "ปี 1", "ปี 2", "year 1", "year 2", "first year", "second year",
        ]
        return any(k in q for k in keywords)
    
    def _find_exact_course_code_matches(self, course_codes: List[str]) -> List[Dict[str, Any]]:
        """
        Directly lookup courses by exact course code match.
        This ensures exact matches are always found, even if semantic search fails.
        
        Args:
            course_codes: List of course codes to search for
            
        Returns:
            List of result dictionaries matching the course codes
        """
        if not course_codes or not self.chunks:
            return []
        
        exact_matches = []
        seen_codes = set()
        
        for chunk in self.chunks:
            chunk_course_code = chunk.metadata.get('course_code', '')
            if chunk_course_code in course_codes and chunk_course_code not in seen_codes:
                # Found exact match - create result with high score
                result = {
                    'content': get_chunk_content(chunk),
                    'metadata': chunk.metadata,
                    'similarity_score': 1.0,  # Perfect match score
                    'chunk_id': chunk.chunk_id,
                    'original_index': chunk.original_index,
                    'data_type': chunk.metadata.get('data_type', 'course'),
                    'is_exact_match': True  # Flag to indicate this is an exact match
                }
                exact_matches.append(result)
                seen_codes.add(chunk_course_code)
                logger.info(f"Found exact match for course code: {chunk_course_code}")
        
        return exact_matches
    
    def _find_similar_course_codes(self, course_code: str, max_distance: int = 2, max_results: int = 5) -> List[str]:
        """
        Find similar course codes using Levenshtein distance (fuzzy matching).
        Useful for suggesting corrections when a course code has a typo.
        
        Args:
            course_code: The course code to find similar matches for
            max_distance: Maximum edit distance allowed (default: 2)
            max_results: Maximum number of similar codes to return
            
        Returns:
            List of similar course codes sorted by similarity
        """
        if not self.chunks or len(course_code) != 8:
            return []
        
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings"""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        # Collect all unique course codes from chunks
        all_course_codes = set()
        for chunk in self.chunks:
            code = chunk.metadata.get('course_code', '')
            if code and len(code) == 8:
                all_course_codes.add(code)
        
        # Calculate distances and filter
        similar_codes = []
        for code in all_course_codes:
            distance = levenshtein_distance(course_code, code)
            if 0 < distance <= max_distance:  # Exclude exact match (distance 0)
                similar_codes.append((code, distance))
        
        # Sort by distance and return top results
        similar_codes.sort(key=lambda x: x[1])
        return [code for code, _ in similar_codes[:max_results]]
    
    def _get_all_course_codes(self) -> List[str]:
        """
        Get all unique course codes from loaded chunks.
        Useful for suggestions and validation.
        
        Returns:
            List of all unique course codes
        """
        if not self.chunks:
            return []
        
        course_codes = set()
        for chunk in self.chunks:
            code = chunk.metadata.get('course_code', '')
            if code and len(code) == 8:
                course_codes.add(code)
        
        return sorted(list(course_codes))
