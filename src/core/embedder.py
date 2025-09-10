import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging
from ..utils.config import config
from ..utils.error_handler import handle_errors, ErrorType
from ..utils.performance_monitor import monitor_operation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = None, cache_dir: str = None):
        self.model_name = model_name or config.models.embedding_model
        self.cache_dir = cache_dir or config.cache.embeddings_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load model with error handling
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Set model parameters for optimal performance
            self.model.max_seq_length = 512  # Default max length
            logger.info(f"Model loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to CPU if CUDA fails
            if self.device == "cuda":
                logger.info("Falling back to CPU")
                self.device = "cpu"
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.model.max_seq_length = 512
                logger.info("Model loaded on CPU")
            else:
                raise e
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on character sets"""
        thai_chars = sum(1 for char in text if '\u0e00' <= char <= '\u0e7f')
        english_chars = sum(1 for char in text if char.isascii())
        
        if thai_chars > english_chars:
            return "th"
        return "en"
    
    def prepare_text_for_embedding(self, text: str, language: str) -> str:
        """Prepare text for embedding with language-specific prefixes"""
        if language == "th":
            return f"Represent this Thai text for retrieval: {text}"
        else:
            return f"Represent this English text for retrieval: {text}"
    
    @monitor_operation("embedding_generation")
    @handle_errors(ErrorType.EMBEDDING_GENERATION, fallback_value=np.array([]))
    def get_embeddings(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Get embeddings for a list of texts with optimized batch processing and caching"""

        if len(texts) == 0:
            logger.warning("Empty texts list provided")
            return np.array([])
        
        # Validate batch size
        if batch_size is not None and not (1 <= batch_size <= 128):
            logger.warning(f"Invalid batch size {batch_size}, using default")
            batch_size = None
        
        # Use default batch size if not specified
        batch_size = batch_size or config.processing.batch_size
        
        # Detect languages for each text
        languages = [self.detect_language(text) for text in texts]
        
        # Prepare texts with language-specific prefixes
        prepared_texts = [
            self.prepare_text_for_embedding(text, lang) 
            for text, lang in zip(texts, languages)
        ]
        
        # Process in batches for memory efficiency with optimized settings
        embeddings = []
        for i in tqdm(range(0, len(prepared_texts), batch_size), desc="Generating embeddings"):
            batch = prepared_texts[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(
                    batch, 
                    convert_to_numpy=True, 
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=min(batch_size, 32),  # Limit internal batch size
                    device=self.device
                )
                embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Continue with other batches
                continue
        
        if not embeddings:
            logger.error("No embeddings generated successfully")
            return np.array([])
        
        return np.vstack(embeddings)
    
    def get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        # Input validation
        if not isinstance(text, str):
            logger.error("Text must be a string")
            return np.array([])
        
        if not text.strip():
            logger.warning("Empty text provided")
            return np.array([])
        
        language = self.detect_language(text)
        prepared_text = self.prepare_text_for_embedding(text, language)
        
        try:
            embedding = self.model.encode(
                [prepared_text], 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
            return embedding[0]
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return np.array([])
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str):
        """Save embeddings to cache"""
        filepath = os.path.join(self.cache_dir, filename)
        np.save(filepath, embeddings)
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filename: str) -> Optional[np.ndarray]:
        """Load embeddings from cache"""
        filepath = os.path.join(self.cache_dir, filename)
        if os.path.exists(filepath):
            embeddings = np.load(filepath)
            logger.info(f"Embeddings loaded from {filepath}")
            return embeddings
        return None
