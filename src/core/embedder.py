import os
import numpy as np
import requests
from typing import List, Optional
from tqdm import tqdm
import logging
from ..utils.config import config
from ..utils.error_handler import handle_errors, ErrorType
from ..utils.performance_monitor import monitor_operation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """Embedder using Ollama /api/embed (no in-process model load)."""

    def __init__(
        self,
        ollama_url: str = None,
        model_name: str = None,
        cache_dir: str = None,
        timeout: int = 120,
    ):
        self.ollama_url = (ollama_url or config.models.ollama_url).rstrip("/")
        self.model_name = model_name or config.models.embedding_model
        self.cache_dir = cache_dir or config.cache.embeddings_dir
        self.timeout = timeout

        os.makedirs(self.cache_dir, exist_ok=True)

        if not self._check_availability():
            raise RuntimeError(
                f"Ollama embedding service not available at {self.ollama_url} "
                f"or model '{self.model_name}' not found. "
                "Ensure Ollama is running and the embedding model is pulled (e.g. ollama pull embeddinggemma:latest)."
            )
        logger.info(
            f"Embedder using Ollama at {self.ollama_url} with model: {self.model_name}"
        )

    def _check_availability(self) -> bool:
        """Check if Ollama is reachable and the embedding model exists."""
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            models = r.json().get("models", [])
            names = [m.get("name", "") for m in models]
            # Ollama tags can be "model:tag", so match by prefix
            if self.model_name in names:
                return True
            for n in names:
                if n.startswith(self.model_name + ":") or n == self.model_name:
                    return True
            logger.warning(
                f"Ollama embedding model '{self.model_name}' not in list: {names}"
            )
            return False
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Call Ollama /api/embed for a list of texts. Returns (n, dim) array."""
        url = f"{self.ollama_url}/api/embed"
        payload = {"model": self.model_name, "input": texts}
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise ValueError("Ollama response missing 'embeddings'")
        out = np.array(embeddings, dtype=np.float32)
        # Ollama returns L2-normalized vectors; ensure unit norm for consistency
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return (out / norms).astype(np.float32)

    def detect_language(self, text: str) -> str:
        """Simple language detection based on character sets"""
        thai_chars = sum(1 for char in text if "\u0e00" <= char <= "\u0e7f")
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
        """Get embeddings for a list of texts via Ollama /api/embed."""

        if len(texts) == 0:
            logger.warning("Empty texts list provided")
            return np.array([])

        if batch_size is not None and not (1 <= batch_size <= 128):
            logger.warning(f"Invalid batch size {batch_size}, using default")
            batch_size = None

        batch_size = batch_size or config.processing.batch_size

        languages = [self.detect_language(text) for text in texts]
        prepared_texts = [
            self.prepare_text_for_embedding(text, lang)
            for text, lang in zip(texts, languages)
        ]

        embeddings = []
        for i in tqdm(
            range(0, len(prepared_texts), batch_size), desc="Generating embeddings"
        ):
            batch = prepared_texts[i : i + batch_size]
            try:
                batch_embeddings = self._embed_batch(batch)
                embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                continue

        if not embeddings:
            logger.error("No embeddings generated successfully")
            return np.array([])

        return np.vstack(embeddings)

    def get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        if not isinstance(text, str):
            logger.error("Text must be a string")
            return np.array([])

        if not text.strip():
            logger.warning("Empty text provided")
            return np.array([])

        language = self.detect_language(text)
        prepared_text = self.prepare_text_for_embedding(text, language)

        try:
            emb = self._embed_batch([prepared_text])
            return emb[0]
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
