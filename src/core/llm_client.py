#!/usr/bin/env python3
"""
LLM Client for the RAG System
Generic client for LLM API interactions with caching, retry logic, and health checks
Supports multiple LLM providers (currently Ollama, extensible for others)
"""

import os
import time
import requests
import json
import asyncio
import aiohttp
from typing import Optional, Dict, Any, Generator, List
from abc import ABC, abstractmethod
from enum import Enum

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    # Add more providers as needed


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> Optional[str]:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response from LLM"""
        pass
    
    @abstractmethod
    async def generate_async(self, session: aiohttp.ClientSession, prompt: str, temperature: Optional[float] = None, **kwargs) -> Optional[str]:
        """Generate async response from LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the client is available"""
        pass


class OllamaClient(BaseLLMClient):
    """Ollama-specific implementation of LLM client"""
    
    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
        num_predict: Optional[int] = None,
    ):
        """
        Initialize the Ollama client

        Args:
            ollama_url: URL of the Ollama server
            model_name: Name of the model to use
            num_predict: Max tokens to predict (default from config)
        """
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat")
        self.num_predict = num_predict if num_predict is not None else config.models.num_predict
        self.available = False
        self.cache = {}  # Simple in-memory cache
        self.cache_max_size = 100
        self.retry_count = 3
        self.retry_delay = 1.0
        self.timeout = 60
        
        # Initialize connection
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama service is available and model exists"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                # Check if model is available
                available_models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in available_models]
                
                if self.model_name in model_names:
                    logger.info(f"Ollama client initialized with model: {self.model_name}")
                    self.available = True
                    return True
                else:
                    logger.error(f"Model {self.model_name} not available. Available models: {model_names}")
                    self.available = False
                    return False
            else:
                logger.warning(f"Ollama server returned status {response.status_code}")
                self.available = False
                return False
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama at {self.ollama_url}: {e}")
            logger.warning("Ollama client will be disabled. Please ensure Ollama is running.")
            self.available = False
            return False
    
    def is_available(self) -> bool:
        """Check if the client is available"""
        return self.available
    
    def _get_cache_key(self, prompt: str, temperature: float, **kwargs) -> str:
        """Generate cache key for the request"""
        # Include relevant parameters in cache key
        key_parts = [prompt, str(temperature)]
        for key, value in sorted(kwargs.items()):
            if key in ['format', 'stream', 'num_predict']:
                key_parts.append(f"{key}={value}")
        return "_".join(key_parts)
    
    def _manage_cache(self, cache_key: str, response: str):
        """Manage cache size and add new entry"""
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = response
    
    def _check_health(self) -> bool:
        """Check Ollama service health"""
        try:
            health_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return health_response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama service health check failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        format: Optional[str] = None,
        stream: bool = False,
        num_predict: int = 8192,
        use_cache: bool = True,
    ) -> Optional[str]:
        """
        Generate response from Ollama API

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            format: Response format (e.g., 'json')
            stream: Whether to stream the response
            num_predict: Maximum number of tokens to predict (default from config)
            use_cache: Whether to use caching for non-streaming requests

        Returns:
            Generated response text or None if failed
        """
        if temperature is None:
            temperature = config.models.temperature_response

        if not self.available:
            logger.warning("Ollama client not available")
            return None

        n = num_predict if num_predict is not None else self.num_predict

        # Check cache first for non-streaming requests
        if use_cache and not stream:
            cache_key = self._get_cache_key(prompt, temperature, format=format, stream=stream, num_predict=n)
            if cache_key in self.cache:
                logger.debug("Using cached Ollama response")
                return self.cache[cache_key]
        
        for attempt in range(self.retry_count):
            try:
                # Check health before making request
                if not self._check_health():
                    logger.error(f"Ollama service not responding (attempt {attempt + 1})")
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    return None
                
                # Prepare payload
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": stream,
                    "options": {
                        "temperature": temperature,
                        "num_predict": n,
                    },
                }

                if format:
                    payload["format"] = format

                # Make request
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    if stream:
                        # Handle streaming response
                        response_text = ""
                        for line in response.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    if 'response' in data:
                                        response_text += data['response']
                                except json.JSONDecodeError:
                                    continue
                        return response_text if response_text else None
                    else:
                        # Handle non-streaming response
                        response_data = response.json()
                        response_text = response_data.get('response', '').strip()
                        
                        # Check for short/generic responses
                        if len(response_text) < 10:
                            logger.warning(f"Ollama response too short ({len(response_text)} chars): {response_text[:100]}")
                            if attempt < self.retry_count - 1:
                                time.sleep(self.retry_delay * (attempt + 1))
                                continue
                            return None
                        
                        # Cache the result
                        if use_cache:
                            cache_key = self._get_cache_key(prompt, temperature, format=format, stream=stream, num_predict=n)
                            self._manage_cache(cache_key, response_text)
                        
                        return response_text
                else:
                    logger.warning(f"Ollama API error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    return None
                    
            except Exception as e:
                logger.warning(f"Error calling Ollama (attempt {attempt + 1}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return None
        
        return None
    
    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: int = 8192,
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from Ollama API

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            num_predict: Maximum number of tokens to predict (default from config)

        Yields:
            Response chunks as they arrive
        """
        if temperature is None:
            temperature = config.models.temperature_response

        if not self.available:
            logger.warning("Ollama client not available for streaming")
            return

        n = num_predict if num_predict is not None else self.num_predict

        try:
            # Check health before making request
            if not self._check_health():
                logger.error("Ollama service not responding for streaming")
                return

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": n,
                },
            }
            
            # Make streaming request
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Stream response chunks
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                chunk = data['response']
                                if chunk:
                                    yield chunk
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error processing streaming chunk: {e}")
                            continue
            else:
                logger.error(f"Ollama API error for streaming: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calling Ollama stream: {e}")
    
    async def generate_async(
        self, 
        session: aiohttp.ClientSession,
        prompt: str, 
        temperature: Optional[float] = None, 
        format: Optional[str] = None,
        num_predict: Optional[int] = 8192,
        use_cache: bool = True
    ) -> Optional[str]:
        """
        Async version of generate method
        
        Args:
            session: aiohttp ClientSession
            prompt: The input prompt
            temperature: Sampling temperature
            format: Response format (e.g., 'json')
            num_predict: Maximum number of tokens to predict
            use_cache: Whether to use caching
            
        Returns:
            Generated response text or None if failed
        """
        if temperature is None:
            temperature = config.models.temperature_response

        if not self.available:
            logger.warning("Ollama client not available for async request")
            return None
        
        n = num_predict if num_predict is not None else self.num_predict

        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(prompt, temperature, format=format, num_predict=n)
            if cache_key in self.cache:
                logger.debug("Using cached async Ollama response")
                return self.cache[cache_key]
        
        for attempt in range(self.retry_count):
            try:
                # Prepare payload
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": n,
                    },
                }

                if format:
                    payload["format"] = format

                # Make async request
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get("response", "").strip()
                        
                        # Cache the result
                        if use_cache:
                            cache_key = self._get_cache_key(prompt, temperature, format=format, num_predict=n)
                            self._manage_cache(cache_key, response_text)

                        return response_text
                    else:
                        logger.warning(f"Ollama API error (attempt {attempt + 1}): {response.status}")
                        if attempt < self.retry_count - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                        return None
                        
            except Exception as e:
                logger.warning(f"Error calling Ollama async (attempt {attempt + 1}): {e}")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                return None
        
        return None
    
    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        logger.info("Ollama client cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            "cache_size": len(self.cache),
            "cache_max_size": self.cache_max_size,
            "cache_usage_percent": (len(self.cache) / self.cache_max_size) * 100
        }
    
    def update_model(self, model_name: str) -> bool:
        """
        Update the model name and check availability
        
        Args:
            model_name: New model name
            
        Returns:
            True if model is available, False otherwise
        """
        self.model_name = model_name
        return self._check_availability()
    
    def update_url(self, ollama_url: str) -> bool:
        """
        Update the Ollama URL and check availability
        
        Args:
            ollama_url: New Ollama URL
            
        Returns:
            True if service is available, False otherwise
        """
        self.ollama_url = ollama_url
        return self._check_availability()


class LLMClient:
    """Generic LLM client that can work with different providers"""
    
    def __init__(self, provider: LLMProvider = LLMProvider.OLLAMA, **kwargs):
        """
        Initialize the LLM client with specified provider
        
        Args:
            provider: LLM provider to use
            **kwargs: Provider-specific configuration
        """
        self.provider = provider
        self.client = self._create_client(provider, **kwargs)
    
    def _create_client(self, provider: LLMProvider, **kwargs) -> BaseLLMClient:
        """Create the appropriate client based on provider"""
        if provider == LLMProvider.OLLAMA:
            return OllamaClient(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate response using the configured provider"""
        return self.client.generate(prompt, **kwargs)
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using the configured provider"""
        return self.client.generate_stream(prompt, **kwargs)
    
    async def generate_async(self, session: aiohttp.ClientSession, prompt: str, **kwargs) -> Optional[str]:
        """Generate async response using the configured provider"""
        return await self.client.generate_async(session, prompt, **kwargs)
    
    def is_available(self) -> bool:
        """Check if the client is available"""
        return self.client.is_available()
    
    def switch_provider(self, provider: LLMProvider, **kwargs) -> bool:
        """
        Switch to a different LLM provider
        
        Args:
            provider: New LLM provider
            **kwargs: Provider-specific configuration
            
        Returns:
            True if switch was successful, False otherwise
        """
        try:
            self.provider = provider
            self.client = self._create_client(provider, **kwargs)
            return self.client.is_available()
        except Exception as e:
            logger.error(f"Failed to switch to provider {provider}: {e}")
            return False
    
    def get_provider(self) -> LLMProvider:
        """Get the current provider"""
        return self.provider
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information if supported by the client"""
        if hasattr(self.client, 'get_cache_info'):
            return self.client.get_cache_info()
        return {"cache_info": "Not supported by current provider"}
    
    def clear_cache(self):
        """Clear cache if supported by the client"""
        if hasattr(self.client, 'clear_cache'):
            self.client.clear_cache()
