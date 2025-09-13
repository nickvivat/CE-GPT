#!/usr/bin/env python3
"""
Query Enhancement Module
Enhance the query using the existing prompt.

"""

import os
import time
import requests
import json
import re
from typing import List, Dict, Optional

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

def detect_language_ascii(text: str) -> str:
    """
    Detect language based on ASCII character analysis.
    Returns 'th' for Thai text, 'en' for English/other languages.
    """
    if not text or not text.strip():
        return 'en'
    
    # Remove whitespace and punctuation for analysis
    clean_text = re.sub(r'[^\w\s]', '', text.strip())
    
    if not clean_text:
        return 'en'
    
    # Count Thai characters (Unicode range: U+0E00-U+0E7F)
    thai_chars = len(re.findall(r'[\u0E00-\u0E7F]', clean_text))
    total_chars = len(clean_text)
    
    # If more than 30% of characters are Thai, classify as Thai
    if total_chars > 0 and (thai_chars / total_chars) > 0.3:
        return 'th'
    
    return 'en'

class Query:
    def __init__(self, ollama_url: str = None, model_name: str = None):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat")
        self.available = False
        self.cache = {}  # Simple in-memory cache for query enhancement
        self.cache_max_size = 100
        self.retry_count = 3
        self.retry_delay = 1.0
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Query enhancer initialized with Ollama model: {self.model_name}")
                self.available = True
            else:
                logger.warning(f"Ollama server returned status {response.status_code}")
                self.available = False
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama at {self.ollama_url}: {e}")
            logger.warning("Query enhancement will be disabled. Please ensure Ollama is running.")
            self.available = False

    # Schema for classification response
    CLASSIFY_SCHEMA = {
        "type": "object",
        "properties": {
            "class": {"type": "string", "enum": ["enhanced", "pass", "external"]}
        },
        "required": ["class"],
        "additionalProperties": False
    }
    
    # Schema for enhancement response
    ENHANCE_SCHEMA = {
        "type": "object",
        "properties": {
            "enhanced": {
                "type": "object",
                "properties": {
                    "expanded_terms": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1
                    }
                },
                "required": ["expanded_terms"],
                "additionalProperties": False
            }
        },
        "required": ["enhanced"],
        "additionalProperties": False
    }
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """Make a call to Ollama API with retry logic and caching"""
        # Check cache first
        cache_key = f"{prompt}_{temperature}"
        if cache_key in self.cache:
            logger.debug("Using cached query enhancement result")
            return self.cache[cache_key]
        
        for attempt in range(self.retry_count):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                }
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=60  # Increased timeout for complex queries
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "").strip()
                    
                    # Cache the result
                    if len(self.cache) >= self.cache_max_size:
                        # Remove oldest entry
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                    
                    self.cache[cache_key] = response_text
                    return response_text
                else:
                    logger.warning(f"Ollama API error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        return ""
                    
            except Exception as e:
                logger.warning(f"Error calling Ollama (attempt {attempt + 1}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return ""
        
        return ""

    def classify_query(self, query: str) -> Optional[str]:
        """
        Classify the query to determine if it needs enhancement.
        Returns: 'enhanced', 'pass', 'external', or None if error
        """
        if not self.available:
            logger.info("Query classification not available, assuming enhanced")
            return "enhanced"
            
        try:
            prompt_file = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "prompt", "query_classifier.md"
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Escape curly braces in the template to prevent format conflicts
            escaped_template = prompt_template.replace("{", "{{").replace("}", "}}")
            # Now format with the query, but we need to unescape the {query} placeholder
            escaped_template = escaped_template.replace("{{query}}", "{query}")
            
            prompt = escaped_template.format(query=query)
            
            logger.debug(f"Classification prompt length: {len(prompt)} characters")
            
            response_text = self._call_ollama(prompt, temperature=0.0)
            
            if not response_text.strip():
                logger.warning("Empty response from Ollama for classification. Assuming enhanced.")
                return "enhanced"

            # Parse and validate JSON response
            response_json = self._parse_classify_response(response_text)
            
            if not response_json:
                logger.warning("Failed to parse classification response, assuming enhanced")
                return "enhanced"
            
            classification = response_json.get("class")
            logger.debug(f"Query classified as '{classification}'")
            
            return classification
                
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            logger.error(f"Query was: '{query}'")
            logger.error("Falling back to enhanced classification.")
            return "enhanced"

    def enhance_query_terms(self, query: str) -> Optional[List[str]]:
        """
        Enhance the query terms using the enhancement prompt.
        Returns: List of expanded terms or None if error
        """
        if not self.available:
            logger.info("Query enhancement not available, returning original query as single term")
            return [query]
            
        try:
            prompt_file = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "prompt", "query_enhancer.md"
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Escape curly braces in the template to prevent format conflicts
            escaped_template = prompt_template.replace("{", "{{").replace("}", "}}")
            # Now format with the query, but we need to unescape the {query} placeholder
            escaped_template = escaped_template.replace("{{query}}", "{query}")
            
            prompt = escaped_template.format(query=query)
            
            logger.debug(f"Enhancement prompt length: {len(prompt)} characters")
            
            response_text = self._call_ollama(prompt, temperature=0.0)
            
            if not response_text.strip():
                logger.warning("Empty response from Ollama for enhancement. Using original query.")
                return [query]

            # Parse and validate JSON response
            response_json = self._parse_enhance_response(response_text)
            
            if not response_json:
                logger.warning("Failed to parse enhancement response, using original query")
                return [query]
            
            enhanced_data = response_json.get("enhanced", {})
            expanded_terms = enhanced_data.get("expanded_terms", [])
            
            if expanded_terms and self._validate_expanded_terms(expanded_terms):
                logger.info(f"Query enhanced: '{query}' -> {expanded_terms}")
                return expanded_terms
            else:
                logger.warning("Invalid or empty expanded_terms, using original query")
                return [query]
                
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            logger.error(f"Query was: '{query}'")
            logger.error("Using original query as single term.")
            return [query]

    def enhance_query(self, query: str, conversation_context: str = "") -> str:
        """
        Two-step query processing: classify first, then enhance if needed.
        Uses ASCII-based language detection instead of prompt-based detection.
        """
        # Detect language using ASCII analysis
        language = detect_language_ascii(query)
        logger.debug(f"Language detected as '{language}' for query: '{query}'")
        
        # Step 1: Classify the query
        classification = self.classify_query(query)
        
        if not classification:
            logger.warning("Classification failed, returning original query")
            return query
        
        logger.debug(f"Query classified as '{classification}'")
        
        # Step 2: Handle based on classification
        if classification == "enhanced":
            # Enhance the query terms
            expanded_terms = self.enhance_query_terms(query)
            if expanded_terms and len(expanded_terms) > 0:
                enhanced_query = " ".join(expanded_terms)
                logger.info(f"Query enhanced: '{query}' -> '{enhanced_query}'")
                return enhanced_query
            else:
                logger.warning("Enhancement failed, keeping original query")
                return query
                
        elif classification == "pass":
            logger.info("Query classified as conversational, keeping original")
            return query
            
        elif classification == "external":
            logger.info("Query classified as external, keeping original")
            return query
            
        else:
            logger.warning(f"Unknown classification '{classification}', keeping original query")
            return query
    
    def _parse_classify_response(self, response_text: str) -> Optional[Dict]:
        """Parse and clean classification JSON response from Ollama"""
        try:
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            response_json = json.loads(cleaned_response)
            
            # Validate the response structure
            if self._validate_classify_schema(response_json):
                logger.debug(f"Successfully parsed classification response: {response_json}")
                return response_json
            else:
                logger.warning("Classification response doesn't match expected schema")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse classification JSON response: {e}")
            logger.debug(f"Raw response was: {response_text}")
            return None

    def _parse_enhance_response(self, response_text: str) -> Optional[Dict]:
        """Parse and clean enhancement JSON response from Ollama"""
        try:
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            response_json = json.loads(cleaned_response)
            
            # Validate the response structure
            if self._validate_enhance_schema(response_json):
                logger.debug(f"Successfully parsed enhancement response: {response_json}")
                return response_json
            else:
                logger.warning("Enhancement response doesn't match expected schema")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse enhancement JSON response: {e}")
            logger.debug(f"Raw response was: {response_text}")
            return None
    
    def _validate_classify_schema(self, response_json: Dict) -> bool:
        """Validate that the classification response matches our expected schema"""
        try:    
            if "class" not in response_json:
                return False
                
            classification = response_json.get("class")
            if classification not in ["enhanced", "pass", "external"]:
                return False
                    
            return True
            
        except Exception as e:
            logger.warning(f"Error validating classification schema: {e}")
            return False

    def _validate_enhance_schema(self, response_json: Dict) -> bool:
        """Validate that the enhancement response matches our expected schema"""
        try:    
            if "enhanced" not in response_json:
                return False
                
            enhanced_data = response_json["enhanced"]
            if not isinstance(enhanced_data, dict) or "expanded_terms" not in enhanced_data:
                return False
                
            expanded_terms = enhanced_data.get("expanded_terms", [])
            if not isinstance(expanded_terms, list) or len(expanded_terms) == 0:
                return False
                    
            return True
            
        except Exception as e:
            logger.warning(f"Error validating enhancement schema: {e}")
            return False
    
    def _validate_expanded_terms(self, expanded_terms: List[str]) -> bool:
        """Validate that expanded terms are valid"""
        try:    
            if len(expanded_terms) == 0:
                return False
                
            # Check that all terms are non-empty strings
            for term in expanded_terms:
                if not isinstance(term, str) or not term.strip():
                    return False
                    
            return True
            
        except Exception as e:
            logger.warning(f"Error validating expanded terms: {e}")
            return False
