#!/usr/bin/env python3
"""
Query Enhancement Module
Enhance the query using the existing prompt.

"""

import os
import time
import requests
import json
from typing import List, Dict, Optional

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

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

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "language": {"type": "string", "enum": ["th", "en"]},
            "class": {"type": "string", "enum": ["enhanced", "pass", "external"]},
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
            },
            "pass": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string", "minLength": 1}
                },
                "required": ["explanation"],
                "additionalProperties": False
            },
            "external": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string", "minLength": 1}
                },
                "required": ["explanation"],
                "additionalProperties": False
            }
        },
        "required": ["language", "class"],
        "additionalProperties": False,
        "allOf": [
            {"if": {"properties": {"class": {"const": "enhanced"}}},
             "then": {"required": ["enhanced"]}},
            {"if": {"properties": {"class": {"const": "pass"}}},
             "then": {"required": ["pass"]}},
            {"if": {"properties": {"class": {"const": "external"}}},
             "then": {"required": ["external"]}}
        ]
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

    def enhance_query(self, query: str, conversation_context: str = "") -> str:
        """
        Simple query enhancement using the existing prompt.
        Let Gemma classify if enhancement is needed and enhance the query if so.
        """
        # Check if Ollama is available
        if not self.available:
            logger.info("Query enhancement not available, returning original query")
            return query
            
        try:
            prompt_file = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "prompt", "query_processor.md"
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Escape curly braces in the template to prevent format conflicts
            escaped_template = prompt_template.replace("{", "{{").replace("}", "}}")
            # Now format with the query, but we need to unescape the {query} placeholder
            escaped_template = escaped_template.replace("{{query}}", "{query}")
            
            prompt = escaped_template.format(
                query=query
            )
            
            # Debug: Log the formatted prompt to see if there are any issues
            logger.debug(f"Formatted prompt length: {len(prompt)} characters")
            logger.debug(f"Prompt ends with: {prompt[-100:]}")
            
            response_text = self._call_ollama(prompt, temperature=0.0)
            
            if not response_text.strip():
                logger.warning("Empty response from Ollama. Falling back to pass through.")
                return query

            # Parse and validate JSON response
            response_json = self._parse_json_response(response_text)
            
            if not response_json:
                logger.warning("Failed to parse JSON response, keeping original query")
                return query
            
            # Handle different classification types
            classification = response_json.get("class")
            language = response_json.get("language", "en")
            
            logger.debug(f"Query classified as '{classification}' in language '{language}'")
            
            if classification == "enhanced" and "enhanced" in response_json:
                expanded_terms = response_json["enhanced"].get("expanded_terms", [])
                if expanded_terms and self._validate_expanded_terms(expanded_terms):
                    enhanced_query = " ".join(expanded_terms)
                    logger.info(f"Query enhanced: '{query}' -> '{enhanced_query}'")
                    return enhanced_query
                else:
                    logger.warning("Invalid or empty expanded_terms, keeping original query")
                    
            elif classification == "pass":
                logger.info("Query classified as conversational, keeping original")
                
            elif classification == "external":
                logger.info("Query classified as external, keeping original")
                
            else:
                logger.warning(f"Unknown classification '{classification}', keeping original query")
            
            return query
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Query was: '{query}'")
            logger.error("Falling back to pass through.")
            return query
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Parse and clean JSON response from Ollama"""
        try:
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            response_json = json.loads(cleaned_response)
            
            # Validate the response structure
            if self._validate_response_schema(response_json):
                logger.debug(f"Successfully parsed JSON response: {response_json}")
                return response_json
            else:
                logger.warning("Response doesn't match expected schema")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response was: {response_text}")
            return None
    
    def _validate_response_schema(self, response_json: Dict) -> bool:
        """Validate that the response matches our expected schema"""
        try:    
            if "class" not in response_json or "language" not in response_json:
                return False
                
            classification = response_json.get("class")
            if classification not in ["enhanced", "pass", "external"]:
                return False
                
            # Check classification-specific fields
            if classification == "enhanced":
                if "enhanced" not in response_json:
                    return False
                enhanced_data = response_json["enhanced"]
                if not isinstance(enhanced_data, dict) or "expanded_terms" not in enhanced_data:
                    return False
                    
            elif classification == "pass":
                if "pass" not in response_json:
                    return False
                    
            elif classification == "external":
                if "external" not in response_json:
                    return False
                    
            return True
            
        except Exception as e:
            logger.warning(f"Error validating response schema: {e}")
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
