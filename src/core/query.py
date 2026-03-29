#!/usr/bin/env python3
"""
Query Enhancement Module
Enhance the query using the existing prompt.

"""

import os
import json
import re
import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple, Any

from ..utils.config import config
from ..utils.logger import get_logger
from .llm_client import LLMClient, LLMProvider

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
        self.llm_client = LLMClient(
            provider=LLMProvider.OLLAMA,
            ollama_url=ollama_url,
            model_name=model_name or config.models.ollama_model_logic
        )
        self.available = self.llm_client.is_available()

    # Schema for classification response
    CLASSIFY_SCHEMA = {
        "type": "object",
        "properties": {
            "class": {"type": "string", "enum": ["enhanced", "pass", "conversational"]},
            "is_follow_up": {"type": "boolean", "description": "True if the query is a follow-up question referencing previous conversation (e.g., 'that', 'those courses', 'explain more about it')"}
        },
        "required": ["class", "is_follow_up"]
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
                "required": ["expanded_terms"]
            }
        },
        "required": ["enhanced"]
    }
    
    # Schema for metadata generation response
    METADATA_SCHEMA = {
        "type": "object",
        "properties": {
            "metadata": {"type": "string", "minLength": 1}
        },
        "required": ["metadata"]
    }
    

    def classify_query(self, query: str, conversation_context: str = None) -> Tuple[Optional[str], bool]:
        """
        Classify the query to determine if it needs enhancement and if it's a follow-up question.
        
        Args:
            query: The user's query
            conversation_context: Optional conversation history to detect follow-up questions
        
        Returns: (classification, is_follow_up) where classification is 'enhanced', 'pass', 'conversational', or None
        """
        if not self.available:
            logger.info("Query classification not available, assuming enhanced")
            return "enhanced", False
            
        try:
            prompt_file = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "prompt", "query_classifier.md"
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            escaped_template = prompt_template.replace("{", "{{").replace("}", "}}")
            escaped_template = escaped_template.replace("{{query}}", "{query}")
            escaped_template = escaped_template.replace("{{conversation}}", "{conversation}")
            
            conversation_section = ""
            if conversation_context:
                escaped_context = conversation_context.replace("{", "{{").replace("}", "}}")
                conversation_section = f"\n\n**CONVERSATION CONTEXT:**\n{escaped_context}\n\nUse this context to detect if the query is a follow-up question referencing previous conversation (e.g., 'that', 'those courses', 'explain more about it')."
            
            prompt = escaped_template.format(query=query, conversation=conversation_section)
            
            logger.debug(f"Classification prompt length: {len(prompt)} characters")
            
            response_text = self.llm_client.generate(prompt, temperature=config.models.temperature_logic, format=self.CLASSIFY_SCHEMA, num_predict=config.models.num_predict_short)
            
            if not response_text.strip():
                logger.warning("Empty response from Ollama for classification. Assuming enhanced.")
                return "enhanced", False

            response_json = self._parse_classify_response(response_text)
            
            if not response_json:
                logger.warning("Failed to parse classification response, assuming enhanced")
                return "enhanced", False
            
            classification = response_json.get("class")
            is_follow_up = response_json.get("is_follow_up", False)
            
            if not conversation_context:
                is_follow_up = False
            
            logger.debug(f"Query classified as '{classification}', is_follow_up: {is_follow_up}")
            
            return classification, is_follow_up
                
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            logger.error(f"Query was: '{query}'")
            logger.error("Falling back to enhanced classification.")
            return "enhanced", False

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
            
            response_text = self.llm_client.generate(prompt, temperature=config.models.temperature_logic, format=self.ENHANCE_SCHEMA, num_predict=config.models.num_predict_short)
            
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

    async def generate_metadata(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Generate metadata tags for the query using async Ollama call.
        Returns: Dict with tags and query_intent or None if error
        """
        if not self.available:
            logger.info("Metadata generation not available, returning default metadata")
            return {"metadata": "general"}
            
        try:
            prompt_file = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "prompt", "metadata_generator.md"
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Escape curly braces in the template to prevent format conflicts
            escaped_template = prompt_template.replace("{", "{{").replace("}", "}}")
            # Now format with the query, but we need to unescape the {query} placeholder
            escaped_template = escaped_template.replace("{{query}}", "{query}")
            
            prompt = escaped_template.format(query=query)
            
            logger.debug(f"Metadata generation prompt length: {len(prompt)} characters")
            
            # Use async session
            async with aiohttp.ClientSession() as session:
                response_text = await self.llm_client.generate_async(session, prompt, temperature=config.models.temperature_logic, format=self.METADATA_SCHEMA, num_predict=config.models.num_predict_short)
            
            # Parse and validate JSON response
            response_json = self._parse_metadata_response(response_text)
            
            if response_json:
                metadata_intent = response_json.get("metadata", "general")
                logger.info(f"Metadata generated (intent: {metadata_intent})")
                return {"metadata": metadata_intent}
            else:
                logger.warning("Failed to parse metadata response, using default metadata")
                return {"metadata": "general"}
                
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            logger.error(f"Query was: '{query}'")
            logger.error("Using default metadata.")
            return {"tags": ["general"], "query_intent": "general"}

    def _extract_course_codes(self, conversation_context: str = None) -> List[str]:
        """
        Extract course codes (8-digit numbers) from conversation context.
        Returns: List of course codes found
        """
        if not conversation_context:
            return []
        
        # Pattern to match 8-digit course codes (e.g., 01076043, 01076634)
        course_code_pattern = r'\b\d{8}\b'
        course_codes = re.findall(course_code_pattern, conversation_context)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in course_codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)
        
        if unique_codes:
            logger.info(f"Extracted course codes from conversation: {unique_codes}")
        
        return unique_codes

    async def enhance_query_async(self, query: str, conversation_context: str = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Async version of enhance_query that runs classification, enhancement, and metadata generation in parallel.
        
        Args:
            query: The user's query
            conversation_context: Optional conversation history to provide context
        
        Returns: (enhanced_query, metadata)
        """
        language = detect_language_ascii(query)
        logger.debug(f"Language detected as '{language}' for query: '{query}'")
        
        original_query = query
        course_codes = self._extract_course_codes(conversation_context) if conversation_context else []
        
        if not self.available:
            logger.info("Query processing not available, returning original query with default metadata")
            return original_query, {"metadata": "general"}
        
        course_codes_appended = False
        
        try:
            classification, is_follow_up = self.classify_query(query, conversation_context)
            
            if is_follow_up and course_codes:
                query_with_codes = f"{query} {' '.join(course_codes)}"
                logger.info(f"Query references previous courses. Including codes: {course_codes}")
                query = query_with_codes
                course_codes_appended = True
            
            if not classification:
                logger.warning("Classification failed, returning original query with default metadata")
                return original_query, {"metadata": "general"}
            
            logger.debug(f"Query classified as '{classification}'")
            
            if classification == "enhanced":
                async with aiohttp.ClientSession() as session:
                    # Create tasks for parallel execution
                    enhancement_task = asyncio.create_task(self._enhance_query_terms_async(session, query, conversation_context))
                    metadata_task = asyncio.create_task(self.generate_metadata(query))
                    
                    # Wait for both to complete
                    enhanced_terms, metadata = await asyncio.gather(
                        enhancement_task, 
                        metadata_task,
                        return_exceptions=True
                    )
                    
                    if isinstance(enhanced_terms, Exception):
                        logger.error(f"Enhancement failed: {enhanced_terms}")
                        enhanced_terms = None
                    
                    if isinstance(metadata, Exception):
                        logger.error(f"Metadata generation failed: {metadata}")
                        metadata = {"metadata": "general"}
                    
                    if enhanced_terms and len(enhanced_terms) > 0:
                        enhanced_query = f"{original_query} {' '.join(enhanced_terms)}"
                        if course_codes_appended:
                            enhanced_query_lower = enhanced_query.lower()
                            missing_codes = [code for code in course_codes if code not in enhanced_query_lower]
                            if missing_codes:
                                enhanced_query = f"{enhanced_query} {' '.join(missing_codes)}"
                                logger.info(f"Preserved missing course codes in enhanced query: {missing_codes}")
                        logger.info(f"Query enhanced: '{original_query}' -> '{enhanced_query}' with metadata: {metadata}")
                        return enhanced_query, metadata
                    else:
                        logger.warning("Enhancement failed, keeping query")
                        if course_codes_appended:
                            logger.info(f"Returning modified query with course codes despite enhancement failure: {course_codes}")
                            return query, metadata
                        else:
                            return original_query, metadata
                        
            elif classification == "pass":
                logger.info("Query classified as pass (clear query - search without enhancement)")
                if course_codes_appended:
                    logger.info(f"Returning modified query with course codes for search: {course_codes}")
                    return query, {"metadata": "course_search"}
                else:
                    return original_query, {"metadata": "course_search"}
                
            elif classification == "conversational":
                logger.info("Query classified as conversational, keeping original")
                if course_codes_appended:
                    logger.info(f"Returning modified query with course codes for conversational query to enable search: {course_codes}")
                    return query, {"metadata": "conversational"}
                else:
                    return original_query, {"metadata": "conversational"}
                

                
            else:
                logger.warning(f"Unknown classification '{classification}', keeping original query")
                return original_query, {"metadata": "general"}
                
        except Exception as e:
            logger.error(f"Error in async query processing: {e}")
            logger.error(f"Query was: '{query}'")
            logger.error("Falling back to original query with default metadata.")
            return original_query, {"metadata": "general"}

    async def _enhance_query_terms_async(self, session: aiohttp.ClientSession, query: str, conversation_context: str = None) -> Optional[List[str]]:
        """
        Async version of enhance_query_terms using the provided session.
        
        Args:
            session: aiohttp session for async requests
            query: The user's query
            conversation_context: Optional conversation history for context-aware enhancement
        """
        try:
            prompt_file = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "prompt", "query_enhancer.md"
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Escape curly braces in the template to prevent format conflicts
            escaped_template = prompt_template.replace("{", "{{").replace("}", "}}")
            # Now format with the query and conversation context
            escaped_template = escaped_template.replace("{{query}}", "{query}")
            escaped_template = escaped_template.replace("{{conversation}}", "{conversation}")
            
            conversation_section = ""
            if conversation_context:
                escaped_context = conversation_context.replace("{", "{{").replace("}", "}}")
                conversation_section = f"\n\n**CONVERSATION CONTEXT:**\n{escaped_context}\n\nUse this context to understand references."
            
            prompt = escaped_template.format(query=query, conversation=conversation_section)
            
            logger.debug(f"Async enhancement prompt length: {len(prompt)} characters")
            
            response_text = await self.llm_client.generate_async(session, prompt, temperature=0.0, format=self.ENHANCE_SCHEMA, num_predict=config.models.num_predict_short)
            
            if not response_text.strip():
                logger.warning("Empty response from Ollama for async enhancement. Using original query.")
                return [query]

            # Parse and validate JSON response
            response_json = self._parse_enhance_response(response_text)
            
            if not response_json:
                logger.warning("Failed to parse async enhancement response, using original query")
                return [query]
            
            enhanced_data = response_json.get("enhanced", {})
            expanded_terms = enhanced_data.get("expanded_terms", [])
            
            if expanded_terms and self._validate_expanded_terms(expanded_terms):
                logger.info(f"Query enhanced async: '{query}' -> {expanded_terms}")
                return expanded_terms
            else:
                logger.warning("Invalid or empty expanded_terms in async enhancement, using original query")
                return [query]
                
        except Exception as e:
            logger.error(f"Error in async query enhancement: {e}")
            logger.error(f"Query was: '{query}'")
            logger.error("Using original query as single term.")
            return [query]

    def enhance_query(self, query: str) -> str:
        """
        Two-step query processing: classify first, then enhance if needed.
        Uses ASCII-based language detection instead of prompt-based detection.
        """
        # Detect language using ASCII analysis
        language = detect_language_ascii(query)
        logger.debug(f"Language detected as '{language}' for query: '{query}'")
        
        # Step 1: Classify the query
        classification, _ = self.classify_query(query)
        
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
            if classification not in ["enhanced", "pass", "conversational"]:
                return False
            
            if "is_follow_up" not in response_json:
                return False
            
            is_follow_up = response_json.get("is_follow_up")
            if not isinstance(is_follow_up, bool):
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

    def _parse_metadata_response(self, response_text: str) -> Optional[Dict]:
        """Parse and clean metadata JSON response from Ollama"""
        try:
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            response_json = json.loads(cleaned_response)
            
            # Validate the response structure
            if self._validate_metadata_schema(response_json):
                logger.debug(f"Successfully parsed metadata response: {response_json}")
                return response_json
            else:
                logger.warning("Metadata response doesn't match expected schema")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse metadata JSON response: {e}")
            logger.debug(f"Raw response was: {response_text}")
            return None

    def _validate_metadata_schema(self, response_json: Dict) -> bool:
        """Validate that the metadata response matches our expected schema"""
        try:    
            if "metadata" not in response_json:
                return False
                
            metadata_val = response_json.get("metadata", "")
            if not isinstance(metadata_val, str) or not metadata_val.strip():
                return False
                    
            return True
            
        except Exception as e:
            logger.warning(f"Error validating metadata schema: {e}")
            return False

    def _validate_metadata_tags(self, tags: List[str]) -> bool:
        """DEPRECATED: Tags are no longer used."""
        return True
    
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
