#!/usr/bin/env python3
"""
Response Generator for the RAG System
Generates contextual responses using retrieved data and system prompts
"""

import os
import re
import time
import requests
import json
from typing import List, Dict, Any, Optional

from ..utils.config import config
from ..utils.logger import get_logger
from .llm_client import LLMClient, LLMProvider

logger = get_logger(__name__)


class ResponseGenerator:
    """Generates contextual responses using retrieved data and LLM"""
    
    def __init__(self, model_name: str = None, chat_history_manager=None):
        """Initialize the response generator"""
        self.llm_client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_name=model_name or config.models.ollama_model
        )
        self.chat_history_manager = chat_history_manager
    
    def _detect_language(self, text: str) -> str:
        """Detect if text contains any Thai character; if so, respond in Thai, else English"""
        for char in text:
            if 0x0E00 <= ord(char) <= 0x0E7F:
                return "th"
        return "en"
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content for better readability"""
        if not content:
            return ""
        
        # Replace newlines and multiple spaces with single space
        cleaned = content.replace('\n', ' ').replace('\r', ' ')
        cleaned = ' '.join(cleaned.split())
        
        # Remove excessive punctuation and normalize
        cleaned = cleaned.strip()
        
        return cleaned

    def _clean_html_text(self, content: str) -> str:
        """Strip HTML from curriculum/studyplan text and convert to plain text with line breaks."""
        if not content:
            return ""
        text = content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        text = text.replace("</tr>", "\n").replace("</td>", " ").replace("</p>", "\n")
        text = re.sub(r"<[^>]+>", "", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _filter_by_rerank_score(self, results: List[Dict[str, Any]], 
                               threshold: float = None) -> List[Dict[str, Any]]:
        """Filter results based on rerank score or hybrid score threshold"""
        if not results:
            return results
        
        # Determine which threshold to use based on available scores
        if threshold is None:
            # Check first result to determine score type
            first_result = results[0]
            if first_result.get('hybrid_score') is not None or first_result.get('rerank_score') is not None:
                threshold = config.search.rerank_threshold
            else:
                # Reranking disabled - use similarity_threshold for similarity_score
                threshold = config.search.similarity_threshold
        
        filtered_results = []
        for result in results:
            # Always keep metadata results (suggestions) and exact matches regardless of score
            if result.get('data_type') == 'metadata' or result.get('bypass_score_filter', False) or result.get('is_exact_match', False):
                filtered_results.append(result)
                continue
            
            score = result.get('hybrid_score')
            if score is None:
                score = result.get('rerank_score')
                if score is None:
                    score = result.get('similarity_score', 0.0)
            
            if score >= threshold:
                filtered_results.append(result)
        
        original_count = len(results)
        filtered_count = len(filtered_results)
        logger.info(f"Score filtering: {original_count} -> {filtered_count} results "
                   f"(threshold: {threshold:.3f})")
        
        # If no results pass the threshold, return the top 5 results anyway
        if not filtered_results and results:
            logger.warning("No results passed score threshold, returning top 5 results")
            
            # Separate metadata results and exact matches - they must always be preserved
            preserved_results = [r for r in results if r.get('data_type') == 'metadata' or r.get('bypass_score_filter', False) or r.get('is_exact_match', False)]
            non_metadata_results = [r for r in results if r.get('data_type') != 'metadata' and not r.get('bypass_score_filter', False) and not r.get('is_exact_match', False)]
            
            # Sort non-metadata results by score and take top 5
            sorted_results = sorted(
                non_metadata_results, 
                key=lambda x: (
                    x.get('hybrid_score') if x.get('hybrid_score') is not None 
                    else (x.get('rerank_score') if x.get('rerank_score') is not None 
                          else x.get('similarity_score', 0.0))
                ), 
                reverse=True
            )
            
            # Combine: top 5 non-metadata results + all preserved results (exact matches and metadata)
            return sorted_results[:5] + preserved_results
        
        return filtered_results
    
    def _extract_course_codes_from_text(self, text: str) -> List[str]:
        """Extract course codes (8-digit numbers) from text"""
        import re
        course_code_pattern = r'\b\d{8}\b'
        course_codes = re.findall(course_code_pattern, text)
        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in course_codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)
        return unique_codes
    
    def _format_history(self, session_id: str, max_messages: int = 10, current_query: str = None) -> str:
        """
        Format chat history with smart selection optimized for large context windows (128k tokens).
        When compression is enabled and message count exceeds trigger, older messages are summarized
        and only the last N messages are kept in full.
        
        Strategy:
        1. If compression enabled and count > trigger: summarize old, keep recent full
        2. Otherwise: Last 2-3 assistant/user responses, messages with course codes, up to 10 messages
        """
        if not session_id or not self.chat_history_manager:
            return "No previous conversation."
        
        try:
            # Compression path: when enabled and history exceeds trigger, summarize old and keep recent full
            sess = config.session
            if sess.chat_history_compression_enabled:
                count = self.chat_history_manager.get_message_count(session_id)
                if count > sess.compression_trigger_after_messages:
                    n_consider = min(count, sess.compression_max_messages_to_consider)
                    messages = self.chat_history_manager.get_messages_for_compression(session_id, n_consider)
                    if messages:
                        # Remove current query if it's the last message
                        if current_query and messages[-1].role == "user" and messages[-1].content.strip() == current_query.strip():
                            messages = messages[:-1]
                        if messages:
                            compressed = self.chat_history_manager.get_or_compute_compressed_history(
                                session_id,
                                messages,
                                sess.compression_recent_messages_full,
                                sess.compression_summary_max_tokens,
                                self.llm_client,
                                message_count_for_cache=count,
                            )
                            history_parts = []
                            if compressed.summary:
                                history_parts.append(f"**Summary of earlier conversation:** {compressed.summary}")
                                history_parts.append("")
                            all_course_codes = set()
                            for msg in compressed.recent_messages:
                                all_course_codes.update(self._extract_course_codes_from_text(msg.content))
                            if compressed.summary:
                                all_course_codes.update(self._extract_course_codes_from_text(compressed.summary))
                            if all_course_codes:
                                history_parts.append(f"**Previously discussed courses:** {', '.join(sorted(all_course_codes))}")
                                history_parts.append("")
                            for i, msg in enumerate(compressed.recent_messages, 1):
                                role_label = "User" if msg.role == "user" else "Assistant"
                                history_parts.append(f"{i}. **{role_label}:** {msg.content}")
                            if history_parts:
                                formatted = "\n".join(history_parts)
                                formatted += "\n\n**Note:** Use the conversation history above to understand context, especially when the user refers to 'those courses', 'the courses you mentioned', or asks follow-up questions. Pay attention to course codes mentioned in the history."
                                return formatted
                            return "No previous conversation."
            
            # Default path: smart selection (no compression or below trigger)
            all_recent_messages = self.chat_history_manager.get_recent_messages(session_id, n=max_messages * 2)
            if not all_recent_messages:
                return "No previous conversation."
            
            # Remove current query if it's the last message
            messages_to_consider = all_recent_messages
            if current_query and all_recent_messages:
                last_msg = all_recent_messages[-1]
                if last_msg.role == "user" and last_msg.content.strip() == current_query.strip():
                    messages_to_consider = all_recent_messages[:-1]
            
            if not messages_to_consider:
                return "No previous conversation."
            
            # Extract ALL course codes from entire conversation (for summary)
            all_course_codes = set()
            for msg in messages_to_consider:
                codes = self._extract_course_codes_from_text(msg.content)
                all_course_codes.update(codes)
            
            # Smart message selection with larger context window
            selected_messages = []
            seen_indices = set()
            
            # Create index mapping for O(1) lookups (avoid O(n) index() calls)
            message_to_index = {id(m): i for i, m in enumerate(messages_to_consider)}
            
            # Strategy 1: Always include the last 2-3 assistant responses (most recent context)
            assistant_count = 0
            for msg in reversed(messages_to_consider):
                if msg.role == "assistant" and assistant_count < 3:
                    idx = message_to_index.get(id(msg), -1)
                    if idx >= 0 and idx not in seen_indices:
                        selected_messages.insert(0, msg)
                        seen_indices.add(idx)
                        assistant_count += 1
            
            # Strategy 2: Include the last 2-3 user queries (conversation flow)
            user_count = 0
            for msg in reversed(messages_to_consider):
                if msg.role == "user" and user_count < 3:
                    idx = message_to_index.get(id(msg), -1)
                    if idx >= 0 and idx not in seen_indices:
                        selected_messages.insert(0, msg)
                        seen_indices.add(idx)
                        user_count += 1
            
            # Strategy 3: Include messages containing course codes (even if older)
            # This ensures we capture context about specific courses mentioned
            if all_course_codes:
                for msg in reversed(messages_to_consider):
                    if len(selected_messages) >= 10:  # Limit to 10 messages max
                        break
                    idx = message_to_index.get(id(msg), -1)
                    if idx < 0 or idx in seen_indices:
                        continue
                    msg_codes = set(self._extract_course_codes_from_text(msg.content))
                    if msg_codes & all_course_codes:  # Has overlapping course codes
                        selected_messages.insert(0, msg)
                        seen_indices.add(idx)
            
            # Sort by original order and limit to reasonable number
            # Create index mapping for O(1) lookup instead of O(n) index() calls
            message_to_index = {id(m): i for i, m in enumerate(messages_to_consider)}
            selected_messages = sorted(
                selected_messages, 
                key=lambda m: message_to_index.get(id(m), len(messages_to_consider))
            )
            selected_messages = selected_messages[-10:]  # Max 10 messages
            
            if not selected_messages:
                # Fallback: at least include the last message
                selected_messages = [messages_to_consider[-1]]
            
            history_parts = []
            
            # Add course code summary (most important for context)
            if all_course_codes:
                codes_list = sorted(list(all_course_codes))
                history_parts.append(f"**Previously discussed courses:** {', '.join(codes_list)}")
                history_parts.append("")
            
            # Format selected messages
            for i, msg in enumerate(selected_messages, 1):
                role_label = "User" if msg.role == "user" else "Assistant"
                history_parts.append(f"{i}. **{role_label}:** {msg.content}")
            
            if history_parts:
                formatted_history = "\n".join(history_parts)
                formatted_history += "\n\n**Note:** Use the conversation history above to understand context, especially when the user refers to 'those courses', 'the courses you mentioned', or asks follow-up questions. Pay attention to course codes mentioned in the history."
                return formatted_history
            else:
                return "No previous conversation."
        except Exception as e:
            logger.error(f"Error formatting history: {e}")
            return "No previous conversation."
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved results into context for the LLM"""
        if not results:
            return "No relevant information found."
        
        filtered_results = self._filter_by_rerank_score(results)
        
        context_parts = []
        
        # Extract suggestions and unfound course codes from metadata results
        suggestions_metadata = {}
        unfound_course_codes = []
        try:
            for result in filtered_results:
                if 'suggestions' in result:
                    suggestions = result.get('suggestions', {})
                    if isinstance(suggestions, dict):
                        suggestions_metadata.update(suggestions)
                if 'unfound_course_codes' in result:
                    unfound_codes = result.get('unfound_course_codes', [])
                    if isinstance(unfound_codes, list):
                        unfound_course_codes.extend(unfound_codes)
            
            # Filter out None values and ensure all codes are strings
            unfound_course_codes = [str(code) for code in unfound_course_codes if code is not None]
            unfound_course_codes = list(set(unfound_course_codes))
        except Exception as e:
            logger.warning(f"Error extracting suggestions metadata: {e}", exc_info=True)
            # Continue with empty suggestions if extraction fails
        
        courses = []
        professors = []
        curriculum = []
        studyplan = []
        
        for result in filtered_results:
            # Skip metadata results (suggestions) - already processed above
            if result.get('data_type') == 'metadata' or result.get('bypass_score_filter', False):
                continue
            
            data_type = result.get('data_type', result.get('metadata', {}).get('data_type', 'course'))
            if data_type == 'professor':
                professors.append(result)
            elif data_type == 'curriculum':
                curriculum.append(result)
            elif data_type == 'studyplan':
                studyplan.append(result)
            else:
                courses.append(result)
        
        if curriculum:
            context_parts.append("Curriculum / Graduation Requirements:")
            context_parts.append("-" * 30)
            for i, result in enumerate(curriculum, 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                source = metadata.get('source', metadata.get('filename', ''))
                if content:
                    cleaned = self._clean_html_text(content)
                    if not cleaned:
                        cleaned = self._clean_content(content)
                    context_parts.append(f"{i}. {cleaned}")
                    if source:
                        context_parts.append(f"   (Source: {source})")
            context_parts.append("")
        
        if studyplan:
            context_parts.append("Study Plan (by semester):")
            context_parts.append("-" * 30)
            for i, result in enumerate(studyplan, 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                source = metadata.get('source', metadata.get('filename', ''))
                if content:
                    cleaned = self._clean_html_text(content)
                    if not cleaned:
                        cleaned = self._clean_content(content)
                    context_parts.append(f"{i}. {cleaned}")
                    if source:
                        context_parts.append(f"   (Source: {source})")
            context_parts.append("")
        
        # Format course information
        if courses:
            context_parts.append("Courses:")
            context_parts.append("-" * 30)
            
            for i, result in enumerate(courses, 1):
                metadata = result.get('metadata', {})
                content = result.get('content', '')
                
                # Create concise course entry
                course_name = metadata.get('course_name', 'N/A')
                course_code = metadata.get('course_code', 'N/A')
                
                context_part = f"{i}. **{course_name} ({course_code})**"
                
                # Add relevance score information (for LLM to understand quality)
                hybrid_score = result.get('hybrid_score')
                rerank_score = result.get('rerank_score')
                if hybrid_score is not None:
                    context_part += f" [Relevance: {hybrid_score:.2f}]"
                elif rerank_score is not None:
                    context_part += f" [Relevance: {rerank_score:.2f}]"
                
                if content:
                    cleaned_content = self._clean_content(content)
                    context_part += f"\n   {cleaned_content}"
                
                context_parts.append(context_part)
        
        # Format professor information
        if professors:
            context_parts.append("\nProfessors:")
            context_parts.append("-" * 30)
            
            for i, result in enumerate(professors, 1):
                metadata = result.get('metadata', {})
                
                # Create structured professor entry
                name = metadata.get('name', 'N/A')
                context_part = f"{i}. **{name}**"
                
                # Add education/degrees if available
                if metadata.get('degrees'):
                    degrees = metadata['degrees']
                    if len(degrees) > 0:
                        context_part += f"\n   **Education:**"
                        for degree in degrees:
                            context_part += f"\n   - {degree}"
                
                # Add teaching subjects if available
                if metadata.get('teaching_subjects'):
                    context_part += f"\n   **Teaching:**"
                    for subject in metadata.get('teaching_subjects', []):
                        if subject not in ["01076311 - PROJECT 1", "01076312 - PROJECT 2"]:
                            context_part += f"\n   - {subject}"
                
                context_parts.append(context_part)
        
        # Show NOTE section for unfound course codes (with suggestions if available)
        if unfound_course_codes:
            context_parts.append("\nNOTE: The following course codes were not found:")
            for code in unfound_course_codes:
                try:
                    # Ensure code is a string for safe dictionary access
                    code_str = str(code) if code else ""
                    if not code_str:
                        continue
                    similar_codes = suggestions_metadata.get(code_str, [])
                    if similar_codes:
                        context_parts.append(f"- {code_str} (not found). Did you mean: {', '.join(map(str, similar_codes))}?")
                    else:
                        context_parts.append(f"- {code_str} (not found)")
                except Exception as e:
                    logger.warning(f"Error formatting course code suggestion for {code}: {e}")
                    continue
        
        return "\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        results: List[Dict[str, Any]],
        user_language: str = None,
        session_id: str = None,
        temperature: Optional[float] = None
    ):
        """Generate a streaming response generator - the only method needed for all response generation"""
        try:
            if user_language is None:
                user_language = self._detect_language(query)
            
            # Load system prompt
            prompt_file = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "prompt", "system_prompt.md"
            )
            
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read()
            except Exception as e:
                logger.error(f"Error loading system prompt: {e}")
                fallback_response = self._format_fallback_response(query, results, user_language)
                yield fallback_response
                return
            
            # Format history from database if session_id is provided
            history = self._format_history(session_id, max_messages=6, current_query=query)
            
            # Format context from retrieved results
            context = self._format_context(results) if results else "No relevant information found."
            
            # Replace placeholders in the prompt
            try:
                full_prompt = system_prompt.format(
                    history=history,
                    context=context,
                    query=query
                )
            except KeyError as e:
                logger.error(f"Missing placeholder in system prompt: {e}")
                # Fallback: use simple string replacement for known placeholders
                full_prompt = system_prompt.replace("{history}", history).replace("{context}", context).replace("{query}", query)
            except Exception as e:
                logger.error(f"Error formatting system prompt: {e}")
                # Last resort: create a minimal prompt
                full_prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nPlease provide a helpful response based on the context above."
            
            # Generate streaming response using Ollama with real-time streaming
            response_chunks = self.llm_client.generate_stream(full_prompt, temperature=temperature)
            
            if response_chunks:
                response_text = ""
                for chunk in response_chunks:
                    if chunk:
                        response_text += chunk
                        yield chunk
                 
            else:
                # Fallback response
                fallback_response = self._format_fallback_response(query, results, user_language)
                yield fallback_response
                
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}", exc_info=True)
            try:
                fallback_response = self._format_fallback_response(query, results, user_language)
                yield fallback_response
            except Exception as fallback_error:
                logger.error(f"Error in fallback response: {fallback_error}", exc_info=True)
                # Last resort: return a simple error message
                # Safely detect language with validation
                if user_language:
                    lang = user_language
                elif query and isinstance(query, str):
                    try:
                        lang = self._detect_language(query)
                    except Exception:
                        lang = "en"  # Default to English if detection fails
                else:
                    lang = "en"  # Default to English if query is invalid
                
                if lang == "th":
                    yield "ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ กรุณาลองใหม่อีกครั้ง"
                else:
                    yield "Sorry, I encountered an error generating a response. Please try again."
    
    def _format_fallback_response(self, query: str, results: List[Dict[str, Any]], user_language: str = None) -> str:
        """Format a fallback response when LLM is unavailable"""
        # Safely detect language with validation
        if user_language:
            lang = user_language
        elif query and isinstance(query, str):
            try:
                lang = self._detect_language(query)
            except Exception:
                lang = "en"  # Default to English if detection fails
        else:
            lang = "en"  # Default to English if query is invalid
        
        if not results:
            return "ขออภัย ฉันยังไม่มีข้อมูลในเรื่องนี้" if lang == "th" else "Sorry, I don't have that information yet."
        else:
            # When results exist but LLM failed, provide a generic response
            return "ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ กรุณาลองใหม่อีกครั้ง" if lang == "th" else "Sorry, I encountered an error generating a response. Please try again."

    def _contains_thai(self, text: str) -> bool:
        """Check if text contains Thai characters"""
        thai_range = range(0x0E00, 0x0E7F)
        return any(ord(char) in thai_range for char in text)
    
    
    def _get_fallback_response(self, detected_lang: str) -> str:
        """Get a simple fallback response in the detected language"""
        if detected_lang == "th":
            return "สวัสดีครับ ยินดีต้อนรับสู่บริการช่วยเหลือสำหรับนักศึกษาวิศวกรรมคอมพิวเตอร์ สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง มีอะไรให้ผมช่วยเหลือวันนี้ครับ? 😊"
        else:
            return "Hello! Welcome to the help service for Computer Engineering students at King Mongkut's Institute of Technology Ladkrabang. How can I help you today? 😊"
	