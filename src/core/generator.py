#!/usr/bin/env python3
"""
Response Generator for the RAG System
Generates contextual responses using retrieved data and system prompts
"""

import os
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
			model_name=model_name
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
	
	def _filter_by_rerank_score(self, results: List[Dict[str, Any]], 
	                           threshold: float = None) -> List[Dict[str, Any]]:
		"""Filter results based on rerank score threshold"""
		if not results:
			return results
		
		# Use config threshold if not provided
		if threshold is None:
			threshold = config.search.rerank_threshold
		
		# Filter results that have rerank scores above threshold
		filtered_results = []
		for result in results:
			rerank_score = result.get('rerank_score', 0.0)
			if rerank_score >= threshold:
				filtered_results.append(result)
		
		# Log filtering results
		original_count = len(results)
		filtered_count = len(filtered_results)
		logger.info(f"Rerank score filtering: {original_count} -> {filtered_count} results "
		           f"(threshold: {threshold:.3f})")
		
		# If no results pass the threshold, return the top 5 results anyway
		if not filtered_results and results:
			logger.warning("No results passed rerank threshold, returning top 5 results")
			# Sort by rerank score and take top 5
			sorted_results = sorted(results, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
			return sorted_results[:5]
		
		return filtered_results
	
	def _format_history(self, session_id: str, max_messages: int = 6, current_query: str = None) -> str:
		"""Format chat history from database for the prompt"""
		if not session_id or not self.chat_history_manager:
			return "No previous conversation."
		
		try:
			recent_messages = self.chat_history_manager.get_recent_messages(session_id, n=max_messages)
			if not recent_messages:
				return "No previous conversation."
			
			messages_to_format = recent_messages
			if current_query and recent_messages:
				last_msg = recent_messages[-1]
				if last_msg.role == "user" and last_msg.content.strip() == current_query.strip():
					messages_to_format = recent_messages[:-1]
			
			if not messages_to_format:
				return "No previous conversation."
			
			history_parts = []
			for msg in messages_to_format:
				role_label = "User" if msg.role == "user" else "Assistant"
				# Limit message length to avoid token overflow
				content = msg.content[:300] if len(msg.content) > 300 else msg.content
				history_parts.append(f"{role_label}: {content}")
			
			if history_parts:
				return "\n".join(history_parts)
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
		
		# Separate courses and professors
		courses = []
		professors = []
		
		for result in filtered_results:
			data_type = result.get('data_type', result.get('metadata', {}).get('data_type', 'course'))
			if data_type == 'professor':
				professors.append(result)
			else:
				courses.append(result)
		
		# Format course information
		if courses:
			context_parts.append("COURSES:")
			context_parts.append("-" * 30)
			
			for i, result in enumerate(courses, 1):
				metadata = result.get('metadata', {})
				content = result.get('content', '')
				
				# Create concise course entry
				course_name = metadata.get('course_name', 'N/A')
				course_code = metadata.get('course_code', 'N/A')
				
				context_part = f"{i}. **{course_name} ({course_code})**"
				
				if content:
					cleaned_content = self._clean_content(content)
					context_part += f"\n   {cleaned_content}"
				
				context_parts.append(context_part)
		
		# Format professor information
		if professors:
			context_parts.append("\nPROFESSORS:")
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
		
		return "\n".join(context_parts)
	
	def generate_response(
		self,
		query: str,
		results: List[Dict[str, Any]],
		user_language: str = None,
		session_id: str = None
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
			full_prompt = system_prompt.format(
				history=history,
				context=context,
				query=query
			)
			
			# Generate streaming response using Ollama with real-time streaming
			response_chunks = self.llm_client.generate_stream(full_prompt)
			
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
			logger.error(f"Error generating streaming response: {e}")
			fallback_response = self._format_fallback_response(query, results, user_language)
			yield fallback_response
	
	def _format_fallback_response(self, query: str, results: List[Dict[str, Any]], user_language: str = None) -> str:
		"""Format a fallback response when LLM is unavailable"""
		if not results:
			lang = user_language or self._detect_language(query)
			return "ขออภัย ฉันยังไม่มีข้อมูลในเรื่องนี้" if lang == "th" else "Sorry, I don't have that information yet."

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
	