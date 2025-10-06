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
	
	def __init__(self, model_name: str = None):
		"""Initialize the response generator"""
		self.llm_client = LLMClient(
			provider=LLMProvider.OLLAMA,
			model_name=model_name
		)
		self.chat_history = []
	
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
	
	def _format_context(self, results: List[Dict[str, Any]]) -> str:
		"""Format retrieved results into context for the LLM"""
		if not results:
			return "No relevant information found."
		
		
		context_parts = []
		
		# Separate courses and professors
		courses = []
		professors = []
		
		for result in results:
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
				
				context_part = f"{i}. {course_name} ({course_code})"
				
				# Add truncated description
				if content:
					clean_content = self._clean_content(content)
				
				context_parts.append(context_part)
		
		# Format professor information
		if professors:
			context_parts.append("\nPROFESSORS:")
			context_parts.append("-" * 30)
			
			for i, result in enumerate(professors, 1):
				metadata = result.get('metadata', {})
				content = result.get('content', '')
				
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
					for subject in metadata['teaching_subjects']:
						context_part += f"\n   - {subject}"
				
				# Add truncated description
				if content:
					clean_content = self._clean_content(content)
					context_part += f"\n   {clean_content}"
				
				context_parts.append(context_part)
		
		return "\n".join(context_parts)
	
	def generate_response(self, query: str, results: List[Dict[str, Any]], user_language: str = None):
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
			
			# Handle conversational responses (when results are empty)
			if not results:
				# Build chat history section for conversational responses
				chat_history_section = ""
				if self.chat_history:
					chat_history_section = "\n**Recent Conversation:**\n"
					for i, (user_msg, assistant_msg) in enumerate(self.chat_history[-3:], 1):
						chat_history_section += f"{i}. User: {user_msg}\n   Assistant: {assistant_msg}\n"
					chat_history_section += "\n"
				
				full_prompt = system_prompt.format(
					chat_history=chat_history_section,
					query=query,
					context="",
					num_results=0
				)
			else:
				# Format context from retrieved results for contextual responses
				context = self._format_context(results)
				system_prompt = system_prompt.replace("{context}", context)
				full_prompt = system_prompt.format(query=query)
			
			# Generate streaming response using Ollama with real-time streaming
			response_chunks = self.llm_client.generate_stream(full_prompt)
			
			if response_chunks:
				response_text = ""
				for chunk in response_chunks:
					if chunk:
						response_text += chunk
						yield chunk
				 
				# Update chat history
				self._update_chat_history(query, response_text.strip())
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
	
	def _update_chat_history(self, user_query: str, assistant_response: str):
		"""Update chat history with the latest exchange"""
		self.chat_history.append((user_query, assistant_response))
		
		if len(self.chat_history) > 10:
			self.chat_history = self.chat_history[-10:]
	
	def _get_fallback_response(self, detected_lang: str) -> str:
		"""Get a simple fallback response in the detected language"""
		if detected_lang == "th":
			return "สวัสดีครับ ยินดีต้อนรับสู่บริการช่วยเหลือสำหรับนักศึกษาวิศวกรรมคอมพิวเตอร์ สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง มีอะไรให้ผมช่วยเหลือวันนี้ครับ? 😊"
		else:
			return "Hello! Welcome to the help service for Computer Engineering students at King Mongkut's Institute of Technology Ladkrabang. How can I help you today? 😊"
	