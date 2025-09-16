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

logger = get_logger(__name__)


class ResponseGenerator:
	"""Generates contextual responses using retrieved data and LLM"""
	
	def __init__(self, model_name: str = None):
		"""Initialize the response generator"""
		self.model_name = model_name or config.models.ollama_model
		self.ollama_url = config.models.ollama_url
		self.chat_history = []
		self.cache = {}  # Simple cache for responses
		self.cache_max_size = 100
		self.retry_count = 3
		self.retry_delay = 1.0
	
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
				
				# Add focus areas if available
				if metadata.get('focus_areas'):
					focus_areas = ', '.join(metadata['focus_areas'][:2])  # Limit to 2 areas
					context_part += f" - {focus_areas}"
				
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
				
				# Create concise professor entry
				name = metadata.get('name', 'N/A')
				context_part = f"{i}. {name}"
				
				# Add research areas if available
				if metadata.get('research_areas'):
					research_areas = ', '.join(metadata['research_areas'])
					context_part += f" - {research_areas}"
				
				# Add teaching subjects if available
				if metadata.get('teaching_subjects'):
					subjects = ', '.join(metadata['teaching_subjects'])
					context_part += f"\n   Teaches: {subjects}"
				
				# Add truncated description
				if content:
					clean_content = self._clean_content(content)
					context_part += f"\n   {clean_content}"
				
				context_parts.append(context_part)
		
		return "\n".join(context_parts)
	
	def generate_response(self, query: str, results: List[Dict[str, Any]], user_language: str = None) -> str:
		"""Generate a contextual response based on retrieved data"""
		try:
			if not results:
				return self._format_fallback_response(query, results, user_language)
			
			# Filter results by user's language preference if specified
			if user_language:
				filtered_results = []
				for result in results:
					metadata = result.get('metadata', {})
					result_language = metadata.get('language', 'en')
					if result_language == user_language:
						filtered_results.append(result)
				
				if filtered_results:
					results = filtered_results
					logger.info(f"Filtered results to {user_language} language: {len(results)} courses")
				else:
					logger.warning(f"No results found in {user_language} language, using all results")
			
			# Load system prompt directly
			prompt_file = os.path.join(
				os.path.dirname(__file__), 
				"..", "..", "prompt", "system_prompt.md"
			)
			
			try:
				with open(prompt_file, 'r', encoding='utf-8') as f:
					system_prompt = f.read()
			except Exception as e:
				logger.error(f"Error loading system prompt: {e}")
				return self._format_fallback_response(query, results, user_language)
			
			# Format context from retrieved results
			context = self._format_context(results)
			
			# Detect language for response
			detected_lang = user_language or self._detect_language(query)
			
			# Build context section
			system_prompt = system_prompt.replace("{context}", context)

			full_prompt = system_prompt.format(
				query=query
			)
			
			# Generate response using Ollama
			response = self._call_ollama(full_prompt, temperature=0.3)
			
			if response:
				# Ensure response is in the same language as the query
				if detected_lang == 'th' and not self._contains_thai(response):
					# Try to generate Thai response
					thai_prompt = f"{full_prompt}\n\nIMPORTANT: Respond in Thai language."
					thai_response = self._call_ollama(thai_prompt)
					if thai_response:
						response = thai_response
				
				# Update chat history
				self._update_chat_history(query, response.strip())
				
				return response.strip()
			else:
				return self._format_fallback_response(query, results, user_language)
				
		except Exception as e:
			logger.error(f"Error generating response: {e}")
			return self._format_fallback_response(query, results, user_language)
	
	def generate_conversational_response(self, query: str) -> str:
		"""Generate a conversational response using Gemma3 instead of hardcoded patterns"""
		try:
			detected_lang = self._detect_language(query)
			
			# Build chat history section
			chat_history_section = ""
			if self.chat_history:
				chat_history_section = "\n**Recent Conversation:**\n"
				for i, (user_msg, assistant_msg) in enumerate(self.chat_history[-3:], 1):
					chat_history_section += f"{i}. User: {user_msg}\n   Assistant: {assistant_msg}\n"
				chat_history_section += "\n"
			
			# Load system prompt directly and format with minimal parameters
			prompt_file = os.path.join(
				os.path.dirname(__file__), 
				"..", "..", "prompt", "system_prompt.md"
			)
			
			with open(prompt_file, 'r', encoding='utf-8') as f:
				system_prompt = f.read()
				
			prompt = system_prompt.format(
				chat_history=chat_history_section,
				query=query,
				context="",
				num_results=0
			)

			response = self._call_ollama(prompt, temperature=0.7)
			
			if response:
				# Ensure response is in the same language as the query
				if detected_lang == 'th' and not self._contains_thai(response):
					# Try to generate Thai response
					thai_prompt = f"{prompt}\n\nIMPORTANT: Respond in Thai language."
					thai_response = self._call_ollama(thai_prompt, temperature=0.7)
					if thai_response:
						response = thai_response
				
				# Update chat history
				self._update_chat_history(query, response.strip())
				
				return response.strip()
			else:
				# Fallback to simple responses if Gemma3 fails
				return self._get_fallback_response(detected_lang)
					
		except Exception as e:
			logger.error(f"Error generating conversational response: {e}")
			# Fallback response
			detected_lang = self._detect_language(query)
			return self._get_fallback_response(detected_lang)
	
	def generate_response_stream(self, query: str, results: List[Dict[str, Any]], user_language: str = None) -> str:
		"""Generate a streaming response for better user experience"""
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
				return self._format_fallback_response(query, results, user_language)
			
			# Format context from retrieved results
			context = self._format_context(results)

			system_prompt = system_prompt.replace("{context}", context)

			full_prompt = system_prompt.format(
				query=query
			)
			
			# Generate streaming response using Ollama
			response = self._call_ollama(full_prompt, stream=True)
			
			if response:
				# Ensure response is in the same language as the query
				if user_language == 'th' and not self._contains_thai(response):
					# Try to generate Thai response
					thai_prompt = f"{full_prompt}\n\nIMPORTANT: Respond in Thai language."
					thai_response = self._call_ollama(thai_prompt, stream=True)
					if thai_response:
						response = thai_response
				
				# Update chat history
				self._update_chat_history(query, response.strip())
				
				return response.strip()
			else:
				return self._format_fallback_response(query, results, user_language)
				
		except Exception as e:
			logger.error(f"Error generating streaming response: {e}")
			return self._format_fallback_response(query, results, user_language)
	
	def generate_response_stream_generator(self, query: str, results: List[Dict[str, Any]], user_language: str = None):
		"""Generate a streaming response generator for real-time output"""
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
			
			# Format context from retrieved results
			context = self._format_context(results)
			
			system_prompt = system_prompt.replace("{context}", context)

			full_prompt = system_prompt.format(
				query=query
			)
			
			# Generate streaming response using Ollama with real-time streaming
			response_chunks = self._call_ollama_stream(full_prompt)
			
			if response_chunks:
				response_text = ""
				for chunk in response_chunks:
					if chunk:
						response_text += chunk
						yield chunk
				
				# Ensure response is in the same language as the query
				if user_language == 'th' and not self._contains_thai(response_text):
					# Try to generate Thai response
					thai_prompt = f"{full_prompt}\n\nIMPORTANT: Respond in Thai language."
					thai_chunks = self._call_ollama_stream(thai_prompt)
					if thai_chunks:
						# Clear previous response and stream Thai response
						for chunk in thai_chunks:
							if chunk:
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
	
	def _call_ollama(self, prompt: str, temperature: float = 0.7, stream: bool = False) -> Optional[str]:
		"""Call Ollama API with the given prompt with retry logic and caching"""
		# Check cache first for non-streaming requests
		if not stream:
			cache_key = f"{prompt}_{temperature}"
			if cache_key in self.cache:
				logger.debug("Using cached response")
				return self.cache[cache_key]
		
		for attempt in range(self.retry_count):
			try:
				# Check Ollama service health
				try:
					health_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
					if health_response.status_code != 200:
						logger.error(f"Ollama service not responding: {health_response.status_code}")
						if attempt < self.retry_count - 1:
							time.sleep(self.retry_delay * (attempt + 1))
							continue
						return None
					
					# Check if model is available
					models_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
					if models_response.status_code == 200:
						available_models = models_response.json().get('models', [])
						model_names = [model.get('name', '') for model in available_models]
						if self.model_name not in model_names:
							logger.error(f"Model {self.model_name} not available. Available models: {model_names}")
							if attempt < self.retry_count - 1:
								time.sleep(self.retry_delay * (attempt + 1))
								continue
							return None
						logger.info(f"Model {self.model_name} is available")
					else:
						logger.warning("Could not verify model availability")
						
				except requests.exceptions.RequestException as e:
					logger.error(f"Ollama service health check failed: {e}")
					if attempt < self.retry_count - 1:
						time.sleep(self.retry_delay * (attempt + 1))
						continue
					return None
				
				payload = {
					"model": self.model_name,
					"prompt": prompt,
					"stream": stream,
					"temperature": temperature,
					"options": {
						"max_tokens": 500
					}
				}
				
				if stream:
					# Handle streaming response
					response = requests.post(f"{self.ollama_url}/api/generate", json=payload, stream=True, timeout=60)
					if response.status_code != 200:
						logger.error(f"Ollama API error: {response.status_code}")
						if attempt < self.retry_count - 1:
							time.sleep(self.retry_delay * (attempt + 1))
							continue
						return None
					
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
					response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=60)
					if response.status_code != 200:
						logger.error(f"Ollama API error: {response.status_code}")
						if attempt < self.retry_count - 1:
							time.sleep(self.retry_delay * (attempt + 1))
							continue
						return None
					
					response_data = response.json()
					response_text = response_data.get('response', '')
					
					# Check for short/generic responses
					if len(response_text) < 50:
						logger.warning(f"Ollama response too short ({len(response_text)} chars): {response_text[:100]}")
						if attempt < self.retry_count - 1:
							time.sleep(self.retry_delay * (attempt + 1))
							continue
						return None
					
					# Cache the result
					if len(self.cache) >= self.cache_max_size:
						# Remove oldest entry
						oldest_key = next(iter(self.cache))
						del self.cache[oldest_key]
					
					self.cache[cache_key] = response_text
					return response_text
					
			except Exception as e:
				logger.error(f"Error calling Ollama (attempt {attempt + 1}): {e}")
				if attempt < self.retry_count - 1:
					time.sleep(self.retry_delay * (attempt + 1))
					continue
				return None
		
		return None
	
	def _call_ollama_stream(self, prompt: str, temperature: float = 0.3):
		"""Call Ollama API with streaming and return a generator for real-time chunks"""
		try:
			# Check Ollama service health
			try:
				health_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
				if health_response.status_code != 200:
					logger.error(f"Ollama service not responding: {health_response.status_code}")
					return None
				
				# Check if model is available
				models_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
				if models_response.status_code == 200:
					available_models = models_response.json().get('models', [])
					model_names = [model.get('name', '') for model in available_models]
					if self.model_name not in model_names:
						logger.error(f"Model {self.model_name} not available. Available models: {model_names}")
						return None
					logger.info(f"Model {self.model_name} is available")
				else:
					logger.warning("Could not verify model availability")
					
			except requests.exceptions.RequestException as e:
				logger.error(f"Ollama service health check failed: {e}")
				return None
			
			payload = {
				"model": self.model_name,
				"prompt": prompt,
				"stream": True,
				"temperature": temperature,
				"options": {
					"max_tokens": 500
				}
			}
			
			# Make streaming request
			response = requests.post(f"{self.ollama_url}/api/generate", json=payload, stream=True, timeout=60)
			if response.status_code != 200:
				logger.error(f"Ollama API error: {response.status_code}")
				return None
			
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
				
		except Exception as e:
			logger.error(f"Error calling Ollama stream: {e}")
			return None
	
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
	