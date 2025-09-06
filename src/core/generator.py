#!/usr/bin/env python3
"""
Response Generator for the RAG System
Generates contextual responses using retrieved data and system prompts
"""

import os
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
	
	def _detect_language(self, text: str) -> str:
		"""Detect if text contains any Thai character; if so, respond in Thai, else English"""
		for char in text:
			if 0x0E00 <= ord(char) <= 0x0E7F:
				return "th"
		return "en"
	
	def _format_context(self, results: List[Dict[str, Any]]) -> str:
		"""Format retrieved results into context for the LLM"""
		if not results:
			return "No relevant information found."
		
		context_parts = []
		context_parts.append("RETRIEVED COURSE INFORMATION:")
		context_parts.append("=" * 50)
		
		for i, result in enumerate(results, 1):
			metadata = result.get('metadata', {})
			content = result.get('content', '')
			
			context_part = f"\nCOURSE {i}:"
			context_part += f"\n- Course Name: {metadata.get('course_name', 'N/A')}"
			context_part += f"\n- Course Code: {metadata.get('course_code', 'N/A')}"
			context_part += f"\n- Language: {metadata.get('language', 'N/A')}"
			
			if metadata.get('focus_areas'):
				context_part += f"\n- Focus Areas: {', '.join(metadata['focus_areas'])}"
			
			if metadata.get('career_tracks'):
				context_part += f"\n- Career Tracks: {', '.join(metadata['career_tracks'])}"
			
			if content:
				clean_content = content.replace('\n', ' ').strip()
				context_part += f"\n- Description: {clean_content}"
			
			context_parts.append(context_part)
		
		context_parts.append("\n" + "=" * 50)
		
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
			context_section = f"**Context:**\n{context}"
			
			chat_history_section = ""
			if self.chat_history:
				chat_history_section = "\n**Recent Conversation:**\n"
				for i, (user_msg, assistant_msg) in enumerate(self.chat_history[-3:], 1):
					chat_history_section += f"{i}. User: {user_msg}\n   Assistant: {assistant_msg}\n"
				chat_history_section += "\n"

			full_prompt = system_prompt.format(
				chat_history=chat_history_section,
				query=query,
				context=context_section,
				num_results=len(results)
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
			
			# Build context section
			context_section = f"**Context:**\n{context}"
			
			# Build response prompt with chat history
			chat_history_section = ""
			if self.chat_history:
				chat_history_section = "\n**Recent Conversation:**\n"
				for i, (user_msg, assistant_msg) in enumerate(self.chat_history[-3:], 1):
					chat_history_section += f"{i}. User: {user_msg}\n   Assistant: {assistant_msg}\n"
				chat_history_section += "\n"

			full_prompt = system_prompt.format(
				chat_history=chat_history_section,
				query=query,
				context=context_section,
				num_results=len(results)
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
			
			# Build context section
			context_section = f"**Context:**\n{context}"
			
			# Build response prompt with chat history
			chat_history_section = ""
			if self.chat_history:
				chat_history_section = "\n**Recent Conversation:**\n"
				for i, (user_msg, assistant_msg) in enumerate(self.chat_history[-3:], 1):
					chat_history_section += f"{i}. User: {user_msg}\n   Assistant: {assistant_msg}\n"
				chat_history_section += "\n"

			full_prompt = system_prompt.format(
				chat_history=chat_history_section,
				query=query,
				context=context_section,
				num_results=len(results)
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
		"""Call Ollama API with the given prompt"""
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
				"stream": stream,
				"temperature": 0.3
			}
			
			if stream:
				# Handle streaming response
				response = requests.post(f"{self.ollama_url}/api/generate", json=payload, stream=True, timeout=60)
				if response.status_code != 200:
					logger.error(f"Ollama API error: {response.status_code}")
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
					return None
				
				response_data = response.json()
				response_text = response_data.get('response', '')
				
				# Check for short/generic responses
				if len(response_text) < 50:
					logger.warning(f"Ollama response too short ({len(response_text)} chars): {response_text[:100]}")
					return None
				
				return response_text
				
		except Exception as e:
			logger.error(f"Error calling Ollama: {e}")
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
				"temperature": temperature
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
		
		lang = user_language or self._detect_language(query)
		
		query_lower = query.lower()
		is_calculus_query = any(word in query_lower for word in ['calculus', 'math', 'mathematics', 'derivative', 'integral', 'แคลคูลัส', 'คณิตศาสตร์', 'อนุพันธ์', 'อินทิกรัล', 'แคล', 'คณิต'])
		is_programming_query = any(word in query_lower for word in ['programming', 'coding', 'software', 'development', 'python', 'java', 'c++', 'เขียนโปรแกรม', 'ซอฟต์แวร์', 'พัฒนา'])
		is_hardware_query = any(word in query_lower for word in ['hardware', 'circuit', 'electronics', 'embedded', 'microcontroller', 'ฮาร์ดแวร์', 'วงจร', 'อิเล็กทรอนิกส์'])
		is_ai_query = any(word in query_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'neural', 'deep learning', 'ปัญญาประดิษฐ์', 'การเรียนรู้ของเครื่อง'])
		
		if lang == "th":
			if is_calculus_query:
				response = f"เกี่ยวกับวิชาแคลคูลัสที่ KMITL มีหลักสูตรดังนี้:\n\n"
			elif is_programming_query:
				response = f"เกี่ยวกับวิชาการเขียนโปรแกรมที่ KMITL มีหลักสูตรดังนี้:\n\n"
			elif is_hardware_query:
				response = f"เกี่ยวกับวิชาอุปกรณ์อิเล็กทรอนิกส์ที่ KMITL มีหลักสูตรดังนี้:\n\n"
			elif is_ai_query:
				response = f"เกี่ยวกับวิชาปัญญาประดิษฐ์ที่ KMITL มีหลักสูตรดังนี้:\n\n"
			else:
				response = f"พบ {len(results)} หลักสูตรที่เกี่ยวข้องกับ '{query}':\n\n"
		else:
			if is_calculus_query:
				response = f"Regarding calculus courses at KMITL, here's what I found:\n\n"
			elif is_programming_query:
				response = f"Regarding programming courses at KMITL, here's what I found:\n\n"
			elif is_hardware_query:
				response = f"Regarding hardware and electronics courses at KMITL, here's what I found:\n\n"
			elif is_ai_query:
				response = f"Regarding AI and machine learning courses at KMITL, here's what I found:\n\n"
			else:
				response = f"Found {len(results)} courses related to '{query}':\n\n"
		
		# Group courses by relevance and provide focused information
		primary_courses = []
		related_courses = []
		
		for result in results:
			metadata = result.get('metadata', {})
			course_name = metadata.get('course_name', '').lower()
			focus_areas = [area.lower() for area in metadata.get('focus_areas', [])]
			
			# Categorize based on query type
			if is_calculus_query and any(word in course_name for word in ['calculus', 'differential', 'integral', 'แคลคูลัส', 'อนุพันธ์', 'อินทิกรัล', 'แคล', 'คณิต', 'math', 'mathematics']):
				primary_courses.append(result)
			elif is_programming_query and any(word in course_name for word in ['programming', 'software', 'development', 'เขียนโปรแกรม', 'ซอฟต์แวร์']) or 'software-development' in focus_areas:
				primary_courses.append(result)
			elif is_hardware_query and any(word in course_name for word in ['circuit', 'electronics', 'hardware', 'วงจร', 'อิเล็กทรอนิกส์', 'ฮาร์ดแวร์']) or 'hardware-electronics' in focus_areas:
				primary_courses.append(result)
			elif is_ai_query and any(word in course_name for word in ['ai', 'intelligence', 'machine', 'neural', 'ปัญญาประดิษฐ์', 'การเรียนรู้']) or 'ai-data-science' in focus_areas:
				primary_courses.append(result)
			else:
				related_courses.append(result)
		
		# Prioritize primary courses for specific queries
		if primary_courses:
			section_title = "**หลักสูตรหลัก:**" if lang == "th" else "**Core Courses:**"
			response += f"{section_title}\n"
			for i, result in enumerate(primary_courses[:3], 1):
				metadata = result.get('metadata', {})
				if lang == "th":
					response += f"{i}. **{metadata.get('course_name', 'N/A')}** ({metadata.get('course_code', 'N/A')})\n"
					response += f"   📚 **รายละเอียด:** {result.get('content', '')[:200]}...\n"
					response += f"   🌍 **ภาษา:** {metadata.get('language', 'N/A')}\n"
					response += f"   🎯 **พื้นที่โฟกัส:** {', '.join(metadata.get('focus_areas', []))}\n"
					response += f"   🚀 **เส้นทางอาชีพ:** {', '.join(metadata.get('career_tracks', []))}\n\n"
				else:
					response += f"{i}. **{metadata.get('course_name', 'N/A')}** ({metadata.get('course_code', 'N/A')})\n"
					response += f"   📚 **Description:** {result.get('content', '')[:200]}...\n"
					response += f"   🌍 **Language:** {metadata.get('language', 'N/A')}\n"
					response += f"   🎯 **Focus Areas:** {', '.join(metadata.get('focus_areas', []))}\n"
					response += f"   🚀 **Career Tracks:** {', '.join(metadata.get('career_tracks', []))}\n\n"
			
			if related_courses:
				section_title = "**หลักสูตรที่เกี่ยวข้อง:**" if lang == "th" else "**Related Courses:**"
				response += f"{section_title}\n"
				for i, result in enumerate(related_courses[:2], 1):
					metadata = result.get('metadata', {})
					if lang == "th":
						response += f"{i}. {metadata.get('course_name', 'N/A')} ({metadata.get('course_code', 'N/A')})\n"
					else:
						response += f"{i}. {metadata.get('course_name', 'N/A')} ({metadata.get('course_code', 'N/A')})\n"
		else:
			# Standard listing for non-calculus queries
			for i, result in enumerate(results[:5], 1):
				metadata = result.get('metadata', {})
				if lang == "th":
					response += f"{i}. **หลักสูตร:** {metadata.get('course_name', 'N/A')} ({metadata.get('course_code', 'N/A')})\n"
					response += f"   📚 **รายละเอียด:** {result.get('content', '')[:150]}...\n"
					response += f"   🌍 **ภาษา:** {metadata.get('language', 'N/A')}\n"
					response += f"   🎯 **พื้นที่โฟกัส:** {', '.join(metadata.get('focus_areas', []))}\n\n"
				else:
					response += f"{i}. **Course:** {metadata.get('course_name', 'N/A')} ({metadata.get('course_code', 'N/A')})\n"
					response += f"   📚 **Description:** {result.get('content', '')[:150]}...\n"
					response += f"   🌍 **Language:** {metadata.get('language', 'N/A')}\n"
					response += f"   🎯 **Focus Areas:** {', '.join(metadata.get('focus_areas', []))}\n\n"
		
		# Add summary
		if lang == "th":
			response += f"\n🎓 **สรุป:** พบทั้งหมด {len(results)} หลักสูตรที่เกี่ยวข้องกับคำถามของคุณ\n"
			response += f"💡 **คำแนะนำ:** หลักสูตรเหล่านี้จะช่วยให้คุณเข้าใจพื้นฐานและพัฒนาทักษะที่จำเป็นสำหรับสาขาวิชาวิศวกรรมคอมพิวเตอร์"
		else:
			response += f"\n🎓 **Summary:** Found {len(results)} courses related to your query\n"
			response += f"💡 **Recommendation:** These courses will help you understand the fundamentals and develop essential skills for Computer Engineering"
		
		return response.strip()
	
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
	