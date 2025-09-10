#!/usr/bin/env python3
"""
Professor Data Processor for the RAG System
Handles professor-specific data processing and chunking
"""

import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProfessorChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    original_index: int

class ProfessorDataProcessor:
    def __init__(self):
        """Initialize the professor data processor"""
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u0e00-\u0e7f\-\.\,\:\;\(\)]', '', text)
        return text.strip()
    
    def create_professor_content(self, professor: Dict[str, Any]) -> str:
        """Create content string from professor data"""
        content_parts = []
        
        # Add name
        name = professor.get('name', '')
        if name:
            content_parts.append(f"Professor: {name}")
        
        # Add degrees
        degrees = professor.get('degrees', [])
        if degrees:
            degrees_text = " ".join(degrees)
            content_parts.append(f"Education: {degrees_text}")
        
        # Add research areas
        research_areas = professor.get('research_areas', [])
        if research_areas:
            research_text = ", ".join(research_areas)
            content_parts.append(f"Research Areas: {research_text}")
        
        # Add teaching subjects with enhanced keywords
        teaching = professor.get('teaching', [])
        if teaching:
            teaching_text = ", ".join(teaching)
            content_parts.append(f"Teaching: {teaching_text}")
            
            # Add enhanced keywords for better searchability
            enhanced_teaching = []
            for subject in teaching:
                enhanced_teaching.append(subject)
                # Add common variations and keywords
                if "programming" in subject.lower():
                    enhanced_teaching.append("computer programming")
                    enhanced_teaching.append("programming course")
                    if "1" in subject:
                        enhanced_teaching.append("introductory programming")
                        enhanced_teaching.append("basic programming")
                    if "2" in subject:
                        enhanced_teaching.append("advanced programming")
                if "computer" in subject.lower():
                    enhanced_teaching.append("computer science")
                if "architecture" in subject.lower():
                    enhanced_teaching.append("computer architecture")
                    enhanced_teaching.append("hardware design")
                if "network" in subject.lower():
                    enhanced_teaching.append("computer networks")
                    enhanced_teaching.append("networking")
                if "security" in subject.lower():
                    enhanced_teaching.append("cybersecurity")
                    enhanced_teaching.append("information security")
                if "database" in subject.lower():
                    enhanced_teaching.append("database systems")
                    enhanced_teaching.append("data management")
                if "operating" in subject.lower():
                    enhanced_teaching.append("operating systems")
                    enhanced_teaching.append("system software")
                if "interaction" in subject.lower():
                    enhanced_teaching.append("human computer interaction")
                    enhanced_teaching.append("user interface")
                if "image" in subject.lower():
                    enhanced_teaching.append("image processing")
                    enhanced_teaching.append("computer vision")
                if "artificial" in subject.lower() or "ai" in subject.lower():
                    enhanced_teaching.append("artificial intelligence")
                    enhanced_teaching.append("machine learning")
                if "software" in subject.lower():
                    enhanced_teaching.append("software engineering")
                    enhanced_teaching.append("software development")
            
            # Add enhanced teaching keywords
            if enhanced_teaching:
                enhanced_text = ", ".join(set(enhanced_teaching))  # Remove duplicates
                content_parts.append(f"Teaching Keywords: {enhanced_text}")
        
        # Add textbooks
        textbooks = professor.get('textbooks', [])
        if textbooks:
            textbook_text = ", ".join(textbooks)
            content_parts.append(f"Textbooks: {textbook_text}")
        
        return " | ".join(content_parts)
    
    def create_professor_metadata(self, professor: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata from professor data"""
        return {
            'data_type': 'professor',
            'name': professor.get('name', ''),
            'research_areas': professor.get('research_areas', []),
            'teaching_subjects': professor.get('teaching', []),
            'textbooks': professor.get('textbooks', []),
            'degrees': professor.get('degrees', []),
            'language': 'en'  # Set to English for better searchability
        }
    
    def process_professors(self, professors: List[Dict[str, Any]]) -> List[ProfessorChunk]:
        """Process a list of professors into chunks"""
        all_chunks = []
        
        for i, professor in enumerate(professors):
            content = self.create_professor_content(professor)
            cleaned_content = self.clean_text(content)
            metadata = self.create_professor_metadata(professor)
            
            chunk = ProfessorChunk(
                content=cleaned_content,
                metadata=metadata,
                chunk_id=f"professor_{i+1:03d}",
                original_index=i
            )
            all_chunks.append(chunk)
        
        logger.info(f"Processed {len(professors)} professors into {len(all_chunks)} chunks")
        return all_chunks
    
    def find_professor_course_links(self, professor_chunks: List[ProfessorChunk], 
                                   course_chunks: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Find links between professors and courses based on teaching subjects"""
        links = {}
        
        for prof_chunk in professor_chunks:
            prof_id = prof_chunk.chunk_id
            linked_courses = []
            
            teaching_subjects = prof_chunk.metadata.get('teaching_subjects', [])
            
            for course_chunk in course_chunks:
                # Check if professor teaches this course
                course_name = course_chunk.metadata.get('course_name', '')
                course_code = course_chunk.metadata.get('course_code', '')
                
                # Simple keyword matching
                for subject in teaching_subjects:
                    if (subject.lower() in course_name.lower() or 
                        subject.lower() in course_code.lower() or
                        any(keyword in course_chunk.content.lower() 
                            for keyword in subject.lower().split())):
                        linked_courses.append({
                            'course_chunk_id': course_chunk.chunk_id,
                            'course_name': course_name,
                            'course_code': course_code,
                            'match_reason': f"Teaches: {subject}"
                        })
                        break
            
            links[prof_id] = linked_courses
        
        return links
    
    def get_chunk_texts(self, chunks: List[ProfessorChunk]) -> List[str]:
        """Extract text content from chunks"""
        return [chunk.content for chunk in chunks]
    
    def get_chunk_metadata(self, chunks: List[ProfessorChunk]) -> List[Dict[str, Any]]:
        """Extract metadata from chunks"""
        return [chunk.metadata for chunk in chunks]
    
    def search_professors_by_keyword(self, chunks: List[ProfessorChunk], keyword: str) -> List[ProfessorChunk]:
        """Search professor chunks by keyword in content or metadata"""
        keyword_lower = keyword.lower()
        matching_chunks = []
        
        for chunk in chunks:
            # Search in content
            if keyword_lower in chunk.content.lower():
                matching_chunks.append(chunk)
                continue
            
            # Search in metadata fields
            metadata = chunk.metadata
            for field in ['name', 'research_areas', 'teaching_subjects', 'textbooks', 'degrees']:
                if field in metadata:
                    field_value = metadata[field]
                    if isinstance(field_value, list):
                        if any(keyword_lower in str(item).lower() for item in field_value):
                            matching_chunks.append(chunk)
                            break
                    elif keyword_lower in str(field_value).lower():
                        matching_chunks.append(chunk)
                        break
        
        return matching_chunks
    
    def get_professor_statistics(self, chunks: List[ProfessorChunk]) -> Dict[str, Any]:
        """Get statistics about processed professor chunks"""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        research_areas = {}
        teaching_subjects = {}
        degrees = {}
        
        for chunk in chunks:
            # Count research areas
            for area in chunk.metadata.get('research_areas', []):
                research_areas[area] = research_areas.get(area, 0) + 1
            
            # Count teaching subjects
            for subject in chunk.metadata.get('teaching_subjects', []):
                teaching_subjects[subject] = teaching_subjects.get(subject, 0) + 1
            
            # Count degrees
            for degree in chunk.metadata.get('degrees', []):
                degrees[degree] = degrees.get(degree, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'research_areas': research_areas,
            'teaching_subjects': teaching_subjects,
            'degrees': degrees
        }
    
    def save_processed_professors(self, chunks: List[ProfessorChunk], file_path: str):
        """Save processed professor chunks to JSON file"""
        try:
            chunk_data = []
            for chunk in chunks:
                chunk_dict = {
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'chunk_id': chunk.chunk_id,
                    'original_index': chunk.original_index
                }
                chunk_data.append(chunk_dict)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(chunks)} processed professor chunks to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed professor chunks: {e}")
    
    def load_processed_professors(self, file_path: str) -> List[ProfessorChunk]:
        """Load processed professor chunks from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            chunks = []
            for data in chunk_data:
                chunk = ProfessorChunk(
                    content=data['content'],
                    metadata=data['metadata'],
                    chunk_id=data['chunk_id'],
                    original_index=data['original_index']
                )
                chunks.append(chunk)
            
            logger.info(f"Loaded {len(chunks)} processed professor chunks from {file_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading processed professor chunks: {e}")
            return []
