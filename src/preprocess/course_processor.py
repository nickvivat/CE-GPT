#!/usr/bin/env python3
"""
Course Data Processor for the RAG System
Handles course-specific data processing and chunking
"""

import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CourseChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    original_index: int

class CourseDataProcessor:
    def __init__(self):
        """Initialize the course data processor with course-based chunking"""
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u0e00-\u0e7f\-\.\,\:\;\(\)]', '', text)
        return text.strip()
    
    def create_chunk(self, course: Dict[str, Any], chunk_id: str, 
                     original_index: int) -> List[CourseChunk]:
        """Create chunks from a single course"""
        chunks = []
        
        # Create main chunk with course information
        metadata = course.get('metadata', {})
        metadata['data_type'] = 'course'  # Ensure data_type is set
        
        chunk = CourseChunk(
            content=self.clean_text(course.get('content', '')),
            metadata=metadata,
            chunk_id=chunk_id,
            original_index=original_index
        )
        chunks.append(chunk)
        
        return chunks
    
    def process_courses(self, courses: List[Dict[str, Any]]) -> List[CourseChunk]:
        """Process a list of courses into chunks"""
        all_chunks = []
        
        for i, course in enumerate(courses):
            chunk_id = f"course_{i+1:03d}"
            chunks = self.create_chunk(course, chunk_id, i)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(courses)} courses into {len(all_chunks)} chunks")
        return all_chunks
    
    def get_chunk_texts(self, chunks: List[CourseChunk]) -> List[str]:
        """Extract text content from chunks"""
        return [chunk.content for chunk in chunks]
    
    def get_chunk_metadata(self, chunks: List[CourseChunk]) -> List[Dict[str, Any]]:
        """Extract metadata from chunks"""
        return [chunk.metadata for chunk in chunks]
    
    def filter_chunks_by_language(self, chunks: List[CourseChunk], language: str) -> List[CourseChunk]:
        """Filter chunks by language"""
        return [chunk for chunk in chunks if chunk.metadata.get('language') == language]
    
    def filter_chunks_by_focus_area(self, chunks: List[CourseChunk], focus_area: str) -> List[CourseChunk]:
        """Filter chunks by focus area"""
        return [chunk for chunk in chunks if focus_area in chunk.metadata.get('focus_areas', [])]
    
    def filter_chunks_by_career_track(self, chunks: List[CourseChunk], career_track: str) -> List[CourseChunk]:
        """Filter chunks by career track"""
        return [chunk for chunk in chunks if career_track in chunk.metadata.get('career_tracks', [])]
    
    def search_chunks_by_keyword(self, chunks: List[CourseChunk], keyword: str) -> List[CourseChunk]:
        """Search chunks by keyword in content"""
        keyword_lower = keyword.lower()
        return [chunk for chunk in chunks if keyword_lower in chunk.content.lower()]
    
    def get_statistics(self, chunks: List[CourseChunk]) -> Dict[str, Any]:
        """Get statistics about processed chunks"""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        languages = {}
        focus_areas = {}
        career_tracks = {}
        
        for chunk in chunks:
            # Count languages
            lang = chunk.metadata.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
            
            # Count focus areas
            for area in chunk.metadata.get('focus_areas', []):
                focus_areas[area] = focus_areas.get(area, 0) + 1
            
            # Count career tracks
            for track in chunk.metadata.get('career_tracks', []):
                career_tracks[track] = career_tracks.get(track, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'languages': languages,
            'focus_areas': focus_areas,
            'career_tracks': career_tracks
        }
    
    def save_processed_chunks(self, chunks: List[CourseChunk], file_path: str):
        """Save processed chunks to JSON file"""
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
            
            logger.info(f"Saved {len(chunks)} processed chunks to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed chunks: {e}")
    
    def load_processed_chunks(self, file_path: str) -> List[CourseChunk]:
        """Load processed chunks from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            chunks = []
            for data in chunk_data:
                chunk = CourseChunk(
                    content=data['content'],
                    metadata=data['metadata'],
                    chunk_id=data['chunk_id'],
                    original_index=data['original_index']
                )
                chunks.append(chunk)
            
            logger.info(f"Loaded {len(chunks)} processed chunks from {file_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading processed chunks: {e}")
            return []
