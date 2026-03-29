import json
import re
from typing import List, Dict, Any, Optional, Protocol, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    original_index: int
    data_type: str  # 'course', 'professor', 'research', etc.

def create_chunk_from_data(data: Dict[str, Any]) -> DataChunk:
    """Create a DataChunk from loaded data, handling both course and professor data"""
    data_type = data.get('metadata', {}).get('data_type', 'unknown')
    
    if data_type == 'professor':
        # For professor data, generate content from metadata
        content_parts = []
        
        # Add name
        name = data['metadata'].get('name', '')
        if name:
            content_parts.append(f"Professor: {name}")
        
        # Add degrees
        degrees = data['metadata'].get('degrees', [])
        if degrees:
            degrees_text = " ".join(degrees)
            content_parts.append(f"Education: {degrees_text}")

        # Add teaching subjects
        teaching = data['metadata'].get('teaching_subjects', [])
        if teaching:
            teaching_text = ", ".join(teaching)
            content_parts.append(f"Teaching: {teaching_text}")
        
        content = " | ".join(content_parts)
    else:
        # For course data, use existing content
        content = data.get('content', '')
    
    return DataChunk(
        content=content,
        metadata=data['metadata'],
        chunk_id=data['chunk_id'],
        original_index=data['original_index'],
        data_type=data_type
    )

class DataTypeHandler(Protocol):
    """Protocol for data type handlers"""
    
    def can_handle(self, data_type: str) -> bool:
        """Check if this handler can process the given data type"""
        ...
    
    def process_data(self, data: List[Dict[str, Any]]) -> List[DataChunk]:
        """Process raw data into chunks"""
        ...
    
    def create_content(self, item: Dict[str, Any]) -> str:
        """Create content string from data item"""
        ...
    
    def create_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata from data item"""
        ...

class BaseDataTypeHandler(ABC):
    """Base class for data type handlers"""
    
    def __init__(self, data_type: str):
        self.data_type = data_type
    
    def can_handle(self, data_type: str) -> bool:
        return data_type == self.data_type
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u0e00-\u0e7f\-\.\,\:\;\(\)]', '', text)
        return text.strip()
    
    def process_data(self, data: List[Dict[str, Any]]) -> List[DataChunk]:
        """Process raw data into chunks"""
        all_chunks = []
        
        for i, item in enumerate(data):
            content = self.create_content(item)
            cleaned_content = self.clean_text(content)
            metadata = self.create_metadata(item)
            
            chunk = DataChunk(
                content=cleaned_content,
                metadata=metadata,
                chunk_id=f"{self.data_type}_{i+1:03d}",
                original_index=i,
                data_type=self.data_type
            )
            
            all_chunks.append(chunk)
        
        logger.info(f"Processed {len(data)} {self.data_type} items into {len(all_chunks)} chunks")
        return all_chunks
    
    @abstractmethod
    def create_content(self, item: Dict[str, Any]) -> str:
        """Create content string from data item"""
        pass
    
    @abstractmethod
    def create_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata from data item"""
        pass
    
    def _get_identifier(self, item: Dict[str, Any]) -> str:
        """Get a unique identifier for the item"""
        return str(item.get('id', item.get('name', 'unknown'))).replace(' ', '_')

class CourseDataHandler(BaseDataTypeHandler):
    """Handler for course data"""
    
    def __init__(self):
        super().__init__("course")
    
    def create_content(self, item: Dict[str, Any]) -> str:
        """Create content string for course"""
        return item.get('content', '')
    
    def create_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for course"""
        metadata = item.get('metadata', {})
        metadata['data_type'] = 'course'
        return metadata

class ProfessorDataHandler(BaseDataTypeHandler):
    """Handler for professor data"""
    
    def __init__(self):
        super().__init__("professor")
    
    def create_content(self, item: Dict[str, Any]) -> str:
        """Create content string for professor"""
        content_parts = []
        
        # Add name
        name = item.get('name', '')
        if name:
            content_parts.append(f"Professor: {name}")
        
        # Add degrees
        degrees = item.get('degrees', [])
        if degrees:
            degrees_text = " ".join(degrees)
            content_parts.append(f"Education: {degrees_text}")
        
        # Add teaching subjects with enhanced keywords
        teaching = item.get('teaching_subjects', [])
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
        
        return " | ".join(content_parts)
    
    def create_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for professor"""
        professor_name = item.get('name', '')
        has_thai_chars = any('\u0E00' <= char <= '\u0E7F' for char in professor_name)
        
        language = 'th' if has_thai_chars else 'en'
        metadata = {
            'data_type': 'professor',
            'name': item.get('name', ''),
            'teaching_subjects': item.get('teaching_subjects', []),
            'degrees': item.get('degrees', []),
            'language': language
        }
        
        return metadata



class DataProcessor:
    """Unified processor that can handle multiple data types"""
    
    def __init__(self):
        self.handlers: Dict[str, DataTypeHandler] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default data type handlers"""
        self.register_handler(CourseDataHandler())
        self.register_handler(ProfessorDataHandler())
    
    def register_handler(self, handler: DataTypeHandler):
        """Register a new data type handler"""
        self.handlers[handler.data_type] = handler
        logger.info(f"Registered handler for data type: {handler.data_type}")
    
    def get_handler(self, data_type: str) -> Optional[DataTypeHandler]:
        """Get handler for specific data type"""
        return self.handlers.get(data_type)
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} items from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return []
    
    def process_data(self, data: List[Dict[str, Any]], data_type: str) -> List[DataChunk]:
        """Process data using appropriate handler"""
        handler = self.get_handler(data_type)
        if not handler:
            logger.error(f"No handler found for data type: {data_type}")
            return []
        
        return handler.process_data(data)
    
    def process_file(self, file_path: str, data_type: str) -> List[DataChunk]:
        """Load and process data from file"""
        data = self.load_data(file_path)
        if not data:
            return []
        
        return self.process_data(data, data_type)
    
    def find_links(self, chunks: List[DataChunk], source_type: str, target_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """Find links between different data types"""
        links = {}
        
        source_chunks = [c for c in chunks if c.data_type == source_type]
        target_chunks = [c for c in chunks if c.data_type == target_type]
        
        for source_chunk in source_chunks:
            source_id = source_chunk.chunk_id
            linked_items = []
            
            for target_chunk in target_chunks:
                # Check for keyword matches
                if self._chunks_are_related(source_chunk, target_chunk):
                    linked_items.append({
                        'chunk_id': target_chunk.chunk_id,
                        'content': target_chunk.content[:200] + "...",
                        'metadata': target_chunk.metadata,
                        'similarity_score': self._calculate_similarity(source_chunk, target_chunk)
                    })
            
            # Sort by similarity and keep top matches
            linked_items.sort(key=lambda x: x['similarity_score'], reverse=True)
            links[source_id] = linked_items[:5]
        
        return links
    
    def _chunks_are_related(self, chunk1: DataChunk, chunk2: DataChunk) -> bool:
        """Check if two chunks are related"""
        # Get linking keywords from metadata
        keywords1 = chunk1.metadata.get('linking_keywords', [])
        keywords2 = chunk2.metadata.get('linking_keywords', [])
        
        # Check for keyword overlap
        if keywords1 and keywords2:
            overlap = set(keywords1) & set(keywords2)
            if overlap:
                return True
        
        # Check content similarity
        content1 = chunk1.content.lower()
        content2 = chunk2.content.lower()
        
        # Simple keyword matching
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        # If there's significant word overlap, consider them related
        overlap_ratio = len(words1 & words2) / min(len(words1), len(words2))
        return overlap_ratio > 0.1  # 10% word overlap threshold
    
    def _calculate_similarity(self, chunk1: DataChunk, chunk2: DataChunk) -> float:
        """Calculate similarity score between two chunks"""
        # Simple keyword-based similarity
        keywords1 = set(chunk1.metadata.get('linking_keywords', []))
        keywords2 = set(chunk2.metadata.get('linking_keywords', []))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_statistics(self, chunks: List[DataChunk]) -> Dict[str, Any]:
        """Get statistics about processed chunks"""
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'by_type': {},
            'avg_content_length': 0
        }
        
        total_length = 0
        for chunk in chunks:
            data_type = chunk.data_type
            if data_type not in stats['by_type']:
                stats['by_type'][data_type] = 0
            stats['by_type'][data_type] += 1
            total_length += len(chunk.content)
        
        stats['avg_content_length'] = total_length / len(chunks) if chunks else 0
        
        return stats
    
    def save_processed_chunks(self, chunks: List[DataChunk], file_path: str):
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
    
    def load_processed_chunks(self, file_path: str) -> List[DataChunk]:
        """Load processed chunks from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            chunks = []
            for data in chunk_data:
                chunk = create_chunk_from_data(data)
                chunks.append(chunk)
            
            logger.info(f"Loaded {len(chunks)} processed chunks from {file_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading processed chunks: {e}")
            return []
