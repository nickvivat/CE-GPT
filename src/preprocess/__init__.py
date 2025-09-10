"""
Data preprocessing module for the RAG system.
Contains processors for different data types and unified processing logic.
"""

from .data_processor import DataProcessor, DataChunk
from .course_processor import CourseDataProcessor, CourseChunk
from .professor_processor import ProfessorDataProcessor, ProfessorChunk

__all__ = [
    'DataProcessor',
    'DataChunk', 
    'CourseDataProcessor',
    'CourseChunk',
    'ProfessorDataProcessor',
    'ProfessorChunk'
]
