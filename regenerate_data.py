#!/usr/bin/env python3
"""
Script to regenerate processed data after removing textbooks and research_areas
"""

import sys
import os
sys.path.append('src')

from src.preprocess.professor_processor import ProfessorDataProcessor
from src.preprocess.course_processor import CourseDataProcessor
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Regenerate processed data files"""
    
    # Process professor data
    logger.info("Processing professor data...")
    professor_processor = ProfessorDataProcessor()
    
    # Load raw professor data
    with open('data/raw/professor_detail.json', 'r', encoding='utf-8') as f:
        raw_professors = json.load(f)
    
    # Process professors
    professor_chunks = professor_processor.process_professors(raw_professors)
    
    # Save processed professors
    professor_processor.save_processed_professors(professor_chunks, 'data/processed/professor_detail_processed.json')
    logger.info(f"Saved {len(professor_chunks)} processed professor chunks")
    
    # Process course data (no changes needed, but let's verify it works)
    logger.info("Processing course data...")
    course_processor = CourseDataProcessor()
    
    # Load raw course data
    with open('data/raw/course_detail.json', 'r', encoding='utf-8') as f:
        raw_courses = json.load(f)
    
    # Process courses
    course_chunks = course_processor.process_courses(raw_courses)
    
    # Save processed courses
    course_processor.save_processed_chunks(course_chunks, 'data/processed/course_detail_processed.json')
    logger.info(f"Saved {len(course_chunks)} processed course chunks")
    
    logger.info("Data regeneration completed successfully!")

if __name__ == "__main__":
    main()
