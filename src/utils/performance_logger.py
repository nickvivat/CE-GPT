#!/usr/bin/env python3
"""
CSV Performance Logger for RAG System
Logs detailed timing information for each RAG step to CSV files
"""

import csv
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import threading
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RAGStepMetrics:
    """Metrics for individual RAG steps"""
    timestamp: str
    session_id: str
    query: str
    step_name: str
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class CSVPerformanceLogger:
    """CSV-based performance logger for RAG system steps"""
    
    def __init__(self, log_dir: str = "logs", max_file_size_mb: int = 100):
        self.log_dir = log_dir
        self.max_file_size_mb = max_file_size_mb
        self.lock = threading.Lock()
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # CSV file paths for different steps (without session_id)
        self.csv_files = {
            'query_enhancement': os.path.join(log_dir, 'query_enhancement_performance.csv'),
            'embedding_search': os.path.join(log_dir, 'embedding_search_performance.csv'),
            'reranking': os.path.join(log_dir, 'reranking_performance.csv'),
            'response_generation': os.path.join(log_dir, 'response_generation_performance.csv'),
            'overall': os.path.join(log_dir, 'overall_rag_performance.csv')
        }
        
        # Initialize CSV files with headers
        self._initialize_csv_files()
        
    def _initialize_csv_files(self):
        """Initialize CSV files with proper headers (without session_id)"""
        headers = {
            'query_enhancement': [
                'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                'original_query', 'enhanced_query', 'classification', 'language', 'model_name'
            ],
            'embedding_search': [
                'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                'top_k', 'results_count', 'language_filter', 'embedding_model', 'vector_store_type'
            ],
            'reranking': [
                'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                'input_count', 'output_count', 'reranker_model'
            ],
            'response_generation': [
                'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                'response_length', 'language', 'model_name', 'streaming', 'context_length'
            ],
            'overall': [
                'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                'total_steps', 'total_duration'
            ]
        }
        
        for step, filepath in self.csv_files.items():
            if not os.path.exists(filepath):
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers[step])
                logger.info(f"Initialized CSV file: {filepath}")
    
    def log_query_enhancement(self, query: str, duration: float, success: bool, 
                             error_message: str = None, original_query: str = None,
                             enhanced_query: str = None, classification: str = None,
                             language: str = None, model_name: str = None):
        """Log query enhancement step metrics"""
        self._write_to_csv('query_enhancement', {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'step_name': 'query_enhancement',
            'duration_seconds': duration,
            'success': success,
            'error_message': error_message,
            'original_query': original_query or query,
            'enhanced_query': enhanced_query,
            'classification': classification,
            'language': language,
            'model_name': model_name
        })
    
    def log_embedding_search(self, query: str, duration: float, success: bool,
                            error_message: str = None, top_k: int = None,
                            results_count: int = None, language_filter: str = None,
                            embedding_model: str = None, vector_store_type: str = None):
        """Log embedding search step metrics"""
        self._write_to_csv('embedding_search', {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'step_name': 'embedding_search',
            'duration_seconds': duration,
            'success': success,
            'error_message': error_message,
            'top_k': top_k,
            'results_count': results_count,
            'language_filter': language_filter,
            'embedding_model': embedding_model,
            'vector_store_type': vector_store_type
        })
    
    def log_reranking(self, query: str, duration: float, success: bool,
                     error_message: str = None, input_count: int = None,
                     output_count: int = None, reranker_model: str = None):
        """Log reranking step metrics"""
        self._write_to_csv('reranking', {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'step_name': 'reranking',
            'duration_seconds': duration,
            'success': success,
            'error_message': error_message,
            'input_count': input_count,
            'output_count': output_count,
            'reranker_model': reranker_model
        })
    
    def log_response_generation(self, query: str, duration: float, success: bool,
                              error_message: str = None, response_length: int = None,
                              language: str = None, model_name: str = None,
                              streaming: bool = False, context_length: int = None):
        """Log response generation step metrics"""
        self._write_to_csv('response_generation', {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'step_name': 'response_generation',
            'duration_seconds': duration,
            'success': success,
            'error_message': error_message,
            'response_length': response_length,
            'language': language,
            'model_name': model_name,
            'streaming': streaming,
            'context_length': context_length
        })
    
    def log_overall_rag(self, query: str, duration: float, success: bool,
                       error_message: str = None, total_steps: int = None,
                       total_duration: float = None):
        """Log overall RAG process metrics"""
        self._write_to_csv('overall', {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'step_name': 'overall_rag',
            'duration_seconds': duration,
            'success': success,
            'error_message': error_message,
            'total_steps': total_steps,
            'total_duration': total_duration
        })
    
    def _write_to_csv(self, step: str, data: Dict[str, Any]):
        """Write metrics to appropriate CSV file"""
        with self.lock:
            try:
                filepath = self.csv_files[step]
                
                # Check file size and rotate if necessary
                if os.path.exists(filepath):
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    if file_size_mb > self.max_file_size_mb:
                        self._rotate_csv_file(filepath)
                
                # Get headers for this step
                headers = {
                    'query_enhancement': [
                        'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                        'original_query', 'enhanced_query', 'classification', 'language', 'model_name'
                    ],
                    'embedding_search': [
                        'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                        'top_k', 'results_count', 'language_filter', 'embedding_model', 'vector_store_type'
                    ],
                    'reranking': [
                        'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                        'input_count', 'output_count', 'reranker_model'
                    ],
                    'response_generation': [
                        'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                        'response_length', 'language', 'model_name', 'streaming', 'context_length'
                    ],
                    'overall': [
                        'timestamp', 'query', 'step_name', 'duration_seconds', 'success', 'error_message',
                        'total_steps', 'total_duration'
                    ]
                }
                
                # Prepare row data in the correct order
                row_data = [data.get(header, '') for header in headers[step]]
                
                # Write to CSV
                with open(filepath, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
                
                logger.debug(f"Logged {step} metrics: {data['duration_seconds']:.4f}s")
                
            except Exception as e:
                logger.error(f"Error writing to CSV file {step}: {e}")
    
    def _rotate_csv_file(self, filepath: str):
        """Rotate CSV file when it gets too large"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(filepath)[0]
            rotated_file = f"{base_name}_{timestamp}.csv"
            
            os.rename(filepath, rotated_file)
            logger.info(f"Rotated CSV file: {filepath} -> {rotated_file}")
            
            # Reinitialize the file
            self._initialize_csv_files()
            
        except Exception as e:
            logger.error(f"Error rotating CSV file {filepath}: {e}")
    
    def get_performance_summary(self, step: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary from CSV logs"""
        try:
            if step and step not in self.csv_files:
                return {"error": f"Invalid step: {step}"}
            
            files_to_check = [self.csv_files[step]] if step else list(self.csv_files.values())
            
            summary = {}
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            
            for filepath in files_to_check:
                if not os.path.exists(filepath):
                    continue
                
                step_name = os.path.basename(filepath).replace('_performance.csv', '')
                step_summary = {
                    'total_operations': 0,
                    'successful_operations': 0,
                    'failed_operations': 0,
                    'avg_duration': 0.0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0,
                    'total_duration': 0.0
                }
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            # Check if within time range
                            row_time = datetime.fromisoformat(row['timestamp']).timestamp()
                            if row_time < cutoff_time:
                                continue
                            
                            step_summary['total_operations'] += 1
                            
                            duration = float(row['duration_seconds'])
                            step_summary['total_duration'] += duration
                            step_summary['min_duration'] = min(step_summary['min_duration'], duration)
                            step_summary['max_duration'] = max(step_summary['max_duration'], duration)
                            
                            if row['success'].lower() == 'true':
                                step_summary['successful_operations'] += 1
                            else:
                                step_summary['failed_operations'] += 1
                                
                        except (ValueError, KeyError) as e:
                            logger.warning(f"Error parsing row in {filepath}: {e}")
                            continue
                
                if step_summary['total_operations'] > 0:
                    step_summary['avg_duration'] = step_summary['total_duration'] / step_summary['total_operations']
                    step_summary['success_rate'] = (step_summary['successful_operations'] / step_summary['total_operations']) * 100
                else:
                    step_summary['success_rate'] = 0.0
                
                if step_summary['min_duration'] == float('inf'):
                    step_summary['min_duration'] = 0.0
                
                summary[step_name] = step_summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def export_data(self, output_file: str = None, hours: int = 24) -> str:
        """Export all data from all CSV log files"""
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.log_dir, f"rag_performance_export_{timestamp}.csv")
            
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            all_data = []
            
            # Collect data from all CSV files
            for step, filepath in self.csv_files.items():
                if not os.path.exists(filepath):
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            # Check if within time range
                            row_time = datetime.fromisoformat(row['timestamp']).timestamp()
                            if row_time >= cutoff_time:
                                all_data.append(row)
                        except (ValueError, KeyError) as e:
                            logger.warning(f"Error parsing row in {filepath}: {e}")
                            continue
            
            # Write combined data
            if all_data:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = all_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_data)
                
                logger.info(f"Exported performance data to {output_file}")
                return output_file
            else:
                logger.warning(f"No data found in the last {hours} hours")
                return ""
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return ""

# Global CSV logger instance
csv_logger = CSVPerformanceLogger()
