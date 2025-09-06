#!/usr/bin/env python3
"""
Clean Interactive Demo for the Multilingual RAG System
Simple interface that calls backend functions and manages chat history
"""

import os
import sys
import time
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.rag import RAGSystem
from src.utils.config import config

class RAG:
    def __init__(self):
        self.rag_system = None
        self.chat_history = []
        self.is_initialized = False
        
    def print_banner(self):
        """Print the demo banner"""
        print("="*40)
        print("               RAG SYSTEM")
        print("="*40)
        
    def initialize_system(self):
        """Initialize the RAG system"""
        print("Initializing RAG System...")
        try:
            # Try to initialize with all features first
            try:
                self.rag_system = RAGSystem(
                    use_reranker=True,
                    use_query_enhancement=True
                )
                print("RAG system initialized with all features!")
            except Exception as e:
                print(f"Failed to initialize with all features: {e}")
                print("Trying with minimal features...")
                
                # Fallback to minimal configuration
                self.rag_system = RAGSystem(
                    use_reranker=False,
                    use_query_enhancement=False
                )
                print("RAG system initialized with minimal features!")
            
            # Load data
            print("Loading course data...")
            data_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "course_detail.json")
            if not os.path.exists(data_file):
                print(f"Data file not found: {data_file}")
                return False
                
            if not self.rag_system.load_and_process_data(data_file):
                print("Failed to load course data!")
                return False
                
            print("Course data loaded successfully!")
            
            # Build vector index
            print("Building vector index...")
            if not self.rag_system.build_vector_index():
                print("Failed to build vector index!")
                return False
                
            print("Vector index built successfully!")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize system: {e}")
            return False
    
    def add_to_chat_history(self, user_input: str, system_response: str):
        """Add interaction to chat history"""
        self.chat_history.append({
            'timestamp': time.time(),
            'user_input': user_input,
            'system_response': system_response
        })
    
    def display_chat_history(self):
        """Display recent chat history"""
        if not self.chat_history:
            print("No chat history yet.")
            return
            
        print(f"\nRecent Chat History ({len(self.chat_history)} interactions):")
        print("=" * 80)
        
        for i, chat in enumerate(self.chat_history[-5:], 1):  # Show last 5 interactions
            timestamp = time.strftime("%H:%M:%S", time.localtime(chat['timestamp']))
            print(f"\n{i}. [{timestamp}]")
            print(f"   You: {chat['user_input']}")
            print(f"   RAG: {chat['system_response'][:100]}{'...' if len(chat['system_response']) > 100 else ''}")
            print("-" * 80)
    
    def process_user_input(self, user_input: str) -> str:
        """Process user input by calling appropriate backend functions"""
        try:
            # Simple intent detection - just pass to backend for processing
            if not user_input.strip():
                return "Please type something to search or ask about courses!"
            
            # Check for quit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                return "QUIT_COMMAND"
            
            # Check for help commands
            if user_input.lower() in ['help', '?']:
                return self.get_help_message()
            
            # Check for history commands
            if user_input.lower() in ['history', 'chat']:
                self.display_chat_history()
                return "History displayed above."
            
            # Check for status commands
            if user_input.lower() in ['status', 'system']:
                return self.get_system_status()
            
            # Check for stats commands
            if user_input.lower() in ['stats', 'statistics']:
                return self.get_system_stats()
            
            # Check for performance commands
            if user_input.lower() in ['performance', 'perf', 'metrics']:
                return self.get_performance_summary()
            
            # Check for export performance command
            if user_input.lower().startswith('export perf'):
                return self.export_performance_data()
            
            # Check for clear conversation command
            if user_input.lower() in ['clear', 'reset', 'new']:
                self.rag_system.clear_conversation_context()
                return "Conversation context cleared. Starting fresh conversation."
            
            if user_input.lower() in ['clear cache', 'clear-cache', 'cache clear']:
                if self.rag_system.clear_embedding_cache():
                    return "Embedding cache cleared. Next run will regenerate embeddings."
                else:
                    return "Failed to clear embedding cache."
            
            # Default: treat as search query - let backend handle everything
            print("Processing your query...")
            
            # Use the new response generation capability
            if hasattr(self.rag_system, 'generate_response'):
                response = self.rag_system.generate_response(user_input, top_k=5)
                return response
            else:
                # Fallback to old search method
                results = self.rag_system.search(user_input, top_k=5)
                
                if not results:
                    return f"No courses found related to '{user_input}'"
                
                # Format results
                return self.format_search_results(results, user_input)
            
        except Exception as e:
            return f"Error processing request: {e}"
    
    def get_help_message(self) -> str:
        """Get help message"""
        return """Available Commands:
                • Just ask naturally about courses in any language
                • 'help' - Show this help message
                • 'history' - Show chat history
                • 'status' - Show system status
                • 'stats' - Show system statistics
                • 'performance' - Show performance metrics
                • 'export perf' - Export performance data to file
                • 'clear' - Clear conversation context and start fresh
                • 'clear cache' - Clear embedding cache (forces regeneration)
                • 'quit' - Exit the demo

                Examples:
                • 'Find AI courses'
                • 'หาคอร์สเกี่ยวกับ cybersecurity'
                • 'Show me hardware design courses'
                • 'รายละเอียดวิชา 01076532'
                • 'Can you summarize the course details?'"""
    
    def get_system_status(self) -> str:
        """Get system status"""
        try:
            status = self.rag_system.get_system_status()
            response = f"System Status:\n"
            response += f"Total Chunks: {status['total_chunks']}\n"
            response += f"Vector Store: {status['vector_store_type']}\n"
            response += f"Vector Count: {status['vector_store_count']}\n"
            response += f"Reranker: {'Enabled' if status['reranker_enabled'] else 'Disabled'}\n"
            response += f"Query Enhancement: {'Enabled' if status['query_enhancement_enabled'] else 'Disabled'}\n"
            response += f"Response Generation: {'Enabled' if status.get('response_generation_enabled', False) else 'Disabled'}"
            
            # Add conversation context info
            if status.get('conversation_context'):
                response += f"\n\nConversation Context:\n{status['conversation_context']}"
            
            return response
        except Exception as e:
            return f"Error getting system status: {e}"
    
    def get_system_stats(self) -> str:
        """Get system statistics"""
        try:
            status = self.rag_system.get_system_status()
            stats = status.get('statistics', {})
            response = f"System Statistics:\n"
            response += f"Total Chunks: {status['total_chunks']}\n"
            response += f"Vector Store: {status['vector_store_type']}\n"
            response += f"Vector Count: {status['vector_store_count']}\n"
            
            if stats:
                response += f"\nLanguage Distribution:\n"
                for lang, count in stats.get('languages', {}).items():
                    response += f"   {lang.upper()}: {count}\n"
                
                response += f"\nFocus Areas:\n"
                for area, count in stats.get('focus_areas', {}).items():
                    response += f"   {area}: {count}\n"
                
                response += f"\nCareer Tracks:\n"
                for track, count in stats.get('career_tracks', {}).items():
                    response += f"   {track}: {count}\n"
            
            return response
        except Exception as e:
            return f"Error getting system statistics: {e}"
    
    def get_performance_summary(self) -> str:
        """Get performance summary"""
        try:
            return self.rag_system.get_performance_summary()
        except Exception as e:
            return f"Error getting performance summary: {e}"
    
    def export_performance_data(self) -> str:
        """Export performance data to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
            filepath = os.path.join(os.path.dirname(__file__), "..", "cache", filename)
            
            # Ensure cache directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            self.rag_system.export_performance_data(filepath)
            return f"Performance data exported to: {filepath}"
        except Exception as e:
            return f"Error exporting performance data: {e}"
    
    def format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results for display"""
        try:
            response = f"Found {len(results)} courses related to '{query}':\n\n"
            
            for i, result in enumerate(results[:3], 1):
                metadata = result['metadata']
                response += f"{i}. Course: {metadata.get('course_name', 'N/A')}\n"
                response += f"   Code: {metadata.get('course_code', 'N/A')}\n"
                response += f"   Language: {metadata.get('language', 'N/A')}\n"
                response += f"   Focus Areas: {', '.join(metadata.get('focus_areas', []))}\n"
                response += f"   Career Tracks: {', '.join(metadata.get('career_tracks', []))}\n"
                response += f"   Similarity: {result.get('similarity_score', 0):.4f}\n"
                
                if 'rerank_score' in result:
                    response += f"   Rerank: {result.get('rerank_score', 0):.4f}\n"
                
                response += "\n"
            
            if len(results) > 3:
                response += f"... and {len(results) - 3} more courses"
            
            return response
            
        except Exception as e:
            return f"Error formatting results: {e}"
    
    def run(self):
        """Run the clean interactive demo"""
        self.print_banner()
        
        # Initialize system
        if not self.initialize_system():
            print("Failed to initialize system. Exiting.")
            return
        
        print("RAG system ready! Just ask naturally in any language.")
        print("Type 'help' for commands, 'quit' to exit, or ask anything about courses!")
        print("Use 'clear' to reset conversation context and start fresh.")
        print()
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Process user input
                print("RAG System: ", end="", flush=True)
                response = self.process_user_input(user_input)
                
                # Check for quit command
                if response == "QUIT_COMMAND":
                    print("Goodbye! Thanks for using the RAG system!")
                    break
                
                # Display response
                print(response)
                
                # Add to chat history (unless it's a system command response)
                if not response.startswith("History displayed") and not response.startswith("Available Commands") and not response.startswith("Conversation context cleared"):
                    self.add_to_chat_history(user_input, response)
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for using the RAG system!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again or type 'quit' to exit")

def main():
    """Main function"""
    demo = RAG()
    demo.run()

if __name__ == "__main__":
    main()
