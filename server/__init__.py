"""
Server package for the CE RAG System.
Provides FastAPI-based web interface with Swagger documentation.
"""

from .app import app

__all__ = ['app']
