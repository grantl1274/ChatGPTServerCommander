import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import logging
import pandas as pd
import io
import numpy as np
from tqdm import tqdm
import json
import time

# Import specific loaders instead of importing all to avoid the pwd module issue
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from .vector_store import VectorStore
from .config import logger, SUPPORTED_EXTENSIONS

class DocumentProcessor:
    """Processes documents and adds them to the vector store."""
    
    def __init__(self, vector_store: VectorStore, 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200,
                supported_extensions: List[str] = None):
        """Initialize the document processor.
        
        Args:
            vector_store: Vector store instance for document storage
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            supported_extensions: List of supported file extensions
        """
        if not vector_store:
            raise ValueError("Vector store is required")
            
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS
        
        # Initialize text splitter with robust settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # More granular separation
        )
        
        # Validate configuration
        if not self.supported_extensions:
            raise ValueError("No supported file extensions configured")
            
        logger.info(f"Document processor initialized with:")
        logger.info(f"- Chunk size: {chunk_size}")
        logger.info(f"- Chunk overlap: {chunk_overlap}")
        logger.info(f"- Supported extensions: {', '.join(self.supported_extensions)}")
    
    def _get_loader(self, file_path: Path):
        """Get the appropriate document loader based on file extension."""
        extension = file_path.suffix.lower()
        if extension == '.txt':
            return TextLoader(str(file_path))
        elif extension == '.md':
            return UnstructuredMarkdownLoader(str(file_path))
        elif extension == '.pdf':
            return PyPDFLoader(str(file_path))
        elif extension in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def _get_loader_for_file(self, file_path: Path):
        """Get the appropriate document loader for the file type."""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.pdf':
                return PyPDFLoader(str(file_path))
            elif suffix == '.docx':
                return UnstructuredWordDocumentLoader(str(file_path))
            elif suffix == '.txt':
                return TextLoader(str(file_path))
            elif suffix in ['.md', '.markdown']:
                return TextLoader(str(file_path))
            elif suffix in ['.html', '.htm']:
                return UnstructuredHTMLLoader(str(file_path))
            elif suffix == '.csv':
                return CSVLoader(str(file_path))
            elif suffix in ['.xlsx', '.xls', '.xlsm']:
                return UnstructuredExcelLoader(str(file_path))
            elif suffix == '.ifc':
                # IFC files need special handling to extract text content
                return UnstructuredFileLoader(str(file_path))
            elif suffix in ['.psd', '.ai', '.indd', '.eps', '.svg']:
                # Adobe formats need special handling to extract text content
                return UnstructuredFileLoader(str(file_path))
            elif suffix == '.pptx':
                return UnstructuredPowerPointLoader(str(file_path))
            else:
                logger.warning(f"No specific loader for {suffix}, falling back to UnstructuredFileLoader")
                return UnstructuredFileLoader(str(file_path))
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {str(e)}")
            return None
    
    def _process_tabular_data(self, file_path: Path) -> str:
        """Special handling for tabular data to preserve structure."""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                df = pd.read_csv(file_path)
            elif suffix in ['.xlsx', '.xls', '.xlsm']:
                df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
                if isinstance(df, dict):  # Multiple sheets
                    text_parts = []
                    for sheet_name, sheet_df in df.items():
                        text_parts.append(f"\nSheet: {sheet_name}\n")
                        text_parts.append(sheet_df.to_string())
                    return "\n".join(text_parts)
                else:  # Single sheet
                    df = df
            else:
                return None  # Not a tabular file
            
            # Convert DataFrame to a well-formatted string
            buffer = io.StringIO()
            df.to_string(buffer, index=True, max_rows=None, max_cols=None)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error processing tabular data {file_path}: {str(e)}")
            return None
    
    def process_file(self, file_path: str, progress_callback: Callable = None) -> dict:
        """Process a single file, extract content, and add to vector store.
        
        Args:
            file_path: Path to the file to process
            progress_callback: Optional callback function for progress updates
            
        Returns:
            dict: Processing results
        """
        file_path = Path(file_path)
        start_time = time.time()
        
        # Initialize progress reporting 
        def report_progress(step: int, total: int, stage: str, details: dict = None):
            if progress_callback:
                progress_callback(step, total, stage, details or {})
            
        try:
            # Validation stage (0-5%)
            report_progress(0, 100, "validating")
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not file_path.is_file():
                raise ValueError(f"Not a file: {file_path}")
            
            if not self._is_supported_file(file_path):
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # File info stage (5-10%)
            report_progress(5, 100, "reading_metadata")
            
            file_info = {
                "filename": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "type": file_path.suffix.lower()[1:],  # Remove leading dot
                "processing_started": datetime.now().isoformat()
            }
            
            # Loading stage (10-30%)
            report_progress(10, 100, "loading")
            
            # Handle tabular files specially
            if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls', '.xlsm']:
                text_content = self._process_tabular_data(file_path)
                if text_content:
                    document = [type('Document', (), {'page_content': text_content})]
                else:
                    loader = self._get_loader_for_file(file_path)
                    document = loader.load() if loader else []
            else:
                # Normal file loading
                loader = self._get_loader_for_file(file_path)
                document = loader.load() if loader else []
            
            # Early return if no content was loaded
            if not document:
                logger.info(f"No content extracted from {file_path.name}")
                return {
                    "file_info": file_info,
                    "chunks": [],
                    "success": False,
                    "error": "No content could be extracted"
                }
            
            # Extract text content
            report_progress(30, 100, "extracting_text")
            
            # Get the raw text from the document
            text = ""
            if hasattr(document, 'page_content'):  # Single document
                text = document.page_content
            elif isinstance(document, list):  # List of documents (e.g., PDF pages)
                if document and hasattr(document[0], 'page_content'):
                    text = "\n\n".join(doc.page_content for doc in document if hasattr(doc, 'page_content'))
                else:
                    logger.warning(f"Unexpected document format from {file_path.name}")
            
            logger.info(f"Chunking document: {file_path.name} ({len(text) if isinstance(text, str) else 0} characters)")
            
            # Chunking stage (30-60%)
            report_progress(30, 100, "chunking")
            
            # Split the text into chunks
            if text:
                # Ensure text is a string before chunking
                if not isinstance(text, str):
                    logger.warning(f"Expected string for chunking, got {type(text)} from {file_path.name}")
                    text = str(text) if text is not None else ""
                
                try:
                    chunks = self.text_splitter.split_text(text)
                    logger.info(f"Created {len(chunks) if isinstance(chunks, list) else 0} chunks from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error splitting text for {file_path.name}: {str(e)}")
                    # Ensure we have a valid chunks variable
                    chunks = []
            else:
                chunks = []
                logger.warning(f"No text content to chunk from {file_path.name}")
            
            # Ensure chunks is always a list
            if not isinstance(chunks, list):
                logger.warning(f"Expected list for chunks, got {type(chunks)} from {file_path.name}")
                if chunks is None:
                    chunks = []
                elif isinstance(chunks, int):
                    # Handle integer case which is causing the error
                    logger.warning(f"Got integer value {chunks} for chunks, converting to string in list")
                    chunks = [str(chunks)]
                else:
                    # Convert any other non-list type to a list with a single item
                    chunks = [str(chunks)]
            
            # Adding to vector store stage (60-95%)
            report_progress(60, 100, "adding_to_index")
            
            # Add the chunks to the vector store
            chunk_metadata = []
            if chunks:
                chunk_ids = []
                for i, chunk in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    metadata = {
                        "chunk_id": chunk_id,
                        "file_id": file_info["filename"],
                        "file_name": file_path.name,
                        "file_type": file_info["type"],
                        "chunk_index": i,
                        "chunk_size": len(chunk) if isinstance(chunk, str) else 0,
                        "total_chunks": len(chunks),
                        "processed_at": datetime.now().isoformat()
                    }
                    chunk_ids.append(chunk_id)
                    chunk_metadata.append(metadata)
                
                if chunk_ids:
                    try:
                        self.vector_store.add_texts(chunks, chunk_metadata)
                    except Exception as e:
                        logger.error(f"Error adding chunks to vector store: {str(e)}")
                        # If the error is severe, re-raise it
                        if "openai" in str(e).lower():
                            raise
            
            # Final processing stage (95-100%)
            report_progress(95, 100, "finalizing")
            
            # Set final processed status
            file_info["processing_completed"] = datetime.now().isoformat()
            file_info["processing_time_seconds"] = time.time() - start_time
            
            # Return processing results
            return {
                "file_info": file_info,
                "chunks": chunks if isinstance(chunks, list) else [],
                "chunk_count": len(chunks) if isinstance(chunks, list) else 0, 
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            # Return error information
            return {
                "filename": file_path.name if file_path else "unknown",
                "error": str(e),
                "success": False
            }
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if the file type is supported."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def process_directory(self, directory_path: str) -> dict:
        """Process all supported files in the directory."""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Directory does not exist: {directory}")
            return {"error": f"Directory not found: {directory}"}
        
        results = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "total_chunks": 0,
            "errors": [],
            "files": []
        }
        
        # Scan directory for supported files
        files_to_process = []
        for file_path in directory.glob("**/*"):
            if file_path.is_file() and self._is_supported_file(file_path):
                files_to_process.append(file_path)
        
        total_files = len(files_to_process)
        logger.info(f"Found {total_files} files to process in {directory}")
        
        # Process files in batches for better memory management
        batch_size = 10  # Adjust based on typical file sizes
        for batch_idx in range(0, total_files, batch_size):
            batch_files = files_to_process[batch_idx:batch_idx + batch_size]
            logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            
            # Process files in the batch
            for i, file_path in enumerate(batch_files):
                logger.info(f"Processing file {batch_idx + i + 1}/{total_files}: {file_path.name}")
                try:
                    file_result = self.process_file(str(file_path))
                    
                    if "error" in file_result:
                        results["failed"] += 1
                        results["errors"].append({
                            "file": str(file_path),
                            "error": file_result["error"]
                        })
                        logger.error(f"Failed to process {file_path}: {file_result['error']}")
                    else:
                        results["processed"] += 1
                        results["total_chunks"] += file_result.get("chunk_count", 0)
                        results["files"].append({
                            "file": str(file_path),
                            "chunk_count": file_result.get("chunk_count", 0)
                        })
                        logger.info(f"Successfully processed {file_path}: {file_result.get('chunk_count', 0)} chunks")
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "file": str(file_path),
                        "error": str(e)
                    })
                    logger.error(f"Exception processing {file_path}: {str(e)}")
            
            # After each batch, save the vector store to ensure progress is not lost
            try:
                self.vector_store._save()
                logger.info(f"Saved progress after batch {batch_idx//batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to save progress after batch: {str(e)}")
        
        # Final save after all files are processed
        try:
            self.vector_store._save()
            logger.info("Final save of vector store completed")
        except Exception as e:
            logger.error(f"Failed to save vector store after processing: {str(e)}")
        
        # Log summary of processing
        logger.info(f"Processed {results['processed']} files, {results['failed']} failed, "
                    f"{results['skipped']} skipped, {results['total_chunks']} total chunks")
        
        return results 

    def process_pdf(self, file_path: Path) -> List[str]:
        """Process a PDF file and return its text content."""
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            if not pages:
                logger.warning(f"No pages found in PDF: {file_path}")
                return []
            
            text_content = []
            for page in pages:
                if isinstance(page.page_content, (str, bytes)):
                    text_content.append(page.page_content)
                elif hasattr(page.page_content, '__str__'):
                    text_content.append(str(page.page_content))
                else:
                    logger.warning(f"Skipping unprocessable page content in {file_path}")
                    continue
                
            return text_content
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return [] 