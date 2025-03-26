import os
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import numpy as np
from openai import OpenAI
import io
import pytesseract
from PIL import ImageEnhance
import threading
import warnings
import tempfile

logger = logging.getLogger(__name__)

def setup_tesseract():
    """Setup Tesseract OCR path and verify installation."""
    if os.name == 'nt':  # Windows
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            try:
                version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract OCR {version} initialized successfully")
                return True
            except Exception as e:
                logger.warning(f"Tesseract OCR found but failed to initialize: {str(e)}")
        else:
            logger.warning("Tesseract OCR not found in default location")
    return False

# Initialize Tesseract
TESSERACT_AVAILABLE = setup_tesseract()

class OpenAIClientSingleton:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        cls._instance = OpenAI()
                        logger.info("OpenAI client initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                        cls._instance = None
        return cls._instance

class ImageProcessor:
    """Processes images using OpenAI's CLIP for embeddings, BLIP for captions, and Tesseract for OCR."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the image processor with OpenAI's CLIP, BLIP models, and Tesseract OCR.
        
        Args:
            device: The device to use for inference ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = None
        self.clip_preprocess = None
        self.blip_processor = None
        self.blip_model = None
        self.initialized = False
        self.tesseract_available = TESSERACT_AVAILABLE
        
        # Get OpenAI client instance
        self.openai_client = OpenAIClientSingleton.get_instance()
        if not self.openai_client:
            logger.error("Failed to initialize OpenAI client")
            return
        
        self.initialized = True
        logger.info("Image processor initialized successfully")
    
    def _ensure_clip_loaded(self):
        """Ensure CLIP model is loaded."""
        if self.clip_model is None:
            try:
                import clip
                logger.info(f"Loading CLIP model on device: {self.device}")
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                logger.info("CLIP model loaded successfully")
            except ImportError as e:
                raise RuntimeError(f"CLIP not available: {str(e)}")
    
    def _ensure_blip_loaded(self):
        """Ensure BLIP model is loaded."""
        if self.blip_model is None:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                logger.info("Loading BLIP model")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(self.device)
                logger.info("BLIP model loaded successfully")
            except ImportError as e:
                raise RuntimeError(f"BLIP not available: {str(e)}")
    
    def is_initialized(self) -> bool:
        """Check if the image processor is properly initialized with required models."""
        return self.initialized
    
    def get_clip_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Returns the CLIP embedding for a PIL image, normalized.
        
        Args:
            image: PIL image to embed
            
        Returns:
            numpy.ndarray: Normalized embedding vector
        """
        if not self.initialized:
            raise RuntimeError("Image processor not initialized. Install required dependencies.")
        
        # Ensure CLIP is loaded
        self._ensure_clip_loaded()
        
        # Preprocess and get embedding
        image_preproc = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_preproc)
        
        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    
    def get_openai_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Returns the OpenAI embedding for a PIL image.
        
        Args:
            image: PIL image to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if not self.initialized or not self.openai_client:
            raise RuntimeError("Image processor not initialized or OpenAI client not available.")
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Get embedding from OpenAI
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=img_byte_arr,
            encoding_format="float"
        )
        
        return np.array(response.data[0].embedding)
    
    def get_image_caption(self, image: Image.Image) -> str:
        """
        Generate a caption for the image using BLIP.
        
        Args:
            image: PIL image to caption
            
        Returns:
            str: Generated caption
        """
        if not self.initialized:
            raise RuntimeError("Image processor not initialized. Install required dependencies.")
        
        # Ensure BLIP is loaded
        self._ensure_blip_loaded()
        
        # Generate caption
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        outputs = self.blip_model.generate(**inputs, max_length=50)
        caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Enhance image for better OCR results.
        
        Args:
            image: PIL image to enhance
            
        Returns:
            PIL.Image: Enhanced image
        """
        # Convert to grayscale
        image = image.convert('L')
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image: PIL image to process
            
        Returns:
            str: Extracted text
        """
        if not self.tesseract_available:
            logger.warning("Tesseract OCR is not available. Text extraction skipped.")
            return ""
            
        try:
            # Enhance image for better OCR
            enhanced_image = self.enhance_image_for_ocr(image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(enhanced_image)
            
            # Clean up the extracted text
            text = text.strip()
            
            return text if text else ""
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return ""
    
    def process_image(self, image_path: Union[str, Path], use_openai: bool = True) -> Dict[str, Any]:
        """
        Process an image file, extracting embeddings, BLIP caption, and OCR text.
        
        Args:
            image_path: Path to the image file
            use_openai: Whether to use OpenAI's embedding model (True) or CLIP (False)
            
        Returns:
            dict: Dictionary containing process results including:
                - embedding: The image embedding vector (from OpenAI or CLIP)
                - blip_caption: The BLIP-generated caption
                - ocr_text: The text extracted using OCR
                - filename: The filename
                - error: Error message if processing failed
        """
        if not self.initialized:
            return {"error": "Image processor not initialized. Install required dependencies."}
        
        image_path = Path(image_path)
        
        try:
            # Validate file
            if not image_path.exists():
                return {"error": f"File not found: {image_path}"}
            
            if not image_path.is_file():
                return {"error": f"Not a file: {image_path}"}
            
            # Get file info
            file_info = {
                "filename": image_path.name,
                "path": str(image_path),
                "size": image_path.stat().st_size,
                "extension": image_path.suffix.lower()
            }
            
            # Open and process image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                return {"error": f"Failed to open image: {str(e)}"}
            
            # Get embedding based on preference
            if use_openai:
                embedding = self.get_openai_image_embedding(image)
                embedding_type = "openai"
            else:
                embedding = self.get_clip_image_embedding(image)
                embedding_type = "clip"
            
            # Generate BLIP caption
            blip_caption = self.get_image_caption(image)
            
            # Extract text using OCR
            ocr_text = self.extract_text_from_image(image)
            
            # Return results
            return {
                **file_info,
                "embedding": embedding.tolist(),
                "embedding_type": embedding_type,
                "blip_caption": blip_caption,
                "ocr_text": ocr_text
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return {"error": f"Error processing image: {str(e)}"}
    
    def supported_extensions(self) -> List[str]:
        """Get list of supported image file extensions."""
        return [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"] 