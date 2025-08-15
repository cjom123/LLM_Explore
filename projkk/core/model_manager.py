"""
Model management module for loading and managing Hugging Face models.
Handles embedding models and language models for the RAG system.
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
from config.settings import (
    EMBEDDING_MODEL, 
    LANGUAGE_MODEL, 
    MODEL_CACHE_DIR,
    USE_GPU,
    BATCH_SIZE
)

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and caching of Hugging Face models."""
    
    def __init__(self):
        self.embedding_model = None
        self.language_model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.model_cache_dir = MODEL_CACHE_DIR
        
        # Create cache directory
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_device(self) -> str:
        """Determine the best available device for model inference."""
        if USE_GPU and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            logger.info("Using Apple Silicon MPS")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        
        return device
    
    def load_embedding_model(self, model_name: str = None) -> SentenceTransformer:
        """Load the embedding model for text vectorization."""
        try:
            if model_name is None:
                model_name = EMBEDDING_MODEL
            
            logger.info(f"Loading embedding model: {model_name}")
            
            # Check if model is already loaded
            if (self.embedding_model is not None and 
                hasattr(self.embedding_model, 'model_name') and 
                self.embedding_model.model_name == model_name):
                logger.info("Embedding model already loaded")
                return self.embedding_model
            
            # Load model with caching
            self.embedding_model = SentenceTransformer(
                model_name,
                cache_folder=str(self.model_cache_dir),
                device=self.device
            )
            
            logger.info(f"Successfully loaded embedding model: {model_name}")
            return self.embedding_model
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def load_language_model(self, model_name: str = None) -> Any:
        """Load the language model for text generation."""
        try:
            if model_name is None:
                model_name = LANGUAGE_MODEL
            
            logger.info(f"Loading language model: {model_name}")
            
            # Check if model is already loaded
            if (self.language_model is not None and 
                hasattr(self.language_model, 'name_or_path') and 
                self.language_model.name_or_path == model_name):
                logger.info("Language model already loaded")
                return self.language_model
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.model_cache_dir),
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization for efficiency
            if self.device == "cuda" and torch.cuda.is_available():
                # Use 4-bit quantization for GPU
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False
                )
                
                self.language_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.model_cache_dir),
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Load on CPU or MPS without quantization
                self.language_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.model_cache_dir),
                    device_map=self.device,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "mps" else torch.float32
                )
            
            logger.info(f"Successfully loaded language model: {model_name}")
            return self.language_model
            
        except Exception as e:
            logger.error(f"Error loading language model: {str(e)}")
            raise
    
    def create_text_generation_pipeline(self, model_name: str = None) -> Any:
        """Create a text generation pipeline for easier inference."""
        try:
            if model_name is None:
                model_name = LANGUAGE_MODEL
            
            # Load model if not already loaded
            if self.language_model is None:
                self.load_language_model(model_name)
            
            # Create pipeline
            pipeline_config = {
                "model": self.language_model,
                "tokenizer": self.tokenizer,
                "device": self.device,
                "max_length": 512,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Adjust for different device types
            if self.device == "cuda":
                pipeline_config["device_map"] = "auto"
            elif self.device == "mps":
                pipeline_config["torch_dtype"] = torch.float16
            
            text_pipeline = pipeline("text-generation", **pipeline_config)
            
            logger.info("Successfully created text generation pipeline")
            return text_pipeline
            
        except Exception as e:
            logger.error(f"Error creating text generation pipeline: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the loaded model."""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        return self.embedding_model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, texts: list, batch_size: int = None) -> torch.Tensor:
        """Generate embeddings for a list of texts."""
        try:
            if self.embedding_model is None:
                self.load_embedding_model()
            
            if batch_size is None:
                batch_size = BATCH_SIZE
            
            logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
            
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.device
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """Generate text using the loaded language model."""
        try:
            if self.language_model is None:
                self.load_language_model()
            
            # Prepare input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device != "cpu":
                inputs = inputs.to(self.device)
            
            # Set default generation parameters
            generation_config = {
                "max_length": max_length,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            # Generate text
            with torch.no_grad():
                outputs = self.language_model.generate(
                    inputs,
                    **generation_config
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            logger.info(f"Generated text with length: {len(generated_text)}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "device": self.device,
            "embedding_model": None,
            "language_model": None,
            "tokenizer": None
        }
        
        if self.embedding_model is not None:
            info["embedding_model"] = {
                "name": getattr(self.embedding_model, 'model_name', 'Unknown'),
                "dimension": self.get_embedding_dimension()
            }
        
        if self.language_model is not None:
            info["language_model"] = {
                "name": getattr(self.language_model, 'name_or_path', 'Unknown'),
                "type": type(self.language_model).__name__
            }
        
        if self.tokenizer is not None:
            info["tokenizer"] = {
                "name": getattr(self.tokenizer, 'name_or_path', 'Unknown'),
                "vocab_size": self.tokenizer.vocab_size
            }
        
        return info
    
    def unload_models(self):
        """Unload models to free memory."""
        try:
            if self.embedding_model is not None:
                del self.embedding_model
                self.embedding_model = None
            
            if self.language_model is not None:
                del self.language_model
                self.language_model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Models unloaded and memory freed")
            
        except Exception as e:
            logger.error(f"Error unloading models: {str(e)}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload_models() 