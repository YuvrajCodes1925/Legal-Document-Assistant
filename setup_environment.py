# setup_environment.py - Environment Setup for Legal Document Assist

# Download spaCy model
# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import warnings
warnings.filterwarnings('ignore')

print("Environment setup completed successfully!")

# CUDA Setup and GPU Configuration
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import subprocess
import os

class GPUSetup:
    def __init__(self):
        self.device = None
        self.setup_cuda()
    
    def check_gpu_availability(self):
        """Check GPU availability and specifications"""
        print("=== GPU Configuration ===")
        
        if torch.cuda.is_available():
            print(f"CUDA Available: {torch.cuda.is_available()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
        else:
            print("CUDA not available. Using CPU.")
            self.device = torch.device("cpu")
            
        return self.device
    
    def setup_cuda(self):
        """Setup CUDA environment variables"""
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Enable mixed precision training
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
    def test_model_loading(self):
        """Test loading a small transformer model on GPU"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased", device_map="auto")
            
            # Test inference
            inputs = tokenizer("Test GPU loading", return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            print("✓ GPU model loading test successful!")
            print(f"Output shape: {outputs.last_hidden_state.shape}")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ GPU model loading test failed: {e}")

# Initialize GPU setup
gpu_setup = GPUSetup()
device = gpu_setup.check_gpu_availability()
gpu_setup.test_model_loading()