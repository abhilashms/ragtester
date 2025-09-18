from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class LocalLLM(LLMProvider):
    """
    Local LLM provider that supports various local model formats including GGUF.
    Uses llama-cpp-python for GGUF models and transformers for other formats.
    """
    
    def __init__(self, model: str = None, **kwargs: Any) -> None:
        """
        Initialize local LLM provider.
        
        Args:
            model: Path to the model file (e.g., .gguf file for llama-cpp-python)
            **kwargs: Additional parameters passed to the underlying model
        """
        self.model_path = model
        self.kwargs = kwargs
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the underlying model based on file extension."""
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError(f"Model file not found: {self.model_path}")
        
        # Check file extension to determine which library to use
        ext = os.path.splitext(self.model_path)[1].lower()
        
        if ext == '.gguf':
            self._init_gguf_model()
        else:
            # Fallback to transformers for other formats
            self._init_transformers_model()
    
    def _init_gguf_model(self) -> None:
        """Initialize GGUF model using llama-cpp-python with automatic CPU-only installation."""
        try:
            from llama_cpp import Llama

            # Default parameters for GGUF models
            default_params = {
                'model_path': self.model_path,
                'n_ctx': 2048,  # Context length
                'n_threads': 4,  # Number of threads
                'verbose': False,
            }

            # Merge with user-provided parameters
            params = {**default_params, **self.kwargs}
            self._model = Llama(**params)

        except ImportError:
            print("⚠️  llama-cpp-python not found. Installing CPU-only version...")
            try:
                import subprocess
                import sys
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "llama-cpp-python", "--force-reinstall", "--no-cache-dir"
                ], check=True)
                print("✅ Successfully installed CPU-only llama-cpp-python!")
                # Retry import and initialize
                from llama_cpp import Llama
                # Merge with user-provided parameters
                params = {**default_params, **self.kwargs}
                self._model = Llama(**params)
            except subprocess.CalledProcessError as e:
                raise ImportError(
                    f"Failed to auto-install llama-cpp-python: {e}\n\n"
                    "To install ragtester with llama support:\n"
                    "  pip install ragtester[llama]\n\n"
                    "Or install llama-cpp-python separately:\n"
                    "  pip install llama-cpp-python --force-reinstall --no-cache-dir\n\n"
                    "Alternatively, use a different LLM provider (OpenAI, Anthropic, AWS Bedrock)."
                )
        except FileNotFoundError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                print("⚠️  CUDA-related error detected. Installing CPU-only llama-cpp-python...")
                try:
                    import subprocess
                    import sys
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", 
                        "llama-cpp-python", "--force-reinstall", "--no-cache-dir"
                    ], check=True)
                    print("✅ Successfully installed CPU-only llama-cpp-python!")
                    # Retry loading the model
                    from llama_cpp import Llama
                    # Merge with user-provided parameters
                    params = {**default_params, **self.kwargs}
                    self._model = Llama(**params)
                except subprocess.CalledProcessError as install_error:
                    raise RuntimeError(
                        f"Failed to auto-install CPU-only llama-cpp-python: {install_error}\n\n"
                        "Manual solutions:\n"
                        "• Install CPU-only version: pip install llama-cpp-python --force-reinstall --no-cache-dir\n"
                        "• Or reinstall with llama support: pip install ragtester[llama] --force-reinstall\n"
                        "• Or use a different LLM provider (OpenAI, Anthropic, AWS Bedrock)\n\n"
                        f"Original error: {e}"
                    )
            else:
                raise RuntimeError(f"Failed to initialize GGUF model: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GGUF model: {e}")
    
    def _init_transformers_model(self) -> None:
        """Initialize model using transformers library."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                **self.kwargs
            )
            
        except ImportError:
            raise ImportError(
                "transformers and torch are required for non-GGUF models. "
                "Install with: pip install transformers torch"
            )
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using the local model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        # Convert messages to prompt format
        prompt = self._format_messages(messages)
        
        # Merge generation parameters
        gen_params = {
            'max_tokens': kwargs.get('max_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
        }
        
        if hasattr(self._model, 'create_completion'):
            # llama-cpp-python format
            response = self._model.create_completion(prompt, **gen_params)
            return response['choices'][0]['text'].strip()
        
        else:
            # transformers format
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=gen_params['max_tokens'],
                    temperature=gen_params['temperature'],
                    top_p=gen_params['top_p'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response.strip()
    
    def _format_messages(self, messages: Sequence[LLMMessage]) -> str:
        """Convert message sequence to prompt format."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
