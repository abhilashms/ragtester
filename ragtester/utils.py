"""
Utility functions for ragtester library.
"""

import os
import sys
from typing import Optional


def check_llama_cpp_installation() -> tuple[bool, Optional[str]]:
    """
    Check if llama-cpp-python is properly installed and working.
    
    Returns:
        Tuple of (is_working, error_message)
    """
    try:
        import llama_cpp
        return True, None
    except ImportError:
        return False, "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
    except FileNotFoundError as e:
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            return False, (
                "CUDA-related error detected. This usually means:\n"
                "1. CUDA is not properly installed on your system\n"
                "2. The CUDA version doesn't match llama-cpp-python's requirements\n\n"
                "Solutions:\n"
                "• Install CPU-only version: pip install llama-cpp-python --force-reinstall --no-cache-dir\n"
                "• Or install with specific CUDA version: pip install llama-cpp-python[cuda] --force-reinstall --no-cache-dir\n"
                "• Or use a different LLM provider (OpenAI, Anthropic, etc.)\n\n"
                f"Original error: {e}"
            )
        else:
            return False, f"llama-cpp-python installation issue: {e}"
    except Exception as e:
        return False, f"Unexpected error with llama-cpp-python: {e}"


def suggest_llm_provider_alternatives() -> str:
    """
    Suggest alternative LLM providers when local models fail.
    
    Returns:
        String with suggestions for alternative providers
    """
    return """
Alternative LLM Providers:

1. OpenAI (Recommended for most users):
   ```python
   config = RAGTestConfig(
       llm=LLMConfig(
           provider="openai",
           model="gpt-3.5-turbo",
           api_key="your-openai-api-key"
       )
   )
   ```

2. Anthropic Claude:
   ```python
   config = RAGTestConfig(
       llm=LLMConfig(
           provider="anthropic",
           model="claude-3-haiku-20240307",
           api_key="your-anthropic-api-key"
       )
   )
   ```

3. AWS Bedrock:
   ```python
   config = RAGTestConfig(
       llm=LLMConfig(
           provider="bedrock",
           model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
           extra={"region": "us-east-1"}
       )
   )
   ```

4. Use a simple callable function (for testing):
   ```python
   def simple_rag_function(question: str) -> str:
       return f"Answer to '{question}': This is a test response."
   
   tester = RAGTester(rag_callable=simple_rag_function, config=config)
   ```
"""


def validate_model_path(model_path: str) -> tuple[bool, Optional[str]]:
    """
    Validate that a model path exists and is accessible.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_path:
        return False, "Model path is empty"
    
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"
    
    if not os.path.isfile(model_path):
        return False, f"Model path is not a file: {model_path}"
    
    # Check file extension
    ext = os.path.splitext(model_path)[1].lower()
    if ext not in ['.gguf', '.bin', '.safetensors', '.pt', '.pth']:
        return False, f"Unsupported model format: {ext}. Supported formats: .gguf, .bin, .safetensors, .pt, .pth"
    
    return True, None


def print_installation_help():
    """Print helpful installation and troubleshooting information."""
    print("🔧 ragtester Installation Help")
    print("=" * 50)
    
    # Check llama-cpp-python
    is_working, error = check_llama_cpp_installation()
    if is_working:
        print("✅ llama-cpp-python is working correctly")
    else:
        print("❌ llama-cpp-python issue detected:")
        print(error)
        print()
    
    print("📦 Installation Commands:")
    print("• Basic installation: pip install ragtester")
    print("• With PDF support: pip install ragtester[pdf]")
    print("• CPU-only llama-cpp: pip install llama-cpp-python --force-reinstall --no-cache-dir")
    print("• CUDA llama-cpp: pip install llama-cpp-python[cuda] --force-reinstall --no-cache-dir")
    print()
    
    print("🚀 Quick Start (without local models):")
    print("""
from ragtester import RAGTester, RAGTestConfig, LLMConfig
from ragtester.types import TestCategory

# Use OpenAI (no local model setup required)
config = RAGTestConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="your-api-key"
    )
)

def my_rag_function(question: str) -> str:
    return f"Answer to '{question}': This is a test response."

tester = RAGTester(rag_callable=my_rag_function, config=config)
tester.upload_documents(["your-document.pdf"])
results = tester.run_all_tests()
tester.print_summary(results)
""")
    
    print(suggest_llm_provider_alternatives())
