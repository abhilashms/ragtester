#!/usr/bin/env python3
"""
Installation utilities for ragtester library.
Ensures CPU-only llama-cpp-python installation to avoid CUDA issues.
"""

import subprocess
import sys
import os
from typing import Tuple, Optional


def install_cpu_only_llama_cpp() -> Tuple[bool, Optional[str]]:
    """
    Install CPU-only version of llama-cpp-python to avoid CUDA issues.
    Returns (success, error_message)
    """
    try:
        print("🔧 Installing CPU-only llama-cpp-python to avoid CUDA issues...")
        
        # Uninstall any existing version first
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "uninstall", "llama-cpp-python", "-y"
            ], check=False, capture_output=True)
        except Exception:
            pass  # Ignore errors during uninstall
        
        # Install CPU-only version
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "llama-cpp-python", 
            "--force-reinstall", 
            "--no-cache-dir",
            "--no-deps"  # Don't install dependencies to avoid conflicts
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CPU-only llama-cpp-python installed successfully!")
            return True, None
        else:
            error_msg = f"Failed to install CPU-only llama-cpp-python: {result.stderr}"
            print(f"❌ {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Exception during llama-cpp-python installation: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg


def check_and_fix_llama_cpp() -> bool:
    """
    Check if llama-cpp-python is working and fix if needed.
    Returns True if working or successfully fixed.
    """
    try:
        from llama_cpp import Llama
        # Try to access a basic function to trigger library loading
        _ = Llama.get_model_type
        print("✅ llama-cpp-python is working correctly")
        return True
    except ImportError:
        print("⚠️  llama-cpp-python not installed, installing CPU-only version...")
        success, _ = install_cpu_only_llama_cpp()
        return success
    except FileNotFoundError as e:
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            print("⚠️  CUDA-related error detected, installing CPU-only version...")
            success, _ = install_cpu_only_llama_cpp()
            return success
        else:
            print(f"❌ llama-cpp-python error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected llama-cpp-python error: {e}")
        return False


def ensure_cpu_only_installation():
    """
    Ensure CPU-only llama-cpp-python is installed.
    This function should be called during ragtester initialization.
    """
    if not check_and_fix_llama_cpp():
        print("\n💡 Alternative solutions:")
        print("1. Use cloud-based LLM providers (OpenAI, Anthropic, AWS Bedrock)")
        print("2. Manually install: pip install llama-cpp-python --force-reinstall --no-cache-dir")
        print("3. Use a different Python environment")
        return False
    return True


if __name__ == "__main__":
    print("🚀 ragtester CPU-only llama-cpp-python installer")
    print("=" * 50)
    
    success = ensure_cpu_only_installation()
    
    if success:
        print("\n🎉 Installation completed successfully!")
        print("You can now use ragtester with local GGUF models without CUDA issues.")
    else:
        print("\n❌ Installation failed.")
        print("Consider using cloud-based LLM providers instead.")
        sys.exit(1)
