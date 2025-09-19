#!/usr/bin/env python3
"""
Setup script to help users install RAGtester with the correct dependencies.
"""

import subprocess
import sys
import argparse
from typing import List


def run_command(command: List[str]) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Successfully ran: {' '.join(command)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {' '.join(command)}: {e}")
        print(f"Error output: {e.stderr}")
        return False


def install_core():
    """Install core dependencies."""
    print("📦 Installing core dependencies...")
    return run_command([sys.executable, "-m", "pip", "install", "-e", "."])


def install_provider(provider: str):
    """Install dependencies for a specific LLM provider."""
    provider_map = {
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.7.0"],
        "bedrock": ["boto3>=1.26.0"],
        "transformers": ["transformers>=4.30.0", "torch>=2.0.0"],
        "llama": ["llama-cpp-python>=0.2.0"],
    }
    
    if provider not in provider_map:
        print(f"❌ Unknown provider: {provider}")
        print(f"Available providers: {', '.join(provider_map.keys())}")
        return False
    
    print(f"📦 Installing dependencies for {provider} provider...")
    packages = provider_map[provider]
    return run_command([sys.executable, "-m", "pip", "install"] + packages)


def install_all():
    """Install all optional dependencies."""
    print("📦 Installing all optional dependencies...")
    return run_command([sys.executable, "-m", "pip", "install", "-e", ".[all]"])


def main():
    parser = argparse.ArgumentParser(description="Setup RAGtester dependencies")
    parser.add_argument(
        "--provider", 
        choices=["openai", "anthropic", "bedrock", "transformers", "llama"],
        help="Install dependencies for specific LLM provider"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Install all optional dependencies"
    )
    parser.add_argument(
        "--core-only", 
        action="store_true",
        help="Install only core dependencies"
    )
    
    args = parser.parse_args()
    
    print("🚀 RAGtester Dependency Setup")
    print("=" * 40)
    
    success = True
    
    if args.core_only:
        success = install_core()
    elif args.all:
        success = install_all()
    elif args.provider:
        success = install_core() and install_provider(args.provider)
    else:
        print("📋 Available options:")
        print("  --core-only     Install only core dependencies")
        print("  --provider X    Install core + specific provider dependencies")
        print("  --all          Install all optional dependencies")
        print()
        print("📦 Installing core dependencies by default...")
        success = install_core()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\n📚 Next steps:")
        print("1. Import the library: from ragtester import RAGTester")
        print("2. Check the README.md for usage examples")
        print("3. Run tests: python -m pytest tests/")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
