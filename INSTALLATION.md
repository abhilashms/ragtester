# RAGtester Installation Guide

## Quick Start

### Basic Installation (Core Features Only)
```bash
pip install ragtester
```

### With LLM Provider Support
```bash
# For OpenAI
pip install ragtester[openai]

# For Anthropic
pip install ragtester[anthropic]

# For AWS Bedrock
pip install ragtester[bedrock]

# For Local Models (Transformers)
pip install ragtester[local-transformers]

# For Local Models (Llama.cpp)
pip install ragtester[local-llama]
```

### Complete Installation (All Features)
```bash
pip install ragtester[all]
```

## Detailed Installation Options

### 1. Core Dependencies
The following packages are always installed with RAGtester:
- `chromadb>=0.4.0` - Vector database for document storage
- `sentence-transformers>=2.2.0` - Text embeddings
- `PyPDF2>=3.0.0` - PDF document processing
- `python-docx>=0.8.11` - Word document processing
- `tqdm>=4.64.0` - Progress bars
- `python-json-logger>=2.0.0` - Structured logging
- `requests>=2.28.0` - HTTP client

### 2. LLM Provider Dependencies

#### OpenAI Provider
```bash
pip install openai>=1.0.0
```
Required for using OpenAI models (GPT-3.5, GPT-4, etc.)

#### Anthropic Provider
```bash
pip install anthropic>=0.7.0
```
Required for using Anthropic models (Claude)

#### AWS Bedrock Provider
```bash
pip install boto3>=1.26.0
```
Required for using AWS Bedrock models

#### Local Transformers Models
```bash
pip install transformers>=4.30.0 torch>=2.0.0
```
Required for running local transformer models

#### Local Llama.cpp Models
```bash
pip install llama-cpp-python>=0.2.0
```
Required for running GGUF model files

### 3. Optional Data Processing
```bash
pip install numpy>=1.24.0 pandas>=2.0.0
```
For advanced data analysis features

### 4. Development Dependencies
```bash
pip install pytest>=7.0.0 black>=23.0.0 flake8>=6.0.0 mypy>=1.0.0
```
For development and testing

## Installation from Source

### Clone and Install
```bash
git clone https://github.com/abhilashms230/ragtester.git
cd ragtester
pip install -e .
```

### With Optional Dependencies
```bash
# Install with specific provider
pip install -e ".[openai]"

# Install with all optional dependencies
pip install -e ".[all]"

# Install development dependencies
pip install -e ".[dev]"
```

## Using the Setup Script

The library includes a setup script to help with dependency installation:

```bash
# Install core dependencies
python setup_dependencies.py --core-only

# Install with specific provider
python setup_dependencies.py --provider openai

# Install all optional dependencies
python setup_dependencies.py --all
```

## Verification

After installation, verify that RAGtester is working correctly:

```python
from ragtester import RAGTester, RAGTestConfig, LLMConfig

# Test basic import
print("✅ RAGtester imported successfully!")

# Test configuration
config = RAGTestConfig(
    llm=LLMConfig(provider="dummy", model="test")
)
print("✅ Configuration created successfully!")
```

## Troubleshooting

### Common Issues

1. **ImportError for optional dependencies**
   - Install the specific provider package: `pip install openai`
   - Or install all optional dependencies: `pip install ragtester[all]`

2. **CUDA/GPU issues with local models**
   - For CPU-only installation: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
   - For GPU support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

3. **PDF processing issues**
   - Ensure PyPDF2 is installed: `pip install PyPDF2>=3.0.0`

4. **Document processing issues**
   - Ensure python-docx is installed: `pip install python-docx>=0.8.11`

### Platform-Specific Notes

#### Windows
- May require Visual C++ Build Tools for some packages
- Use `pip install --only-binary=all` if compilation issues occur

#### macOS
- May require Xcode Command Line Tools
- Use `pip install --only-binary=all` if compilation issues occur

#### Linux
- Ensure development tools are installed: `sudo apt-get install build-essential`
- For GPU support, install CUDA toolkit

## Environment Variables

Set these environment variables for API providers:

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

## Next Steps

1. **Read the README.md** for usage examples
2. **Check the examples/** directory for sample code
3. **Run tests** to verify installation: `python -m pytest tests/`
4. **Start testing your RAG system** with the quick start guide
