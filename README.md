# RAGtester

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/ragtester.svg)](https://pypi.org/project/ragtester/)

> **A Python library for automated testing and evaluation of Retrieval-Augmented Generation (RAG) systems using large language models (LLMs).**

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🚀 Key Features](#-key-features)
- [📦 Installation](#-installation)
- [🎯 Quick Start](#-quick-start)
- [📊 Output Formats](#-output-formats)
- [🌐 Provider Setup](#-provider-setup)
- [📁 Project Structure](#-project-structure)
- [🔧 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)
- [🆘 Support](#-support)

## 🎯 Overview

RAGtester is like a test suite for your RAG’s answers. You give it your documents (PDFs, text files) and your RAG function (or API). It will then generate questions from your docs, ask your RAG system those questions, check the answers for things like accuracy, clarity, and safety, and create reports (CSV, JSON, Markdown). This saves you from writing manual tests or checking answers by hand.

### Why RAGtester?

- **🔍 Comprehensive Evaluation**: 5-dimensional assessment covering faithfulness, quality, toxicity, robustness, and security
- **🤖 LLM-Powered**: Uses advanced language models for realistic user question generation and evaluation
- **📊 Rich Reporting**: Detailed CSV, JSON, and Markdown reports with actionable insights
- **⚡ Easy Integration**: Simple API that works with any RAG system

## 🚀 Key Features

### 📊 **5-Dimensional Evaluation System**

| Dimension | Description | What It Tests |
|-----------|-------------|---------------|
| **Faithfulness** | How well responses match provided context | Factual accuracy, hallucination detection |
| **Answer Quality** | Fluency, clarity, and conciseness | Response coherence, completeness |
| **Toxicity** | Detection of harmful content | Safety, appropriateness, bias |
| **Robustness & Reliability** | System behavior under stress | Error handling, edge cases |
| **Security & Safety** | Resistance to malicious inputs | Prompt injection, data protection |

### 🎯 **Smart Question Generation**
- **Realistic**: Questions reflect real-world scenarios
- **Context-Aware**: Questions tailored to specific document content
- **Random Page Selection**: Each question uses different document pages
- **Metric-Specific**: Questions designed for each evaluation dimension
- **Behavior Testing**: General questions to test system behavior

### 🤖 **Multiple LLM Support**

| Provider |
|----------|
| **OpenAI** |
| **Anthropic** |
| **AWS Bedrock** |
| **Grok (xAI)** |
| **Google Gemini** |
| **Mistral AI** |
| **Cohere** |
| **Hugging Face** |
| **Fireworks AI** |
| **Together AI** |
| **Perplexity** |
| **Local** |

### 📁 **Document Support**

- **PDF Files**: Automatic text extraction and page selection
- **Text Files**: Direct processing with encoding detection
- **Markdown Files**: Full support with formatting preservation
- **Extensible**: Easy to add new document types

## 📦 Installation

### Basic Installation

```bash
pip install ragtester
```


### With Optional Dependencies

```bash
# For specific LLM providers
pip install ragtester[openai]        # OpenAI API support
pip install ragtester[anthropic]     # Anthropic API support
pip install ragtester[bedrock]       # AWS Bedrock support
pip install ragtester[grok]          # Grok (xAI) API support
pip install ragtester[gemini]        # Google Gemini API support
pip install ragtester[mistral]       # Mistral AI API support
pip install ragtester[cohere]        # Cohere API support
pip install ragtester[huggingface]   # Hugging Face Inference API support
pip install ragtester[fireworks]     # Fireworks AI API support
pip install ragtester[together]      # Together AI API support
pip install ragtester[perplexity]    # Perplexity AI API support
pip install ragtester[deepseek]      # DeepSeek API support
pip install ragtester[reka]          # Reka AI API support
pip install ragtester[qwen]          # Qwen (Alibaba) API support
pip install ragtester[moonshot]      # Moonshot AI API support
pip install ragtester[zhipu]         # Zhipu AI API support
pip install ragtester[baidu]         # Baidu ERNIE API support
pip install ragtester[zeroone]       # 01.AI API support


# Local model support
pip install ragtester[local-llama] # llamma models
pip install ragtester[ollama]        # Ollama local models
pip install ragtester[local-transformers]  # Local transformers models

### From Source

```bash
git clone https://github.com/abhilashms230/ragtester.git
cd ragtester
pip install -e .
```

## 🎯 Quick Start

### 1. Basic RAG Evaluation

```python
from ragtester import RAGTester, RAGTestConfig, LLMConfig
from ragtester.config import GenerationPlan
from ragtester.types import TestCategory

def my_rag_function(question: str) -> str:
    """Your RAG system implementation"""
    # Your retrieval and generation logic here
    return "Generated answer based on documents"

# Configure the evaluation
config = RAGTestConfig(
    llm=LLMConfig(
        provider="openai",  # or "anthropic", "grok", "gemini", "mistral", "cohere", "huggingface", "fireworks", "together", "perplexity", "bedrock", "local"
        model="gpt-4o-mini",
        api_key="your-api-key",
        temperature=0.7, # configurable by user
        max_tokens=2048, # configurable by user
    ),
    generation=GenerationPlan(
        per_category={
            TestCategory.FAITHFULNESS: 10, # configurable by user
            TestCategory.ANSWER_QUALITY: 10, # configurable by user
            TestCategory.TOXICITY: 10, # configurable by user
            TestCategory.ROBUSTNESS_RELIABILITY: 10, # configurable by user
            TestCategory.SECURITY_SAFETY: 10, # configurable by user
        }
    )
)

# Create tester and run evaluation
tester = RAGTester(rag_callable=my_rag_function, config=config)
tester.upload_documents(["docs/manual.pdf", "docs/guide.txt"])
results = tester.run_all_tests()
# Export results to CSV
csv_path = "rag_test_results.csv"
print(f"\n💾 Exporting detailed results to: {csv_path}")
tester.export_results(results, csv_path)
# View results
tester.print_summary(results)

```

### 2. API-Based RAG Evaluation

```python
from ragtester import RAGTester, RAGTestConfig, LLMConfig

config = RAGTestConfig(
    llm=LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
)

tester = RAGTester(
    rag_api_url="https://your-rag-api.com/query",
    config=config
)

tester.upload_documents(["docs/knowledge_base.pdf"])
results = tester.run_all_tests()
tester.print_summary(results)
```

### 3. Local Model RAG 
```python
from ragtester import RAGTester, RAGTestConfig, LLMConfig
from ragtester.config import GenerationPlan
from ragtester.types import TestCategory

def my_rag_function(question: str) -> str:
    """Your RAG system implementation"""
    # Your retrieval and generation logic here
    return "Generated answer based on documents"

# Configure the evaluation
config = RAGTestConfig(
    llm = LLMConfig(
        provider="local",
        model="path/to/your/model.gguf",  # Replace with actual path
        temperature=0.7,
        max_tokens=2048, # configurable by user
        extra={
            "n_ctx": 4096 # configurable by user
        }
    ),
    generation=GenerationPlan(
        per_category={
            TestCategory.FAITHFULNESS: 5, # configurable by user
            TestCategory.ANSWER_QUALITY: 5, # configurable by user
            TestCategory.TOXICITY: 3, # configurable by user
            TestCategory.ROBUSTNESS_RELIABILITY: 3, # configurable by user
            TestCategory.SECURITY_SAFETY: 3, # configurable by user
        }
    )
)

# Create tester and run evaluation
tester = RAGTester(rag_callable=my_rag_function, config=config)
tester.upload_documents(["docs/manual.pdf", "docs/guide.txt"])
results = tester.run_all_tests()
# Export results to CSV
csv_path = "rag_test_results.csv"
print(f"\n💾 Exporting detailed results to: {csv_path}")
tester.export_results(results, csv_path)
# View results
tester.print_summary(results)

```


## 📊 Output Formats

### Console Summary
```
============================================================
RAG EVALUATION RESULTS
============================================================
Overall Score: 3.8/5.0

📊 Detailed Breakdown:
├── Faithfulness: 4.2/5.0 (5 questions)
├── Answer Quality: 3.6/5.0 (5 questions)  
├── Toxicity: 4.0/5.0 (3 questions)
├── Robustness & Reliability: 3.4/5.0 (3 questions)
└── Security & Safety: 3.8/5.0 (3 questions)

✅ 19/19 tests completed successfully
⏱️  Total evaluation time: 2m 34s
```

### CSV Export
```python
from ragtester.reporter import export_csv

# Export detailed results
export_csv(results, "rag_evaluation_results.csv")
```

**CSV Columns:**
- `category`: The metric being evaluated
- `question`: The generated question
- `rag_Answer`: Your RAG system's response
- `score`: Integer score (1-5)
- `reasoning`: Detailed evaluation reasoning
- `page_number`: Context

### JSON Export
```python
from ragtester.reporter import export_json

export_json(results, "results.json")
```

### Markdown Report
```python
from ragtester.reporter import export_markdown

export_markdown(results, "results.md")
```

## 🌐 Provider Setup

### Anthropic Claude Setup

1. **Install Dependencies**
   ```bash
   pip install anthropic
   ```

2. **Get API Key**
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create an account and get your API key

3. **Configure Environment**
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

### AWS Bedrock Setup

1. **Install Dependencies**
   ```bash
   pip install boto3
   ```

2. **Configure Credentials**
   ```bash
   # Environment variables
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```

3. **Enable Model Access**
   - Go to AWS Bedrock console
   - Navigate to "Model access"
   - Request access to desired models


### Local Model Setup

1. **Download GGUF Model**
   ```bash
   # Example: Download Vicuna 7B
   wget https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF/resolve/main/vicuna-7b-v1.5.Q4_K_M.gguf
   ```

2. **Configure Local Provider**
   ```python
   config = RAGTestConfig(
       llm=LLMConfig(
           provider="local",
           model="path/to/vicuna-7b-v1.5.Q4_K_M.gguf",
           temperature=0.7,
           max_tokens=2048,
           extra={
               "n_ctx": 4096,
               "n_gpu_layers": -1,  # Use GPU if available
           }
       )
   )
   ```

## 📁 Project Structure

```
ragtester/
├── __init__.py                 # Main package exports
├── config.py                  # Configuration classes
├── tester.py                  # Main RAGTester class
├── types.py                   # Data structures and enums
├── utils.py                   # Utility functions
├── document_loader/           # Document processing
│   ├── base.py               # Base document loader
│   ├── simple_loaders.py     # PDF, text, markdown loaders
│   └── random_page_loader.py # Advanced page selection
├── evaluator/                # Response evaluation
│   ├── base.py              # Base evaluator interface
│   └── metrics_judge.py     # LLM-based evaluation
├── llm/                     # LLM providers
│   ├── base.py             # Base LLM interface
│   ├── providers.py        # Provider factory
│   ├── providers_openai.py # OpenAI integration
│   ├── providers_anthropic.py # Anthropic integration
│   ├── providers_bedrock.py # AWS Bedrock integration
│   └── providers_local.py  # Local model support
├── question_generator/      # Question generation
│   ├── base.py            # Base generator interface
│   └── generators.py      # LLM-based generation
├── rag_client/            # RAG system clients
│   ├── base.py           # Base client interface
│   └── clients.py        # API and callable clients
└── reporter/             # Result reporting
    ├── base.py          # Base reporter interface
    └── reporter.py      # CSV, JSON, Markdown export
```

## 🔧 Troubleshooting

### Common Issues

#### AWS Bedrock Model Access Issues

#### CUDA/llama-cpp-python Issues
```bash
# Install CPU-only version (recommended)
pip install llama-cpp-python --force-reinstall --no-cache-dir

# For GPU support (if you have CUDA)
pip install llama-cpp-python[server] --force-reinstall --no-cache-dir
```

#### Import Errors
```bash
# Check Python version (3.9+ required)
python --version
```

#### API Key Issues
```python
# Set environment variables
import os
os.environ["OPENAI_API_KEY"] = "your-key"
os.environ["ANTHROPIC_API_KEY"] = "your-key"
os.environ["XAI_API_KEY"] = "your-key"
os.environ["GOOGLE_API_KEY"] = "your-key"
os.environ["MISTRAL_API_KEY"] = "your-key"
os.environ["COHERE_API_KEY"] = "your-key"
os.environ["HF_TOKEN"] = "your-key"
os.environ["FIREWORKS_API_KEY"] = "your-key"
os.environ["TOGETHER_API_KEY"] = "your-key"
os.environ["PERPLEXITY_API_KEY"] = "your-key"
os.environ["DEEPSEEK_API_KEY"] = "your-key"
os.environ["REKA_API_KEY"] = "your-key"
os.environ["QWEN_API_KEY"] = "your-key"
os.environ["MOONSHOT_API_KEY"] = "your-key"
os.environ["ZHIPU_API_KEY"] = "your-key"
os.environ["BAIDU_API_KEY"] = "your-key"
os.environ["ZEROONE_API_KEY"] = "your-key"

# Or pass directly
config = RAGTestConfig(
    llm=LLMConfig(
        provider="openai",  # or "anthropic", "grok", "gemini", "mistral", "cohere", "huggingface", "fireworks", "together", "perplexity", "deepseek", "reka", "qwen", "moonshot", "zhipu", "baidu", "zeroone"
        api_key="your-key"
    )
)
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **📧 Email**: abhilashms230@gmail.com
- **💬 Discussions**: [GitHub Discussions](https://github.com/abhilashms/ragtester/discussions)


## 🙏 Acknowledgments

- Built with ❤️ for the RAG community
- Inspired by the need for standardized RAG evaluation
- Thanks to all contributors and users

---

**Made with ❤️ by [ABHILASH M S](https://github.com/abhilashms230)**

*Star ⭐ this repository if you find it helpful!*
