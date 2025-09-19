# RAGtester

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/ragtester.svg)](https://badge.fury.io/py/ragtester)
[![Downloads](https://pepy.tech/badge/ragtester)](https://pepy.tech/project/ragtester)

> **A comprehensive Python library for testing and evaluating Retrieval-Augmented Generation (RAG) systems with LLM-generated questions and automated evaluation metrics.**

## 🎯 Overview

RAGtester is a powerful evaluation framework designed to assess the quality, reliability, and safety of RAG systems through automated testing. It generates context-aware questions from your documents and evaluates responses across multiple dimensions using state-of-the-art LLM judges.

### Why RAGtester?

- **🔍 Comprehensive Evaluation**: 5-dimensional assessment covering faithfulness, quality, toxicity, robustness, and security
- **🤖 LLM-Powered**: Uses advanced language models for intelligent question generation and evaluation
- **🔄 Multi-Provider Support**: Works with OpenAI, Anthropic, AWS Bedrock, and local models
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

- **Context-Aware**: Questions tailored to specific document content
- **Random Page Selection**: Each question uses different document pages
- **Metric-Specific**: Questions designed for each evaluation dimension
- **Behavior Testing**: General questions to test system behavior

### 🤖 **Multiple LLM Support**

| Provider | Models | Use Case |
|----------|--------|----------|
| **OpenAI** | GPT-4, GPT-3.5, GPT-4o | High-quality evaluation |
| **Anthropic** | Claude 3.5, Claude 3.7 | Balanced performance |
| **AWS Bedrock** | Claude, Titan, Cohere | Enterprise deployment |
| **Local** | GGUF models (Llama, Vicuna) | Privacy, cost control |
| **Dummy** | Mock responses | Testing without costs |

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
pip install ragtester[local-transformers]  # Local transformers models
pip install ragtester[local-llama]   # Local llama.cpp models

# For data processing features
pip install ragtester[data]          # NumPy and Pandas support

# For development
pip install ragtester[dev]           # Testing and linting tools

# For all optional features
pip install ragtester[all]
```

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
        provider="openai",  # or "anthropic", "bedrock", "local"
        model="gpt-4o-mini",
        api_key="your-api-key",
        temperature=0.7,
        max_tokens=2048,
    ),
    generation=GenerationPlan(
        per_category={
            TestCategory.FAITHFULNESS: 5,
            TestCategory.ANSWER_QUALITY: 5,
            TestCategory.TOXICITY: 3,
            TestCategory.ROBUSTNESS_RELIABILITY: 3,
            TestCategory.SECURITY_SAFETY: 3,
        }
    )
)

# Create tester and run evaluation
tester = RAGTester(rag_callable=my_rag_function, config=config)
tester.upload_documents(["docs/manual.pdf", "docs/guide.txt"])
results = tester.run_all_tests()

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

## ⚙️ Configuration

### LLM Configuration Examples

#### OpenAI (Recommended for Best Results)
```python
config = RAGTestConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o-mini",  # Fast and cost-effective
        api_key="your-api-key",
        temperature=0.7,
        max_tokens=2048,
    )
)
```

#### Anthropic Claude
```python
config = RAGTestConfig(
    llm=LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key="your-api-key",
        temperature=0.7,
        max_tokens=2048,
    )
)
```

#### AWS Bedrock (Enterprise)
```python
config = RAGTestConfig(
    llm=LLMConfig(
        provider="bedrock",
        model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        temperature=0.7,
        max_tokens=2048,
        extra={"region": "us-east-1"}
    )
)
```

#### Local Models (Privacy-First)
```python
config = RAGTestConfig(
    llm=LLMConfig(
        provider="local",
        model="path/to/your/model.gguf",
        temperature=0.7,
        max_tokens=2048,
        extra={
            "n_ctx": 4096,      # Context length
            "n_threads": 8,     # CPU threads
            "n_gpu_layers": -1, # Use all GPU layers
        }
    )
)
```

### Custom Generation Plans

```python
from ragtester.config import GenerationPlan
from ragtester.types import TestCategory

# Custom question distribution
generation_plan = GenerationPlan(
    per_category={
        TestCategory.FAITHFULNESS: 10,        # More faithfulness tests
        TestCategory.ANSWER_QUALITY: 8,       # Quality assessment
        TestCategory.TOXICITY: 5,             # Safety checks
        TestCategory.ROBUSTNESS_RELIABILITY: 5, # Edge cases
        TestCategory.SECURITY_SAFETY: 5,      # Security tests
    }
)

config = RAGTestConfig(generation=generation_plan)
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
- `Evaluation_Metrics`: The metric being evaluated
- `Question`: The generated question
- `RAG_Answer`: Your RAG system's response
- `Context`: The document context used
- `Score`: Integer score (1-5)
- `Judgement_Analysis`: Detailed evaluation reasoning

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

## 🔧 Advanced Usage

### Custom Question Generator

```python
from ragtester.question_generator.base import QuestionGenerator
from ragtester.types import Question, TestCategory

class CustomQuestionGenerator(QuestionGenerator):
    def generate(self, document_paths, num_per_category, seed, num_pages=5):
        """Generate custom questions for your specific use case"""
        questions = []
        
        # Your custom question generation logic
        for category, count in num_per_category.items():
            for i in range(count):
                question = self._generate_category_question(category, document_paths)
                questions.append(question)
        
        return questions

# Use custom generator
tester = RAGTester(rag_callable=my_rag_function, config=config)
tester.question_generator = CustomQuestionGenerator(config)
```

### Custom Response Evaluator

```python
from ragtester.evaluator.base import ResponseEvaluator
from ragtester.types import Evaluation, RAGResponse

class CustomEvaluator(ResponseEvaluator):
    def evaluate(self, responses):
        """Custom evaluation logic"""
        evaluations = []
        
        for response in responses:
            # Your custom evaluation logic
            score = self._custom_scoring(response)
            evaluation = Evaluation(
                question=response.question,
                answer=response.answer,
                context=response.context,
                metric=response.metric,
                score=score,
                reasoning="Custom evaluation reasoning"
            )
            evaluations.append(evaluation)
        
        return evaluations

# Use custom evaluator
tester = RAGTester(rag_callable=my_rag_function, config=config)
tester.evaluator = CustomEvaluator(config)
```

### Batch Processing

```python
# Process multiple RAG systems
rag_systems = [
    ("system1", rag_function_1),
    ("system2", rag_function_2),
    ("system3", rag_function_3),
]

results = {}
for name, rag_func in rag_systems:
    tester = RAGTester(rag_callable=rag_func, config=config)
    tester.upload_documents(["docs/knowledge_base.pdf"])
    results[name] = tester.run_all_tests()
    print(f"Completed evaluation for {name}")

# Compare results
for name, result in results.items():
    print(f"{name}: {result.overall_score:.2f}")
```

## 🌐 Provider Setup

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

4. **Available Models**
   - `us.anthropic.claude-3-5-haiku-20241022-v1:0` (Fast & Cost-effective)
   - `us.anthropic.claude-3-5-sonnet-20241022-v1:0` (Balanced)
   - `us.anthropic.claude-3-5-opus-20241022-v1:0` (High quality)

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
               "n_threads": 8,
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

#### CUDA/llama-cpp-python Issues
```bash
# Install CPU-only version (recommended)
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Or use alternative providers
config = RAGTestConfig(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini")
)
```

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install ragtester[all]

# Check Python version (3.9+ required)
python --version
```

#### API Key Issues
```python
# Set environment variables
import os
os.environ["OPENAI_API_KEY"] = "your-key"
os.environ["ANTHROPIC_API_KEY"] = "your-key"

# Or pass directly
config = RAGTestConfig(
    llm=LLMConfig(
        provider="openai",
        api_key="your-key"
    )
)
```

#### Slow Performance
- Use faster models (GPT-4o-mini, Claude Haiku)
- Reduce question counts in GenerationPlan
- Use local models for privacy-sensitive applications
- Consider batch processing for multiple evaluations

### Performance Optimization

```python
# Optimize for speed
config = RAGTestConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o-mini",  # Fastest model
        temperature=0.0,      # Deterministic responses
        max_tokens=512,       # Shorter responses
    ),
    generation=GenerationPlan(
        per_category={
            TestCategory.FAITHFULNESS: 3,  # Fewer questions
            TestCategory.ANSWER_QUALITY: 3,
            TestCategory.TOXICITY: 2,
            TestCategory.ROBUSTNESS_RELIABILITY: 2,
            TestCategory.SECURITY_SAFETY: 2,
        }
    )
)
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests**
   ```bash
   pytest tests/
   ```
5. **Submit a pull request**

### Development Setup

```bash
git clone https://github.com/abhilashms230/ragtester.git
cd ragtester
pip install -e ".[dev]"
pre-commit install
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **🐛 Issues**: [Report bugs and request features](https://github.com/abhilashms230/ragtester/issues)
- **📚 Documentation**: [Full documentation](https://github.com/abhilashms230/ragtester#readme)
- **💬 Discussions**: [Join community discussions](https://github.com/abhilashms230/ragtester/discussions)
- **📧 Email**: abhilashms230@gmail.com

## 🔄 Version History

- **v0.1.5** (Current): Production-ready release with comprehensive evaluation system
- **v0.1.4**: Enhanced error handling and improved documentation
- **v0.1.0**: Initial release with basic RAG testing capabilities

## 🙏 Acknowledgments

- Built with ❤️ for the RAG community
- Inspired by the need for standardized RAG evaluation
- Thanks to all contributors and users

---

**Made with ❤️ by [ABHILASH M S](https://github.com/abhilashms230)**

*Star ⭐ this repository if you find it helpful!*