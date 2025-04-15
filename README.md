# Giskard LLM Utils

A Python library providing utility functions and tools for working with Large Language Models (LLMs). This library is part of the Giskard ecosystem and provides various utilities for LLM operations, including model management, clustering, and more.

## Purpose

This library aims to simplify working with LLMs by providing:

- A unified interface for different LLM providers through LiteLLM
- Support for both cloud-based and local embedding models
- Easy configuration through environment variables or direct initialization
- Synchronous and asynchronous operations for better performance

## Installation

### Standard Installation

```bash
pip install giskard-lmutils
```

### Local Embedding Support

For local embedding capabilities, install with the `local-embedding` extra:

```bash
pip install "giskard-lmutils[local-embedding]"
```

This will install the required dependencies (`torch` and `transformers`) for running embedding models locally.

### Development Installation

1. Install python, Rye and make
2. Clone this repository
3. Setup the virtual environment using `make setup`

## Using LiteLLMModel

The `LiteLLMModel` class provides a unified interface for working with various LLM providers through the [LiteLLM](https://github.com/BerriAI/litellm) library. It supports both completion and embedding operations, with both synchronous and asynchronous methods.

### Configuration

You can configure the model in two ways:

1. Through environment variables:

```bash
# Required for OpenAI models
export OPENAI_API_KEY="your-api-key"

# Model configuration
export GSK_COMPLETION_MODEL="gpt-3.5-turbo"
export GSK_EMBEDDING_MODEL="text-embedding-ada-002"
```

```python
from giskard_lmutils.model import LiteLLMModel

# This will use environment variables for model names
model = LiteLLMModel(
    completion_params={"temperature": 0.7},
    embedding_params={"is_local": False}  # Optional, defaults to False
)
```

Note: The environment variable prefix can be customized by passing an `env_prefix` parameter to the `LiteLLMModel` initialization. This allows you to use different models within the same application by setting different environment variables (e.g., `CUSTOM_PREFIX_COMPLETION_MODEL`).

2. Through specified model names:

```python
model = LiteLLMModel(
    completion_model="gpt-3.5-turbo",
    embedding_model="text-embedding-ada-002",
    completion_params={"temperature": 0.7},
    embedding_params={"is_local": False}  # Optional, defaults to False
)
```

Note: When using OpenAI models, you must set the `OPENAI_API_KEY` environment variable. For other providers, refer to the [LiteLLM documentation](https://github.com/BerriAI/litellm) for their specific API key requirements.

### Usage Examples

#### Text Completion

```python
# Synchronous completion
response = model.complete([
    {"role": "user", "content": "What is the capital of France?"}
])

# Asynchronous completion
response = await model.acomplete([
    {"role": "user", "content": "What is the capital of France?"}
])
```

#### Text Embedding

```python
# Synchronous embedding
embeddings = model.embed(["Hello, world!", "Another text"])

# Asynchronous embedding
embeddings = await model.aembed(["Hello, world!", "Another text"])

# Local embedding
model = LiteLLMModel(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_params={"is_local": True}
)
embeddings = model.embed(["Hello, world!"])
```

## Requirements

- Python >= 3.9, < 3.14
- Core dependencies:
  - numpy >= 2.2.2
  - litellm >= 1.59.3
- Optional dependencies (for local embedding):
  - torch >= 2.6.0
  - transformers >= 4.51.3

## License

This project is licensed under the Apache Software License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- Kevin Messiaen (kevin@giskard.ai)
