# vlllm: High-Performance Text Generation with vLLM and Multiprocessing

[![PyPI version](https://badge.fury.io/py/vlllm.svg)](https://badge.fury.io/py/vlllm) `vlllm` is a Python utility package designed to simplify and accelerate text generation tasks using the powerful [vLLM](https://github.com/vllm-project/vllm) library. It offers a convenient interface for batch processing, chat templating, multiple sampling strategies, and multi-GPU inference with tensor and pipeline parallelism, all wrapped in an easy-to-use `generate` function with multiprocessing support.

## Features

* **Batch Processing**: Efficiently process lists of prompts.
* **Flexible Input**: Supports both single string prompts and list-based chat message formats (e.g., `[{"role": "user", "content": "Hello!"}]`).
* **System Prompts**: Easily integrate system-level instructions.
* **Multiple Samples (`n`)**: Generate multiple completions per prompt.
    * **Input Duplication Strategy (`use_sample=False`)**: Duplicates input prompts `n` times for generation.
    * **vLLM Native Sampling (`use_sample=True`)**: Uses vLLM's internal sampling parameter (`SamplingParams(n=n)`) for generating `n` completions.
* **Multiprocessing (`worker_num`)**: Distribute generation tasks across multiple CPU worker processes, each potentially managing its own vLLM instance and GPU(s).
* **Tensor Parallelism (`tp` or `gpu_assignments`)**: Configure tensor parallelism for vLLM instances within each worker.
* **Pipeline Parallelism (`pp`)**: Supports vLLM's pipeline parallelism (requires `pp > 1` and uses `distributed_executor_backend="ray"`).
* **Chunking (`chunk_size`)**: Control the maximum number of prompts processed by a vLLM engine in a single call, useful for managing memory and very large datasets.
* **Customizable Output**: Specify the key under which results are stored.
* **Robust GPU Management**: Automatic or manual assignment of GPUs to workers.

## Installation

```bash
pip install vlllm
```

## Quick Start

```python
from vlllm import generate

# Example data with string prompts
data = [
    {"prompt": "Write a story about a dragon"},
    {"prompt": "Explain quantum computing"}
] * 1000

# Basic usage
results = generate(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    data=data,
    worker_num=2,  # Use 2 worker processes
    tp=1,          # 1 GPU per worker
)

# Each item in results will have a new 'results' field with the generated text
print(results[0]["results"])
```

## Parameters

### Core Parameters

- **`model_id`** (str): Model identifier or path to load
- **`data`** (List[Dict]): List of dictionaries containing prompts/messages
- **`message_key`** (str, default: "prompt"): Key in each dictionary containing the prompt or messages
- **`system`** (str, optional): Global system prompt to prepend to all messages
- **`result_key`** (str, default: "results"): Key name for storing generation results

### Message Format Handling

The package intelligently handles different input formats:

1. **String format**: If `data[i][message_key]` is a string, it's automatically converted to a chat message format
2. **List format**: If `data[i][message_key]` is a list, it's treated as a chat conversation with roles and content

When a system prompt is provided:
- For string inputs: Creates a message list with system and user messages
- For list inputs: Prepends the system message (unless one already exists)

### Generation Parameters

- **`n`** (int, default: 1): Number of samples to generate per prompt
- **`use_sample`** (bool, default: False): 
  - If `False`: Duplicates each prompt `n` times in the generation list
  - If `True`: Uses vLLM's native `SamplingParams(n=n)` for efficient sampling
- **`temperature`** (float, default: 0.7): Sampling temperature
- **`max_output_len`** (int, default: 1024): Maximum tokens to generate per sample

### Result Format

- If `n=1`: The `result_key` field contains a single string
- If `n>1`: The `result_key` field contains a list of strings

### Parallelization Parameters

- **`worker_num`** (int, default: 1): Number of worker processes
  - If 1: Single process execution
  - If >1: Multi-process execution with data evenly distributed
- **`tp`** (int, default: 1): Tensor parallel size per worker
- **`pp`** (int, default: 1): Pipeline parallel size
  - If >1: Uses Ray distributed backend (requires `worker_num=1`)
- **`gpu_assignments`** (List[List[int]], optional): Custom GPU assignments per worker

### Performance Parameters

- **`chunk_size`** (int, optional): Maximum items per generation batch
  - If not set: Each worker processes its entire partition at once
  - If set: Data is processed in chunks of this size
- **`max_model_len`** (int, default: 4096): Maximum model sequence length
- **`gpu_memory_utilization`** (float, default: 0.90): Target GPU memory usage
- **`dtype`** (str, default: "auto"): Model data type
- **`trust_remote_code`** (bool, default: True): Whether to trust remote code

## Advanced Usage

### Chat Format with Multiple Samples

```python
# Data with chat message format
data = [
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ]
    }
] * 100

# Generate 3 different responses per prompt
results = generate(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    data=data,
    message_key="messages",        # Specify the key containing messages
    system="You are a helpful assistant.",  # Global system prompt
    n=3,                          # Generate 3 samples
    use_sample=True,              # Use vLLM's native sampling
    temperature=0.8,
    worker_num=2,
    tp=2                          # Use 2 GPUs per worker
)

# results[0]["results"] will be a list of 3 different responses
for i, response in enumerate(results[0]["results"]):
    print(f"Response {i+1}: {response}")
```

### Processing Large Datasets with Chunking

```python
# Large dataset
data = [{"prompt": f"Question {i}"} for i in range(10000)]

results = generate(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    data=data,
    worker_num=4,
    chunk_size=100,  # Process in chunks of 100 items
    tp=1,
    max_output_len=512
)
```

### Custom GPU Assignment

```python
# Assign specific GPUs to each worker
results = generate(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    data=data,
    worker_num=2,
    gpu_assignments=[[0, 1], [2, 3]],  # Worker 0 uses GPU 0,1; Worker 1 uses GPU 2,3
)
```

### Pipeline Parallelism

```python
# Use pipeline parallelism (requires worker_num=1)
results = generate(
    model_id="meta-llama/Llama-2-70b-chat-hf",
    data=data,
    worker_num=1,
    pp=4,  # 4-way pipeline parallelism
    tp=2,  # 2-way tensor parallelism
)
```

## Important Notes

- **Pipeline Parallelism**: When `pp > 1`, `worker_num` must be 1
- **GPU Requirements**: Total GPUs needed = `worker_num * tp` (when not using custom assignments)
- **Memory Management**: The package automatically handles memory cleanup between batches
- **Error Handling**: Failed generations are marked with error messages in the results
- **Process Safety**: Uses spawn method for multiprocessing on POSIX systems

## Example: Batch Processing Pipeline

```python
from vlllm import generate
import json

# Load your dataset
with open("questions.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Configure generation
results = generate(
    model_id="meta-llama/Llama-2-13b-chat-hf",
    data=data,
    message_key="question",     # Your data has questions in 'question' field
    system="Answer concisely and accurately.",
    n=1,
    temperature=0.1,           # Low temperature for consistency
    worker_num=4,              # 4 parallel workers
    tp=2,                      # 2 GPUs per worker
    chunk_size=50,             # Process 50 items at a time
    max_output_len=256,
    result_key="answer"        # Store results in 'answer' field
)

# Save results
with open("answers.jsonl", "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")
```

## Requirements

- Python >= 3.8
- vLLM
- PyTorch
- Transformers
- CUDA-capable GPUs (for GPU acceleration)

## License

Apache-2.0 License