# RAG-PDF
Gradio-based RAG app for querying PDF content using multiple LLMs from OpenAI and Huggingface. Extracts and summarizes PDF text, performs Retrieval-Augmented Generation with classic and manual methods, leveraging Langchain and FAISS for text splitting and embeddings. Easily compare responses across different models and RAG techniques.

# Deep Learning Multi-Agent Systems

## Overview

**Deep Learning Multi-Agent Systems** is a comprehensive framework designed to develop, train, and deploy deep learning models using advanced multi-agent architectures. Leveraging cutting-edge technologies such as Langchain, Langgraph, Huggingface, and OpenAI, this project facilitates the creation of sophisticated multi-agent environments tailored for complex problem-solving tasks. The framework integrates essential Python libraries including PyTorch, Diffusers, Transformers, and Gradio to ensure robust model development and seamless deployment.

## Features

- **Modular Architecture:** Organized into distinct modules for data handling, model development, training, evaluation, and agent management.
- **Multi-Agent Framework:** Design and implement multiple interacting agents with clear roles and responsibilities using Langchain and Langgraph.
- **Advanced Deep Learning Models:** Utilize state-of-the-art architectures from Huggingface and Transformers for diverse applications.
- **Efficient Training Pipelines:** Optimized with GPU acceleration, mixed precision training, and distributed data parallelism.
- **Comprehensive Configuration:** Manage hyperparameters and model settings via YAML configuration files for easy experimentation.
- **Experiment Tracking:** Integrate with TensorBoard and Weights & Biases for detailed monitoring of training progress and performance metrics.
- **Robust Error Handling:** Implemented with thorough logging and try-except blocks to ensure reliability during data processing and model inference.
- **Scalable Deployments:** Seamlessly deploy models and agents using Gradio interfaces for interactive applications.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-compatible GPU (optional, for GPU acceleration)

### Steps

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/aktraiser/RAG-PDF.git
    cd RAG-PDF
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

Execute the training script with the desired configuration file:


### Running Agents

Agents are defined within the `agents/` directory. To initialize and run a specific agent:


### Monitoring with TensorBoard

Launch TensorBoard to visualize training metrics:


