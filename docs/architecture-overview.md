# Architecture Overview

This document provides an overview of the architecture for the Titan Memory MCP Server, describing the key components involved in processing and managing memory for LLMs (Large Language Models).

---

## System Architecture Diagram

```mermaid
graph TD;
    tokenizer[Tokenizer] --> embedding[Embedding]
    embedding --> encoder[Encoder (TF-JS)]
    encoder --> memory[Memory Retrieval]
    memory --> decoder[Decoder]
    decoder --> online[Online Update]
```

---

## Tokenizer
**Function:**
The tokenizer component is responsible for breaking down input text into manageable tokens. This allows the rest of the model to process data efficiently without dealing directly with natural language inputs.

---

## Embedding
**Function:**
Once tokenized, text is converted into vector embeddings. These embeddings represent the meaning of the text in a numerical format suitable for further processing.

---

## Encoder (TF-JS)
**Function:**
The encoder, implemented using TensorFlow.js, processes the embeddings to extract features and create contextually relevant representations. These representations are essential for understanding the nuances of input data.

---

## Memory Retrieval
**Function:**
Using attention mechanisms and HNSW (Hierarchical Navigable Small World) algorithms, the system retrieves memory items relevant to the current context from long-term memory, enhancing decision-making and prediction accuracy.

---

## Decoder
**Function:**
The decoder takes the refined context from the encoder and retrieved memory to generate coherent outputs. This step is critical in forming meaningful responses based on both input and context.

---

## Online Update
**Function:**
This process involves updating the model in real-time, allowing it to adapt to new data and improve its performance continuously. The online update mechanism ensures that the model remains current with the latest information without needing full retraining.

---

## Checkpoint Flow and MCP Tool Mapping

### Checkpoint Flow
1. **Save**: Memory state is periodically saved as checkpoints to ensure data persistence and facilitate recovery.
2. **Load**: Checkpoints can be loaded to restore the model's state to a known good configuration.

### MCP Tools Mapping
- **`init_model`**: Initializes the model configuration.
- **`save_checkpoint`**: Saves the current memory state to a file.
- **`load_checkpoint`**: Loads a saved memory state file.
- **`forward_pass`**: Processes data through the model to make predictions.
- **`train_step`**: Updates model based on new data inputs.
- **`get_memory_state`**: Retrieves current memory statistics.
- **`prune_memory`**: Cleans up older or less useful memories to optimize performance.

---

## Additional Information
For more detailed guidance on system prompts, consult the [LLM System Prompt Documentation](llm-system-prompt.md).

---
