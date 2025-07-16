# Titan Memory MCP Server

The project has been fundamentally fixed - the core architectural problem (incompatible custom ToolResponse interface) has been resolved, and the MCP Titan Memory System is now compatible with the official MCP SDK v1.12.0.
The remaining errors are primarily code quality and type safety improvements rather than blocking functionality issues. The server should now be able to run and provide its 16 sophisticated memory tools for agentic AI models.

## System Prompt for LLMs (Cursor/Claude MCP)

```markdown
You are connected to the @henryhawke/mcp-titan MCP server. Use the tools exactly as documented in docs/llm-system-prompt.md. For a comprehensive overview of the system architecture, see [docs/architecture-overview.md](docs/architecture-overview.md). No human intervention is required except for adding the mcp-titan llm-system-prompt rule to the client.

- Always use the MCP tools for all memory, training, and state operations.
- Always initialize the model with `init_model` before using any other tool.
- Use `help` to discover available tools and their parameter schemas.
- Use `save_checkpoint` and `load_checkpoint` to persist and restore memory state.
- Use `reset_gradients` if you encounter training instability or errors.
- Use `prune_memory` when memory capacity drops below 30%.
- Always check tool responses for errors (`isError: true` or `type: "error"`) and handle them as documented.
- Follow all best practices and error handling as described in docs/llm-system-prompt.md.
- Do not use any implementation details or code not exposed by the server.
- Reference docs/llm-system-prompt.md for the latest schemas and usage examples.

This prompt is copy-pastable and should be used as the system prompt for any LLM (Cursor, Claude, or other MCP-compliant clients) to ensure correct and robust operation with MCP Titan.
```

## Installation & Usage as MCP Server for Cursor or Claude

### Prerequisites

- Node.js (v18 or later recommended)
- npm (comes with Node.js)
- (Optional) Docker, if you want to run in a container

### 1. Clone the Repository

```bash
git clone https://github.com/henryhawke/mcp-titan.git
cd titan-memory
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Build the Project

```bash
npm run build
```

### 4. Start the MCP Server

```bash
npm start
```

The server will start and listen for MCP tool requests. By default, it runs on port 8080 (or as configured in your environment).

### 5. Integrate with Cursor or Claude

- **Cursor**: Ensure MCP is enabled in Cursor settings. Cursor will auto-detect and connect to the running MCP server.

    "titan-memory": {
      "command": "node",
      "args": ["index.js"],
      "cwd": "/Users/henrymayo/Desktop/mcp-titan",
      "autoapprove": [
        "create_entities",
        "create_relations",
        "add_observations",
        "delete_entities",
        "delete_observations",
        "delete_relations",
        "read_graph",
        "search_nodes",
        "open_nodes"
      ]
    },

    
- **Claude Desktop**: Set the MCP server endpoint in Claude's settings to `http://localhost:8080` (or your configured host/port).

### 6. Test the MCP Server

You can use the provided tool APIs (see below) or connect via Cursor/Claude to verify memory operations.

---
Ideally this just runs in yolo mode in cursor (or claude desktop) without human intervention and creates a "brain" available independent of LLM version.

A neural memory system for LLMs that can learn and predict sequences while maintaining state through a memory vector. This MCP (Model Context Protocol) server provides tools for Claude 3.7 Sonnet and other LLMs to maintain memory state across interactions.

## Features

- **Perfect for Cursor**: Now that Cursor automatically runs MCP in yolo mode, you can take your hands off the wheel with your LLM's new memory
- **Neural Memory Architecture**: Transformer-based memory system that can learn and predict sequences
- **Memory Management**: Efficient tensor operations with automatic memory cleanup
- **MCP Integration**: Fully compatible with Cursor and other MCP clients
- **Text Encoding**: Convert text inputs to tensor representations
- **Memory Persistence**: Save and load memory states between sessions

## Available Tools

The Titan Memory MCP server provides the following tools:

### `help`

Get help about available tools.

**Parameters:**

- `tool` (optional): Specific tool name to get help for
- `category` (optional): Category of tools to explore
- `showExamples` (optional): Include usage examples
- `verbose` (optional): Include detailed descriptions

### `init_model`

Initialize the Titan Memory model with custom configuration.

**Parameters:**

- `inputDim`: Input dimension size (default: 768)
- `hiddenDim`: Hidden dimension size (default: 512)
- `memoryDim`: Memory dimension size (default: 1024)
- `transformerLayers`: Number of transformer layers (default: 6)
- `numHeads`: Number of attention heads (default: 8)
- `ffDimension`: Feed-forward dimension (default: 2048)
- `dropoutRate`: Dropout rate (default: 0.1)
- `maxSequenceLength`: Maximum sequence length (default: 512)
- `memorySlots`: Number of memory slots (default: 5000)
- `similarityThreshold`: Similarity threshold (default: 0.65)
- `surpriseDecay`: Surprise decay rate (default: 0.9)
- `pruningInterval`: Pruning interval (default: 1000)
- `gradientClip`: Gradient clipping value (default: 1.0)

### `forward_pass`

Perform a forward pass through the model to get predictions.

**Parameters:**

- `x`: Input vector or text
- `memoryState` (optional): Memory state to use

### `train_step`

Execute a training step to update the model.

**Parameters:**

- `x_t`: Current input vector or text
- `x_next`: Next input vector or text

### `get_memory_state`

Get the current memory state and statistics.

**Parameters:**

- `type` (optional): Optional memory type filter

### `manifold_step`

Update memory along a manifold direction.

**Parameters:**

- `base`: Base memory state
- `velocity`: Update direction

### `prune_memory`

Remove less relevant memories to free up space.

**Parameters:**

- `threshold`: Pruning threshold (0-1)

### `save_checkpoint`

Save memory state to a file.

**Parameters:**

- `path`: Checkpoint file path

### `load_checkpoint`

Load memory state from a file.

**Parameters:**

- `path`: Checkpoint file path

### `reset_gradients`

Reset accumulated gradients to recover from training issues.

**Parameters:** None

## Usage with Claude 3.7 Sonnet in Cursor

The Titan Memory MCP server is designed to work seamlessly with Claude 3.7 Sonnet in Cursor. Here's an example of how to use it:

```javascript
// Initialize the model
const result = await callTool("init_model", {
  inputDim: 768,
  memorySlots: 10000,
  transformerLayers: 8,
});

// Perform a forward pass
const { predicted, memoryUpdate } = await callTool("forward_pass", {
  x: "const x = 5;", // or vector: [0.1, 0.2, ...]
  memoryState: currentMemory,
});

// Train the model
const result = await callTool("train_step", {
  x_t: "function hello() {",
  x_next: "  console.log('world');",
});

// Get memory state
const state = await callTool("get_memory_state", {});
```

## How the MCP Server Learns: TITANS-Inspired Neural Memory

The Titan Memory MCP server implements a sophisticated neural memory architecture inspired by transformer mechanisms and continual learning principles. Here's a detailed breakdown of the learning process:

### Core Learning Architecture

**Three-Tier Memory Hierarchy:**
1. **Short-term Memory** - Recent activations and immediate context
2. **Long-term Memory** - Consolidated patterns and persistent knowledge  
3. **Meta Memory** - Statistics about memory usage and learning patterns

**Surprise-Driven Learning:**
The system uses surprise-based learning mechanisms where unexpected inputs trigger stronger memory updates:
- **Surprise Calculation**: `surprise = ||decoded_output - input||` (L2 norm of prediction error)
- **Surprise Decay**: Previous surprise scores decay exponentially with configurable rate (default: 0.9)
- **Memory Gating**: High surprise opens memory gates for stronger encoding

### Forward Pass Learning Process

1. **Input Encoding**: Text inputs are encoded using advanced BPE tokenization or TF-IDF fallback
2. **Memory Attention**: Transformer-style attention mechanism computes relevance scores across stored memories
3. **Prediction Generation**: Decoder network generates predictions based on attended memories
4. **Surprise Computation**: Compare predictions with actual inputs to calculate surprise
5. **Memory Update**: Update all three memory tiers based on surprise magnitude

```typescript
// Core forward pass with memory updates
public forward(input: ITensor, state?: IMemoryState): {
  predicted: ITensor;
  memoryUpdate: IMemoryUpdateResult;
} {
  const encodedInput = this.encoder.predict(inputTensor);
  const memoryResult = this.computeMemoryAttention(encodedInput);
  const decoded = this.decoder.predict([encodedInput, memoryResult]);
  const surprise = tf.norm(tf.sub(decoded, inputTensor));
  
  // Update memory based on surprise
  const newMemoryState = this.updateMemory(encodedInput, surprise, memoryState);
  
  return { predicted: decoded, memoryUpdate: newMemoryState };
}
```

### Training Step Learning Process

**Predictive Learning:**
- Each training step predicts the next input given the current input
- Loss computed as mean squared error between prediction and target
- Gradients computed using TensorFlow.js automatic differentiation

**Memory-Augmented Training:**
1. **Attention-Based Retrieval**: Query current memory using input as key
2. **Gradient Computation**: Backpropagate through attention mechanism
3. **Weight Updates**: Update encoder, decoder, and attention networks
4. **Memory Consolidation**: Move important patterns from short-term to long-term memory

### Online Learning Service

**Ring Buffer Replay System:**
- Maintains circular buffer of training samples (default: 10,000 samples)
- Samples mini-batches for continuous learning (default: 32 samples)
- Three learning objectives combined with configurable weights:

```typescript
// Mixed loss function combining multiple learning signals
private computeMixedLoss(batch: TrainingSample[]): {
  loss: tf.Scalar;
  gradients: Map<string, tf.Tensor>;
} {
  let totalLoss = tf.scalar(0);
  
  // Next-token prediction (40% weight)
  if (this.config.nextTokenWeight > 0) {
    const nextTokenLoss = this.computeNextTokenLoss(batch);
    totalLoss = tf.add(totalLoss, tf.mul(nextTokenLoss, 0.4));
  }
  
  // Contrastive learning (20% weight)
  if (this.config.contrastiveWeight > 0) {
    const contrastiveLoss = this.computeContrastiveLoss(batch);
    totalLoss = tf.add(totalLoss, tf.mul(contrastiveLoss, 0.2));
  }
  
  // Masked language modeling (40% weight)
  if (this.config.mlmWeight > 0) {
    const mlmLoss = this.computeMLMLoss(batch);
    totalLoss = tf.add(totalLoss, tf.mul(mlmLoss, 0.4));
  }
  
  return { loss: totalLoss, gradients };
}
```

### Advanced Learning Features

**Hierarchical Memory (Optional):**
- Multiple memory levels with different time scales
- Higher levels update less frequently (powers of 2)
- Enables long-term pattern recognition and forgetting

**Information-Gain Based Pruning:**
- Automatically removes low-relevance memories when capacity reached
- Scores memories based on: recency, frequency, surprise history
- Distills important patterns into long-term storage before pruning

**Gradient Management:**
- **Gradient Clipping**: Prevents exploding gradients (default: 1.0)
- **Gradient Accumulation**: Accumulates gradients over multiple steps for stability
- **NaN Guards**: Detects and skips corrupted gradient updates

### Memory Persistence and Bootstrapping

**Automatic State Persistence:**
- Memory state auto-saved every 60 seconds
- Checkpoint system for manual save/load operations
- Graceful shutdown with state preservation

**Bootstrap Learning:**
- `bootstrap_memory` tool initializes memory from URLs or text corpora
- TF-IDF vectorizer provides sparse fallback for untrained models
- Seed summaries populate initial memory state

### Continual Learning Loop

The online learning service runs continuously in the background:

1. **Sample Collection**: Gather training samples from interactions
2. **Batch Formation**: Create mini-batches from replay buffer
3. **Mixed Loss Computation**: Combine multiple learning objectives
4. **Gradient Application**: Update model weights with clipped gradients
5. **Memory Consolidation**: Promote important short-term memories to long-term storage
6. **Pruning**: Remove irrelevant memories to maintain performance

This architecture enables the MCP server to continuously learn from interactions while maintaining stable, long-term memory that persists across sessions and model updates.

## Memory Management

The Titan Memory MCP server includes sophisticated memory management to prevent memory leaks and ensure efficient tensor operations:

1. **Automatic Cleanup**: Periodically cleans up unused tensors using `tf.tidy()`
2. **Memory Encryption**: Securely stores memory states with AES-256-CBC encryption
3. **Tensor Validation**: Ensures tensors have correct shapes and are not disposed
4. **Error Recovery**: Handles tensor errors gracefully with fallback mechanisms

## Architecture

The Titan Memory MCP server is built with a modular architecture:

- **TitanMemoryServer**: Main server class that registers 16 MCP tools and handles requests
- **TitanMemoryModel**: Neural memory model with transformer-inspired attention mechanisms
- **LearnerService**: Online learning loop with replay buffer and mixed loss functions
- **MemoryPruner**: Information-gain based pruning for memory management
- **AdvancedTokenizer**: BPE tokenization with embedding capabilities
- **VectorProcessor**: Text encoding and tensor operations with safety guards

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
