# Setting up MCP Titan Memory Server in Cursor

## Quick Setup

1. **Ensure the project is built:**
   ```bash
   cd /Users/henrymayo/Desktop/mcp-titan
   npm install
   npm run build
   ```

2. **Test the server manually:**
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"help","arguments":{}}}' | node index.js
   ```

3. **Configure Cursor MCP:**
   - Open Cursor Settings
   - Navigate to "MCP Servers" or "Extensions" > "MCP"
   - Add a new server configuration:
     ```json
     {
       "mcpServers": {
         "titan-memory": {
           "command": "node",
           "args": ["index.js"],
           "cwd": "/Users/henrymayo/Desktop/mcp-titan"
         }
       }
     }
     ```

4. **Restart Cursor** to load the MCP server

## Available Tools

The Titan Memory MCP server provides these tools:

- **`help`** - Get help about available tools
- **`init_model`** - Initialize the Titan Memory model with custom configuration
- **`forward_pass`** - Perform a forward pass through the model
- **`train_step`** - Execute a training step to update the model
- **`get_memory_state`** - Get current memory state and statistics  
- **`manifold_step`** - Update memory along a manifold direction
- **`prune_memory`** - Remove less relevant memories to free up space
- **`save_checkpoint`** - Save memory state to a file
- **`load_checkpoint`** - Load memory state from a file
- **`reset_gradients`** - Reset accumulated gradients

## Usage Example

1. **Initialize the model first:**
   ```
   Use the init_model tool with parameters like:
   {
     "inputDim": 768,
     "memorySlots": 1000,
     "transformerLayers": 4
   }
   ```

2. **Process text with forward_pass:**
   ```
   Use forward_pass with text input:
   {
     "x": "Your text input here"
   }
   ```

3. **Train on sequences:**
   ```
   Use train_step with:
   {
     "x_t": "Current text",
     "x_next": "Next text in sequence"
   }
   ```

## Memory Rule Integration

Add this rule to your Cursor rules to ensure proper usage:

```
Rule Name: mcp-titan-memory
Description: When using MCP Titan memory tools, always initialize the model first with init_model before using other tools. Use appropriate input dimensions (768 recommended) and memory slots based on your use case.
```

## Troubleshooting

- **"Method not found" errors:** Ensure you're using the correct MCP protocol format
- **Shape mismatch errors:** Initialize the model with `init_model` first with appropriate `inputDim`
- **Server not starting:** Check that Node.js is installed and `npm run build` completed successfully
- **Permission errors:** Ensure the working directory is writable for checkpoint files 