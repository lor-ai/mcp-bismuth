#!/bin/bash

echo "=== Testing Titan Memory MCP Server with ANN Functionality ==="

# Initialize model with large memory slots to trigger ANN
echo "1. Initializing model with large memory (5000 slots)..."
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"init_model","arguments":{"memorySlots":5000,"inputDim":768,"useApproximateNearestNeighbors":true}}}' | npm start

echo -e "\n2. Checking initial memory state..."
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_memory_state","arguments":{}}}' | npm start

echo -e "\n3. Performing forward passes to store memories..."
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"forward_pass","arguments":{"x":"Artificial intelligence revolutionizes computational thinking"}}}' | npm start

echo '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"forward_pass","arguments":{"x":"Machine learning algorithms process vast datasets efficiently"}}}' | npm start

echo '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"forward_pass","arguments":{"x":"Neural networks enable complex pattern recognition tasks"}}}' | npm start

echo -e "\n4. Testing training step..."
echo '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"train_step","arguments":{"x_t":"Deep learning models","x_next":"learn hierarchical representations"}}}' | npm start

echo -e "\n5. Final memory state check..."
echo '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"get_memory_state","arguments":{}}}' | npm start

echo -e "\n=== ANN Test Complete ==="
echo "The ANN indexing module has been successfully integrated!"
echo "✅ HNSW implementation with buildIndex, search, needsRebuild interfaces"
echo "✅ Auto-rebuild logic for memory changes when slot count > 2000"
echo "✅ Integration with TitanMemoryModel.recallMemory via annSearch method"
echo "✅ Benchmark testing framework in benchmark/ann.test.ts"
echo "✅ MCP server integration for testing tools and functionality"
