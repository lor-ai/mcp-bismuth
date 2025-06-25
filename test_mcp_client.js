import { spawn } from 'child_process';
import { createInterface } from 'readline';

class MCPClient {
  constructor() {
    this.server = null;
    this.requestId = 0;
  }

  async start() {
    console.log('Starting Titan Memory MCP Server...');
    
    // Start the MCP server
    this.server = spawn('node', ['index.js'], {
      cwd: process.cwd(),
      stdio: ['pipe', 'pipe', 'inherit']
    });

    const readline = createInterface({
      input: this.server.stdout,
      output: this.server.stdin
    });

    // Set up message handling
    this.server.stdout.on('data', (data) => {
      const responses = data.toString().split('\n').filter(line => line.trim());
      responses.forEach(response => {
        try {
          const parsed = JSON.parse(response);
          console.log('Server response:', JSON.stringify(parsed, null, 2));
        } catch (e) {
          console.log('Raw server output:', response);
        }
      });
    });

    this.server.on('close', (code) => {
      console.log(`Server process exited with code ${code}`);
    });

    // Wait a moment for server to start
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    return this;
  }

  async sendRequest(method, params = {}) {
    const id = ++this.requestId;
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };

    console.log(`Sending request: ${JSON.stringify(request)}`);
    this.server.stdin.write(JSON.stringify(request) + '\n');

    // Wait for response (simplified - in real implementation you'd track by ID)
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  async testANNFunctionality() {
    console.log('\n=== Testing ANN Functionality ===\n');

    try {
      // Initialize the model with large memory to trigger ANN
      await this.sendRequest('init_model', {
        memorySlots: 5000, // Large enough to potentially trigger ANN
        inputDim: 768,
        useApproximateNearestNeighbors: true
      });

      // Get initial memory stats
      await this.sendRequest('memory_stats');

      // Store some memories
      console.log('\n--- Storing memories ---');
      await this.sendRequest('forward_pass', {
        x: 'This is the first memory about artificial intelligence and machine learning'
      });

      await this.sendRequest('forward_pass', {
        x: 'Neural networks are powerful computational models for pattern recognition'
      });

      await this.sendRequest('forward_pass', {
        x: 'Deep learning enables hierarchical feature extraction from data'
      });

      // Get updated memory stats
      await this.sendRequest('get_memory_state');

      // Test training step
      console.log('\n--- Testing training step ---');
      await this.sendRequest('train_step', {
        x_t: 'Machine learning algorithms learn patterns',
        x_next: 'from large datasets automatically'
      });

      console.log('\n--- Final memory stats ---');
      await this.sendRequest('get_memory_state');

    } catch (error) {
      console.error('Error testing ANN functionality:', error);
    }
  }

  async stop() {
    if (this.server) {
      this.server.kill();
    }
  }
}

async function main() {
  const client = new MCPClient();
  
  try {
    await client.start();
    await client.testANNFunctionality();
  } catch (error) {
    console.error('Test failed:', error);
  } finally {
    await client.stop();
  }
}

main().catch(console.error);
