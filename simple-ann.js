// Simple ANN implementation without TensorFlow.js for testing
export class HNSW {
  constructor() {
    this.indexBuilt = false;
    this.nodes = new Map();
    this.entryPoint = null;
    this.maxLevel = 0;
    this.data = [];
  }

  async buildIndex(data) {
    console.log(`Building index with ${data.length} vectors...`);
    this.data = data;
    this.indexBuilt = true;
    this.maxLevel = Math.floor(Math.log2(data.length));
    this.entryPoint = 0;
    return Promise.resolve();
  }

  async search(query, topK) {
    if (!this.indexBuilt) {
      throw new Error('Index not built yet');
    }

    // Simple linear search for testing
    const results = [];
    for (let i = 0; i < Math.min(this.data.length, topK); i++) {
      results.push(this.data[i]);
    }

    return Promise.resolve(results);
  }

  needsRebuild(memoryChanged, slotCount) {
    // Simple rebuild logic
    return memoryChanged && slotCount > 2000;
  }

  dispose() {
    this.data = [];
    this.nodes.clear();
    this.indexBuilt = false;
  }
}
