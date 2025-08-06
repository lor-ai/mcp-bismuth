// ANN implementation without TensorFlow.js dependencies for Windows development
export class HNSW {
  constructor() {
    this.indexBuilt = false;
    this.nodes = new Map();
    this.entryPoint = null;
    this.maxLevel = 0;
    this.maxConnections = 16;
    this.maxConnectionsLevel0 = 32;
    this.levelMultiplier = 1 / Math.log(2);
    this.efConstruction = 200;
    this.efSearch = 50;
    this.memoryChangedFlag = false;
    this.lastSlotCount = 0;
    this.data = [];
  }

  // Utility function to compute cosine similarity
  computeSimilarity(vecA, vecB) {
    if (vecA.length !== vecB.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    
    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  // Utility function to compute euclidean distance
  computeDistance(vecA, vecB) {
    if (vecA.length !== vecB.length) return Infinity;
    
    let sum = 0;
    for (let i = 0; i < vecA.length; i++) {
      const diff = vecA[i] - vecB[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  getRandomLevel() {
    let level = 0;
    while (Math.random() < 0.5 && level < 16) {
      level++;
    }
    return level;
  }

  async buildIndex(data) {
    console.log(`Building HNSW index with ${data.length} vectors of dimension ${data[0]?.length || 0}...`);
    
    this.nodes.clear();
    this.entryPoint = null;
    this.maxLevel = 0;
    this.data = [...data]; // Keep reference to original data

    // Insert nodes one by one
    for (let i = 0; i < data.length; i++) {
      this.insertNode(i, data[i]);
    }

    this.indexBuilt = true;
    this.memoryChangedFlag = false;
    this.lastSlotCount = data.length;
    
    console.log(`Index built with ${this.nodes.size} nodes, max level: ${this.maxLevel}`);
    return Promise.resolve();
  }

  insertNode(id, vector) {
    const level = this.getRandomLevel();
    const node = {
      id,
      vector: [...vector], // Clone the vector
      connections: new Map()
    };

    // Initialize connections for all levels
    for (let l = 0; l <= level; l++) {
      node.connections.set(l, new Set());
    }

    this.nodes.set(id, node);

    if (this.entryPoint === null || level > this.maxLevel) {
      this.entryPoint = id;
      this.maxLevel = level;
    }

    // Connect to existing nodes (simplified version)
    if (this.nodes.size > 1) {
      this.connectNode(node, level);
    }
  }

  connectNode(node, level) {
    // Find nearest neighbors and connect (simplified implementation)
    const candidates = [];
    
    // Collect all existing nodes for comparison
    for (const [nodeId, existingNode] of this.nodes) {
      if (nodeId === node.id) continue;
      
      const distance = this.computeDistance(node.vector, existingNode.vector);
      candidates.push({ id: nodeId, distance });
    }

    // Sort by distance and take the best candidates
    candidates.sort((a, b) => a.distance - b.distance);
    
    for (let l = 0; l <= level; l++) {
      const maxConns = l === 0 ? this.maxConnectionsLevel0 : this.maxConnections;
      const selectedCandidates = candidates.slice(0, maxConns);

      // Connect to selected candidates
      for (const candidate of selectedCandidates) {
        this.addConnection(node.id, candidate.id, l);
      }
    }
  }

  addConnection(nodeId1, nodeId2, level) {
    const node1 = this.nodes.get(nodeId1);
    const node2 = this.nodes.get(nodeId2);

    if (node1 && node2) {
      node1.connections.get(level)?.add(nodeId2);
      node2.connections.get(level)?.add(nodeId1);
    }
  }

  async search(query, topK) {
    if (!this.indexBuilt || this.entryPoint === null) {
      throw new Error('Index not built yet');
    }

    console.log(`Searching for top ${topK} similar vectors...`);

    // Simplified search - compute distances to all nodes and return top K
    const candidates = [];
    
    for (const [nodeId, node] of this.nodes) {
      const distance = this.computeDistance(query, node.vector);
      candidates.push({ 
        id: nodeId, 
        distance, 
        vector: node.vector 
      });
    }

    // Sort by distance (ascending) and take top K
    candidates.sort((a, b) => a.distance - b.distance);
    const results = candidates.slice(0, topK).map(c => c.vector);
    
    console.log(`Found ${results.length} results, distances: ${candidates.slice(0, topK).map(c => c.distance.toFixed(4)).join(', ')}`);
    
    return Promise.resolve(results);
  }

  needsRebuild(memoryChanged, slotCount) {
    const shouldRebuild = memoryChanged && slotCount > 2000;

    if (shouldRebuild) {
      this.memoryChangedFlag = true;
    }

    // Also rebuild if the index size has changed significantly
    const sizeChangeThreshold = 0.1; // 10% change
    const sizeChanged = this.lastSlotCount > 0 &&
      Math.abs(slotCount - this.lastSlotCount) / Math.max(this.lastSlotCount, 1) > sizeChangeThreshold;

    return shouldRebuild || (this.indexBuilt && sizeChanged);
  }

  dispose() {
    this.data = [];
    this.nodes.clear();
    this.indexBuilt = false;
    this.entryPoint = null;
  }

  // Public getters for compatibility
  get isIndexBuilt() {
    return this.indexBuilt;
  }

  get hnswNodes() {
    return this.nodes;
  }

  getParameters() {
    return {
      maxConnections: this.maxConnections,
      maxConnectionsLevel0: this.maxConnectionsLevel0,
      efConstruction: this.efConstruction,
      efSearch: this.efSearch
    };
  }

  setParameters(params) {
    if (params.maxConnections !== undefined) { this.maxConnections = params.maxConnections; }
    if (params.maxConnectionsLevel0 !== undefined) { this.maxConnectionsLevel0 = params.maxConnectionsLevel0; }
    if (params.efConstruction !== undefined) { this.efConstruction = params.efConstruction; }
    if (params.efSearch !== undefined) { this.efSearch = params.efSearch; }
  }
}
