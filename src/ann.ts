import * as tf from '@tensorflow/tfjs-node';

export interface AnnIndex {
  buildIndex(data: tf.Tensor[]): Promise<void>;
  search(query: tf.Tensor, topK: number): Promise<tf.Tensor[]>;
  needsRebuild(memoryChanged: boolean, slotCount: number): boolean;
}

interface HnswNode {
  id: number;
  vector: tf.Tensor;
  connections: Map<number, Set<number>>; // level -> set of connection IDs
}

export class HNSW implements AnnIndex {
  private indexBuilt: boolean = false;
  private nodes: Map<number, HnswNode> = new Map();
  private entryPoint: number | null = null;
  private maxLevel: number = 0;
  private maxConnections: number = 16;
  private maxConnectionsLevel0: number = 32;
  private levelMultiplier: number = 1 / Math.log(2);
  private efConstruction: number = 200;
  private efSearch: number = 50;
  private memoryChangedFlag: boolean = false;
  private lastSlotCount: number = 0;

  constructor() {
    // Initialize with default parameters
  }

  async buildIndex(data: tf.Tensor[]): Promise<void> {
    return tf.tidy(() => {
      this.nodes.clear();
      this.entryPoint = null;
      this.maxLevel = 0;
      
      // Insert nodes one by one
      for (let i = 0; i < data.length; i++) {
        this.insertNode(i, data[i]);
      }
      
      this.indexBuilt = true;
      this.memoryChangedFlag = false;
      this.lastSlotCount = data.length;
    });
  }

  private insertNode(id: number, vector: tf.Tensor): void {
    const level = this.getRandomLevel();
    const node: HnswNode = {
      id,
      vector: vector.clone(),
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
    
    // Search for closest nodes and connect
    if (this.nodes.size > 1) {
      this.connectNode(node, level);
    }
  }
  
  private connectNode(node: HnswNode, level: number): void {
    const candidates = this.searchLevel(node.vector, this.entryPoint!, level, this.efConstruction);
    
    for (let l = 0; l <= level; l++) {
      const maxConns = l === 0 ? this.maxConnectionsLevel0 : this.maxConnections;
      const selectedCandidates = this.selectNeighbors(candidates, maxConns);
      
      // Connect to selected candidates
      for (const candidateId of selectedCandidates) {
        this.addConnection(node.id, candidateId, l);
        this.pruneConnections(candidateId, l);
      }
    }
  }
  
  private addConnection(nodeId1: number, nodeId2: number, level: number): void {
    const node1 = this.nodes.get(nodeId1);
    const node2 = this.nodes.get(nodeId2);
    
    if (node1 && node2) {
      node1.connections.get(level)?.add(nodeId2);
      node2.connections.get(level)?.add(nodeId1);
    }
  }
  
  private pruneConnections(nodeId: number, level: number): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    
    const connections = node.connections.get(level);
    if (!connections) return;
    
    const maxConns = level === 0 ? this.maxConnectionsLevel0 : this.maxConnections;
    
    if (connections.size <= maxConns) return;
    
    // Select best connections to keep
    const candidates = Array.from(connections).map(id => ({
      id,
      distance: this.computeDistance(node.vector, this.nodes.get(id)!.vector)
    }));
    
    candidates.sort((a, b) => a.distance - b.distance);
    
    // Keep only the best connections
    const newConnections = new Set(candidates.slice(0, maxConns).map(c => c.id));
    node.connections.set(level, newConnections);
  }
  
  private selectNeighbors(candidates: number[], maxCount: number): number[] {
    return candidates.slice(0, maxCount);
  }
  
  private searchLevel(query: tf.Tensor, entryPoint: number, level: number, ef: number): number[] {
    const visited = new Set<number>();
    const candidates: Array<{id: number, distance: number}> = [];
    const w = new Set<number>();
    
    const entryDistance = this.computeDistance(query, this.nodes.get(entryPoint)!.vector);
    candidates.push({id: entryPoint, distance: entryDistance});
    w.add(entryPoint);
    visited.add(entryPoint);
    
    while (candidates.length > 0) {
      // Get closest candidate
      candidates.sort((a, b) => a.distance - b.distance);
      const current = candidates.shift()!;
      
      // Stop if we have enough candidates and current is farther than worst in w
      if (w.size >= ef) {
        const worstInW = Math.max(...Array.from(w).map(id => 
          this.computeDistance(query, this.nodes.get(id)!.vector)
        ));
        if (current.distance > worstInW) break;
      }
      
      // Explore neighbors
      const currentNode = this.nodes.get(current.id)!;
      const connections = currentNode.connections.get(level) || new Set();
      
      for (const neighborId of connections) {
        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          const neighborDistance = this.computeDistance(query, this.nodes.get(neighborId)!.vector);
          
          if (w.size < ef) {
            candidates.push({id: neighborId, distance: neighborDistance});
            w.add(neighborId);
          } else {
            const worstInW = Math.max(...Array.from(w).map(id => 
              this.computeDistance(query, this.nodes.get(id)!.vector)
            ));
            if (neighborDistance < worstInW) {
              candidates.push({id: neighborId, distance: neighborDistance});
              // Remove worst from w
              const worstId = Array.from(w).find(id => 
                this.computeDistance(query, this.nodes.get(id)!.vector) === worstInW
              )!;
              w.delete(worstId);
              w.add(neighborId);
            }
          }
        }
      }
    }
    
    return Array.from(w);
  }
  
  private computeDistance(v1: tf.Tensor, v2: tf.Tensor): number {
    return tf.tidy(() => {
      // Cosine distance: 1 - cosine similarity
      const dot = tf.sum(tf.mul(v1, v2));
      const norm1 = tf.norm(v1);
      const norm2 = tf.norm(v2);
      const cosineSim = tf.div(dot, tf.mul(norm1, norm2));
      return 1 - cosineSim.dataSync()[0];
    });
  }
  
  private getRandomLevel(): number {
    let level = 0;
    while (Math.random() < 0.5 && level < 16) {
      level++;
    }
    return level;
  }

  async search(query: tf.Tensor, topK: number): Promise<tf.Tensor[]> {
    if (!this.indexBuilt || this.entryPoint === null) {
      throw new Error('Index not built yet');
    }
    
    return tf.tidy(() => {
      // Search from top level down to level 1
      let candidates = [this.entryPoint!];
      
      for (let level = this.maxLevel; level > 0; level--) {
        candidates = this.searchLevel(query, candidates[0], level, 1);
      }
      
      // Search level 0 with larger ef
      const finalCandidates = this.searchLevel(query, candidates[0], 0, Math.max(this.efSearch, topK));
      
      // Sort by distance and return top K
      const results = finalCandidates
        .map(id => ({
          id,
          distance: this.computeDistance(query, this.nodes.get(id)!.vector),
          vector: this.nodes.get(id)!.vector
        }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, topK)
        .map(result => result.vector.clone());
      
      return results;
    });
  }

  needsRebuild(memoryChanged: boolean, slotCount: number): boolean {
    const shouldRebuild = memoryChanged && slotCount > 2000;
    
    if (shouldRebuild) {
      this.memoryChangedFlag = true;
    }
    
    // Also rebuild if the index size has changed significantly
    const sizeChangeThreshold = 0.1; // 10% change
    const sizeChanged = Math.abs(slotCount - this.lastSlotCount) / Math.max(this.lastSlotCount, 1) > sizeChangeThreshold;
    
    return shouldRebuild || (this.indexBuilt && sizeChanged);
  }
  
  dispose(): void {
    // Clean up tensors
    for (const node of this.nodes.values()) {
      if (!node.vector.isDisposed) {
        node.vector.dispose();
      }
    }
    this.nodes.clear();
    this.indexBuilt = false;
    this.entryPoint = null;
  }
}
