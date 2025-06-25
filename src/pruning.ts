/**
 * @fileoverview Smart Memory Pruning with Information-Gain
 * 
 * This module implements intelligent memory pruning based on:
 * - Score = entropy × surprise – redundancy (pairwise cosine similarity)
 * - Keep top percentage of memories
 * - Distill vectors by averaging into long-term memory
 * - Triggered by capacity limits or MCP `prune_memory` command
 */

import * as tf from '@tensorflow/tfjs-node';
import type { IMemoryState, ITensor } from './types.js';

export interface PruningConfig {
  /** Percentage of memories to keep after pruning (0.0 to 1.0) */
  keepPercentage: number;
  /** Minimum number of memories to keep regardless of score */
  minMemoriesToKeep: number;
  /** Maximum capacity before automatic pruning triggers */
  maxCapacity: number;
  /** Weight for entropy component in scoring */
  entropyWeight: number;
  /** Weight for surprise component in scoring */
  surpriseWeight: number;
  /** Weight for redundancy penalty in scoring */
  redundancyWeight: number;
  /** Similarity threshold for considering memories redundant */
  redundancyThreshold: number;
  /** Whether to enable distillation of pruned memories */
  enableDistillation: boolean;
}

export interface PruningResult {
  /** Number of memories before pruning */
  originalCount: number;
  /** Number of memories after pruning */
  finalCount: number;
  /** Number of memories distilled into long-term storage */
  distilledCount: number;
  /** Average information score of kept memories */
  averageScore: number;
  /** Memory reduction ratio */
  reductionRatio: number;
  /** Updated memory state */
  newMemoryState: IMemoryState;
}

export interface MemoryScore {
  /** Memory index */
  index: number;
  /** Information gain score */
  score: number;
  /** Entropy component */
  entropy: number;
  /** Surprise component */
  surprise: number;
  /** Redundancy penalty */
  redundancy: number;
}

/**
 * Smart Memory Pruning System
 * 
 * Implements information-gain based pruning to maintain memory quality
 * while reducing storage requirements.
 */
export class MemoryPruner {
  private config: PruningConfig;
  private pruningHistory: number[] = [];
  private lastPruningTime: number = 0;

  constructor(config: Partial<PruningConfig> = {}) {
    this.config = {
      keepPercentage: 0.7, // Keep 70% of memories by default
      minMemoriesToKeep: 100,
      maxCapacity: 5000,
      entropyWeight: 1.0,
      surpriseWeight: 1.2,
      redundancyWeight: 0.8,
      redundancyThreshold: 0.85,
      enableDistillation: true,
      ...config
    };
  }

  /**
   * Check if pruning should be triggered based on capacity
   */
  shouldPrune(memoryState: IMemoryState): boolean {
    const currentSize = this.getMemorySize(memoryState);
    return currentSize >= this.config.maxCapacity;
  }

  /**
   * Get the current number of active memories
   */
  private getMemorySize(memoryState: IMemoryState): number {
    return tf.tidy(() => {
      // Count non-zero entries in timestamps as active memories
      const nonZeroMask = tf.greater(memoryState.timestamps, 0);
      return tf.sum(tf.cast(nonZeroMask, 'int32')).dataSync()[0];
    });
  }

  /**
   * Perform smart pruning with information-gain scoring
   */
  async pruneMemory(memoryState: IMemoryState): Promise<PruningResult> {
    const originalCount = this.getMemorySize(memoryState);
      
      if (originalCount <= this.config.minMemoriesToKeep) {
        return {
          originalCount,
          finalCount: originalCount,
          distilledCount: 0,
          averageScore: 0,
          reductionRatio: 0,
          newMemoryState: memoryState
        };
      }

      // Calculate information-gain scores for all memories
      const scores = this.calculateInformationScores(memoryState);
      
      // Determine how many memories to keep
      const targetCount = Math.max(
        this.config.minMemoriesToKeep,
        Math.floor(originalCount * this.config.keepPercentage)
      );

      // Sort by score and select top memories
      const sortedIndices = this.sortMemoriesByScore(scores);
      const keepIndices = sortedIndices.slice(0, targetCount);
      const pruneIndices = sortedIndices.slice(targetCount);

      // Create pruned memory state
      const newMemoryState = this.createPrunedState(memoryState, keepIndices);

      // Optionally distill pruned memories into long-term storage
      let distilledCount = 0;
      if (this.config.enableDistillation && pruneIndices.length > 0) {
        newMemoryState.longTerm = this.distillMemories(
          memoryState,
          pruneIndices,
          newMemoryState.longTerm
        );
        distilledCount = pruneIndices.length;
      }

      // Calculate statistics
      const keptScores = keepIndices.map(i => scores[i].score);
      const averageScore = keptScores.reduce((a, b) => a + b, 0) / keptScores.length;
      const reductionRatio = (originalCount - targetCount) / originalCount;

      // Update pruning history
      this.pruningHistory.push(reductionRatio);
      this.lastPruningTime = Date.now();

      return {
        originalCount,
        finalCount: targetCount,
        distilledCount,
        averageScore,
        reductionRatio,
        newMemoryState
      };
  }

  /**
   * Calculate information-gain scores for all memories
   * Score = entropy × surprise – redundancy
   */
  private calculateInformationScores(memoryState: IMemoryState): MemoryScore[] {
    const memoryCount = memoryState.shortTerm.shape[0];
    const scores: MemoryScore[] = [];

      // Calculate entropy for each memory
      const entropies = this.calculateEntropy(memoryState.shortTerm);
      
      // Use surprise history directly
      const surprises = Array.from(memoryState.surpriseHistory.dataSync());
      
      // Calculate pairwise redundancy
      const redundancies = this.calculateRedundancy(memoryState.shortTerm);

      for (let i = 0; i < memoryCount; i++) {
        const entropy = entropies[i];
        const surprise = surprises[i];
        const redundancy = redundancies[i];

        const score = 
          (this.config.entropyWeight * entropy) +
          (this.config.surpriseWeight * surprise) -
          (this.config.redundancyWeight * redundancy);

        scores.push({
          index: i,
          score,
          entropy,
          surprise,
          redundancy
        });
      }

      return scores;
  }

  /**
   * Calculate entropy for each memory vector
   */
  private calculateEntropy(memories: ITensor): number[] {
    return tf.tidy(() => {
      // Normalize memories to probability distributions
      const memoryData = memories as tf.Tensor2D;
      const probabilities = tf.softmax(memoryData, 1);
      
      // Calculate entropy: -∑(p * log(p))
      const logProbs = tf.log(tf.add(probabilities, tf.scalar(1e-8))); // Add epsilon for numerical stability
      const entropies = tf.neg(tf.sum(tf.mul(probabilities, logProbs), 1));
      
      return Array.from(entropies.dataSync());
    });
  }

  /**
   * Calculate redundancy scores based on pairwise cosine similarity
   */
  private calculateRedundancy(memories: ITensor): number[] {
    return tf.tidy(() => {
      const memoryData = memories as tf.Tensor2D;
      const memoryCount = memoryData.shape[0];
      
      // Normalize vectors for cosine similarity
      const normalizedMemories = tf.div(
        memoryData,
        tf.norm(memoryData, 2, 1, true)
      );
      
      // Calculate pairwise cosine similarities
      const similarities = tf.matMul(normalizedMemories, normalizedMemories, false, true);
      
      // Calculate redundancy for each memory as max similarity with others
      const redundancies: number[] = [];
      const similarityData = similarities.dataSync();
      
      for (let i = 0; i < memoryCount; i++) {
        let maxSimilarity = 0;
        for (let j = 0; j < memoryCount; j++) {
          if (i !== j) {
            const similarity = similarityData[i * memoryCount + j];
            if (similarity > maxSimilarity) {
              maxSimilarity = similarity;
            }
          }
        }
        redundancies.push(maxSimilarity);
      }
      
      return redundancies;
    });
  }

  /**
   * Sort memory indices by their information scores (descending)
   */
  private sortMemoriesByScore(scores: MemoryScore[]): number[] {
    return scores
      .sort((a, b) => b.score - a.score)
      .map(score => score.index);
  }

  /**
   * Create a new memory state with only the selected indices
   */
  private createPrunedState(
    originalState: IMemoryState,
    keepIndices: number[]
  ): IMemoryState {
    return tf.tidy(() => {
      const indices = tf.tensor1d(keepIndices, 'int32');
      
      return {
        shortTerm: tf.gather(originalState.shortTerm, indices) as tf.Tensor2D,
        longTerm: originalState.longTerm, // Keep original long-term memory
        meta: tf.gather(originalState.meta, indices) as tf.Tensor2D,
        timestamps: tf.gather(originalState.timestamps, indices) as tf.Tensor1D,
        accessCounts: tf.gather(originalState.accessCounts, indices) as tf.Tensor1D,
        surpriseHistory: tf.gather(originalState.surpriseHistory, indices) as tf.Tensor1D
      };
    });
  }

  /**
   * Distill pruned memories by averaging into long-term memory
   */
  private distillMemories(
    originalState: IMemoryState,
    pruneIndices: number[],
    currentLongTerm: ITensor
  ): ITensor {
    return tf.tidy(() => {
      if (pruneIndices.length === 0) {
        return currentLongTerm;
      }

      const indices = tf.tensor1d(pruneIndices, 'int32');
      const prunedMemories = tf.gather(originalState.shortTerm, indices) as tf.Tensor2D;
      
      // Calculate weighted average based on access counts
      const prunedAccessCounts = tf.gather(originalState.accessCounts, indices) as tf.Tensor1D;
      const weights = tf.div(
        prunedAccessCounts,
        tf.sum(prunedAccessCounts)
      ).expandDims(1);
      
      // Create distilled representation
      const distilledVector = tf.sum(tf.mul(prunedMemories, weights), 0);
      
      // Add to long-term memory (concatenate and truncate if needed)
      const currentLongTermData = currentLongTerm as tf.Tensor2D;
      const expanded = distilledVector.expandDims(0);
      const newLongTerm = tf.concat([currentLongTermData, expanded], 0);
      
      // If long-term memory exceeds capacity, remove oldest entries
      const maxLongTermSize = Math.floor(this.config.maxCapacity * 0.3); // 30% of total capacity
      if (newLongTerm.shape[0] > maxLongTermSize) {
        return tf.slice(newLongTerm, [newLongTerm.shape[0] - maxLongTermSize, 0], [-1, -1]);
      }
      
      return newLongTerm;
    });
  }

  /**
   * Get pruning statistics
   */
  getPruningStats(): {
    totalPrunings: number;
    averageReduction: number;
    lastPruningTime: number;
    timeSinceLastPruning: number;
  } {
    const totalPrunings = this.pruningHistory.length;
    const averageReduction = totalPrunings > 0 
      ? this.pruningHistory.reduce((a, b) => a + b, 0) / totalPrunings 
      : 0;
    const timeSinceLastPruning = Date.now() - this.lastPruningTime;

    return {
      totalPrunings,
      averageReduction,
      lastPruningTime: this.lastPruningTime,
      timeSinceLastPruning
    };
  }

  /**
   * Update pruning configuration
   */
  updateConfig(newConfig: Partial<PruningConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Reset pruning history and statistics
   */
  reset(): void {
    this.pruningHistory = [];
    this.lastPruningTime = 0;
  }

  /**
   * Validate memory state for consistency after pruning
   */
  validatePrunedState(state: IMemoryState): boolean {
    return tf.tidy(() => {
      try {
        const shortTermShape = state.shortTerm.shape;
        const longTermShape = state.longTerm.shape;
        const metaShape = state.meta.shape;
        
        // Check shape consistency
        const memoryCount = shortTermShape[0];
        const isConsistent = 
          state.timestamps.shape[0] === memoryCount &&
          state.accessCounts.shape[0] === memoryCount &&
          state.surpriseHistory.shape[0] === memoryCount &&
          metaShape[0] === memoryCount &&
          shortTermShape[1] === longTermShape[1] && // Same embedding dimension
          shortTermShape[1] === metaShape[1];

        // Check for NaN or infinite values
        const hasValidValues = [
          state.shortTerm,
          state.longTerm,
          state.meta,
          state.timestamps,
          state.accessCounts,
          state.surpriseHistory
        ].every(tensor => {
          const values = tensor.dataSync();
          return Array.from(values).every(v => isFinite(v));
        });

        return isConsistent && hasValidValues;
      } catch (error) {
        console.warn('Error validating pruned state:', error);
        return false;
      }
    });
  }
}

/**
 * Helper function to create a default pruning configuration
 */
export function createDefaultPruningConfig(): PruningConfig {
  return {
    keepPercentage: 0.7,
    minMemoriesToKeep: 100,
    maxCapacity: 5000,
    entropyWeight: 1.0,
    surpriseWeight: 1.2,
    redundancyWeight: 0.8,
    redundancyThreshold: 0.85,
    enableDistillation: true
  };
}

/**
 * Utility function to analyze memory quality before and after pruning
 */
export function analyzeMemoryQuality(
  originalState: IMemoryState,
  prunedState: IMemoryState
): {
  originalEntropy: number;
  prunedEntropy: number;
  entropyImprovement: number;
  originalSurprise: number;
  prunedSurprise: number;
  surpriseImprovement: number;
} {
  return tf.tidy(() => {
    // Calculate average entropy
    const originalEntropyData = tf.softmax(originalState.shortTerm as tf.Tensor2D, 1);
    const originalLogProbs = tf.log(tf.add(originalEntropyData, tf.scalar(1e-8)));
    const originalEntropy = tf.mean(tf.neg(tf.sum(tf.mul(originalEntropyData, originalLogProbs), 1))).dataSync()[0];

    const prunedEntropyData = tf.softmax(prunedState.shortTerm as tf.Tensor2D, 1);
    const prunedLogProbs = tf.log(tf.add(prunedEntropyData, tf.scalar(1e-8)));
    const prunedEntropy = tf.mean(tf.neg(tf.sum(tf.mul(prunedEntropyData, prunedLogProbs), 1))).dataSync()[0];

    // Calculate average surprise
    const originalSurprise = tf.mean(originalState.surpriseHistory).dataSync()[0];
    const prunedSurprise = tf.mean(prunedState.surpriseHistory).dataSync()[0];

    return {
      originalEntropy,
      prunedEntropy,
      entropyImprovement: (prunedEntropy - originalEntropy) / originalEntropy,
      originalSurprise,
      prunedSurprise,
      surpriseImprovement: (prunedSurprise - originalSurprise) / originalSurprise
    };
  });
}
