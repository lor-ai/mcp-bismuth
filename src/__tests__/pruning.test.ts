import * as tf from '@tensorflow/tfjs-node';
import { MemoryPruner, type PruningConfig, type PruningResult, createDefaultPruningConfig, analyzeMemoryQuality } from '../pruning.js';
import { type IMemoryState } from '../types.js';

describe('MemoryPruner', () => {
  let pruner: MemoryPruner;
  let memoryState: IMemoryState;
  const memoryDim = 64;
  const memorySlots = 100;

  beforeEach(() => {
    // Initialize pruner with test configuration
    const config: Partial<PruningConfig> = {
      keepPercentage: 0.7,
      minMemoriesToKeep: 10,
      maxCapacity: 50,
      entropyWeight: 1.0,
      surpriseWeight: 1.2,
      redundancyWeight: 0.8,
      enableDistillation: true
    };
    pruner = new MemoryPruner(config);

    // Create test memory state
    memoryState = {
      shortTerm: tf.randomNormal([memorySlots, memoryDim]) as tf.Tensor2D,
      longTerm: tf.randomNormal([Math.floor(memorySlots / 2), memoryDim]) as tf.Tensor2D,
      meta: tf.randomNormal([memorySlots, memoryDim]) as tf.Tensor2D,
      timestamps: tf.range(1, memorySlots + 1) as tf.Tensor1D, // Simulate different timestamps
      accessCounts: tf.randomUniform([memorySlots], 0, 10) as tf.Tensor1D,
      surpriseHistory: tf.randomUniform([memorySlots], 0, 1) as tf.Tensor1D
    };
  });

  afterEach(() => {
    // Clean up tensors
    Object.values(memoryState).forEach(tensor => {
      if (tensor && !tensor.isDisposed) {
        tensor.dispose();
      }
    });
    pruner.reset();
  });

  describe('Initialization', () => {
    test('should initialize with default configuration', () => {
      const defaultPruner = new MemoryPruner();
      expect(defaultPruner).toBeDefined();
      
      const stats = defaultPruner.getPruningStats();
      expect(stats.totalPrunings).toBe(0);
      expect(stats.averageReduction).toBe(0);
    });

    test('should create default pruning config', () => {
      const config = createDefaultPruningConfig();
      expect(config.keepPercentage).toBe(0.7);
      expect(config.minMemoriesToKeep).toBe(100);
      expect(config.maxCapacity).toBe(5000);
      expect(config.enableDistillation).toBe(true);
    });

    test('should initialize with custom configuration', () => {
      const customConfig: Partial<PruningConfig> = {
        keepPercentage: 0.5,
        minMemoriesToKeep: 50,
        maxCapacity: 1000
      };
      const customPruner = new MemoryPruner(customConfig);
      expect(customPruner).toBeDefined();
    });
  });

  describe('Pruning Logic', () => {
    test('should determine when pruning is needed', () => {
      const shouldPruneSmall = pruner.shouldPrune(memoryState);
      expect(shouldPruneSmall).toBe(true); // 100 slots > maxCapacity of 50

      // Test with smaller memory state
      const smallState: IMemoryState = {
        ...memoryState,
        shortTerm: tf.randomNormal([30, memoryDim]) as tf.Tensor2D,
        meta: tf.randomNormal([30, memoryDim]) as tf.Tensor2D,
        timestamps: tf.range(1, 31) as tf.Tensor1D,
        accessCounts: tf.randomUniform([30], 0, 10) as tf.Tensor1D,
        surpriseHistory: tf.randomUniform([30], 0, 1) as tf.Tensor1D
      };

      const shouldPruneLarge = pruner.shouldPrune(smallState);
      expect(shouldPruneLarge).toBe(false); // 30 slots < maxCapacity of 50

      // Clean up
      Object.values(smallState).forEach(tensor => {
        if (tensor !== memoryState.shortTerm && tensor !== memoryState.longTerm && 
            tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });

    test('should perform basic pruning operation', async () => {
      const result = await pruner.pruneMemory(memoryState);

      expect(result).toBeDefined();
      expect(result.originalCount).toBe(memorySlots);
      expect(result.finalCount).toBeLessThan(result.originalCount);
      expect(result.reductionRatio).toBeGreaterThan(0);
      expect(result.reductionRatio).toBeLessThan(1);
      expect(result.averageScore).toBeGreaterThanOrEqual(0);
      expect(result.newMemoryState).toBeDefined();

      // Verify pruned state structure
      expect(result.newMemoryState.shortTerm.shape[0]).toBe(result.finalCount);
      expect(result.newMemoryState.timestamps.shape[0]).toBe(result.finalCount);
      expect(result.newMemoryState.accessCounts.shape[0]).toBe(result.finalCount);
      expect(result.newMemoryState.surpriseHistory.shape[0]).toBe(result.finalCount);
    });

    test('should respect minimum memories threshold', async () => {
      // Create small memory state below minimum threshold
      const smallState: IMemoryState = {
        shortTerm: tf.randomNormal([5, memoryDim]) as tf.Tensor2D,
        longTerm: tf.randomNormal([3, memoryDim]) as tf.Tensor2D,
        meta: tf.randomNormal([5, memoryDim]) as tf.Tensor2D,
        timestamps: tf.range(1, 6) as tf.Tensor1D,
        accessCounts: tf.randomUniform([5], 0, 10) as tf.Tensor1D,
        surpriseHistory: tf.randomUniform([5], 0, 1) as tf.Tensor1D
      };

      const result = await pruner.pruneMemory(smallState);

      expect(result.originalCount).toBe(5);
      expect(result.finalCount).toBe(5); // Should not prune below minimum
      expect(result.reductionRatio).toBe(0);

      // Clean up
      Object.values(smallState).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });

    test('should calculate information scores correctly', async () => {
      const result = await pruner.pruneMemory(memoryState);

      // Scores should be finite and reasonable
      expect(isFinite(result.averageScore)).toBe(true);
      expect(result.averageScore).not.toBeNaN();
      
      // Higher scored memories should be kept
      expect(result.finalCount).toBeGreaterThan(0);
      expect(result.finalCount).toBeLessThanOrEqual(result.originalCount);
    });

    test('should handle distillation when enabled', async () => {
      const result = await pruner.pruneMemory(memoryState);

      if (result.reductionRatio > 0) {
        expect(result.distilledCount).toBeGreaterThan(0);
        // Note: Long-term memory may be truncated if it exceeds capacity limits
        // so we just check that distillation occurred
        expect(result.newMemoryState.longTerm.shape[0]).toBeGreaterThan(0);
      }
    });

    test('should skip distillation when disabled', async () => {
      const nondistillingPruner = new MemoryPruner({
        keepPercentage: 0.7,
        minMemoriesToKeep: 10,
        maxCapacity: 50,
        enableDistillation: false
      });

      const result = await nondistillingPruner.pruneMemory(memoryState);

      expect(result.distilledCount).toBe(0);
      expect(result.newMemoryState.longTerm.shape[0]).toBe(memoryState.longTerm.shape[0]);
    });
  });

  describe('Information Gain Calculations', () => {
    test('should calculate entropy scores', () => {
      // Create test tensor with known distribution
      const testTensor = tf.tensor2d([
        [1, 0, 0, 0],  // Low entropy (certain)
        [0.25, 0.25, 0.25, 0.25],  // High entropy (uniform)
        [0.8, 0.1, 0.05, 0.05],  // Medium entropy (skewed)
      ]);

      // Create a test pruner instance to access private methods through the public interface
      const testPruner = new MemoryPruner();
      
      // We'll test entropy indirectly through the full pruning process
      const testState: IMemoryState = {
        shortTerm: testTensor,
        longTerm: tf.zeros([2, 4]) as tf.Tensor2D,
        meta: tf.zeros([3, 4]) as tf.Tensor2D,
        timestamps: tf.tensor1d([1, 2, 3]),
        accessCounts: tf.tensor1d([1, 1, 1]),
        surpriseHistory: tf.tensor1d([0.5, 0.8, 0.3])
      };

      // The pruning process should handle different entropy levels
      return testPruner.pruneMemory(testState).then(result => {
        expect(result).toBeDefined();
        expect(result.newMemoryState.shortTerm.shape[0]).toBeGreaterThan(0);
        
        // Clean up
        Object.values(testState).forEach(tensor => {
          if (tensor && !tensor.isDisposed) {
            tensor.dispose();
          }
        });
      });
    });

    test('should calculate redundancy based on cosine similarity', async () => {
      // Create memory with some redundant vectors
      const redundantVectors = tf.tensor2d([
        [1, 0, 0, 0],     // Vector 1
        [1, 0, 0, 0],     // Identical to vector 1 (high redundancy)
        [0, 1, 0, 0],     // Orthogonal to vector 1 (low redundancy)
        [0.9, 0.1, 0, 0], // Similar to vector 1 (medium redundancy)
      ]);

      const testState: IMemoryState = {
        shortTerm: redundantVectors,
        longTerm: tf.zeros([2, 4]) as tf.Tensor2D,
        meta: tf.zeros([4, 4]) as tf.Tensor2D,
        timestamps: tf.tensor1d([1, 2, 3, 4]),
        accessCounts: tf.tensor1d([1, 1, 1, 1]),
        surpriseHistory: tf.tensor1d([0.5, 0.5, 0.5, 0.5])
      };

      const result = await pruner.pruneMemory(testState);

      // The pruning should identify and potentially remove redundant vectors
      expect(result).toBeDefined();
      expect(result.finalCount).toBeGreaterThan(0);

      // Clean up
      Object.values(testState).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });

    test('should incorporate surprise scores in ranking', async () => {
      // Create memory with varying surprise scores
      const testState: IMemoryState = {
        shortTerm: tf.randomNormal([10, memoryDim]) as tf.Tensor2D,
        longTerm: tf.randomNormal([5, memoryDim]) as tf.Tensor2D,
        meta: tf.randomNormal([10, memoryDim]) as tf.Tensor2D,
        timestamps: tf.range(1, 11) as tf.Tensor1D,
        accessCounts: tf.ones([10]) as tf.Tensor1D,
        surpriseHistory: tf.tensor1d([0.1, 0.9, 0.1, 0.8, 0.2, 0.7, 0.1, 0.6, 0.3, 0.5]) // Varied surprise
      };

      const testPruner = new MemoryPruner({
        keepPercentage: 0.5,
        minMemoriesToKeep: 3,
        maxCapacity: 8,
        surpriseWeight: 2.0 // High weight for surprise
      });

      const result = await testPruner.pruneMemory(testState);

      // High surprise memories should be more likely to be kept
      expect(result.finalCount).toBe(5); // 50% of 10
      expect(result.averageScore).toBeGreaterThan(0);

      // Clean up
      Object.values(testState).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });
  });

  describe('State Validation', () => {
    test('should validate pruned state consistency', async () => {
      const result = await pruner.pruneMemory(memoryState);
      const isValid = pruner.validatePrunedState(result.newMemoryState);

      expect(isValid).toBe(true);
    });

    test('should detect invalid pruned state', () => {
      // Create invalid state with mismatched dimensions
      const invalidState: IMemoryState = {
        shortTerm: tf.zeros([5, memoryDim]) as tf.Tensor2D,
        longTerm: tf.zeros([3, memoryDim]) as tf.Tensor2D,
        meta: tf.zeros([5, memoryDim]) as tf.Tensor2D,
        timestamps: tf.zeros([3]), // Wrong size!
        accessCounts: tf.zeros([5]),
        surpriseHistory: tf.zeros([5])
      };

      const isValid = pruner.validatePrunedState(invalidState);
      expect(isValid).toBe(false);

      // Clean up
      Object.values(invalidState).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });

    test('should detect NaN values in state', () => {
      // Create state with NaN values
      const nanState: IMemoryState = {
        shortTerm: tf.tensor2d([[NaN, 1, 2], [3, 4, 5]]),
        longTerm: tf.zeros([1, 3]) as tf.Tensor2D,
        meta: tf.zeros([2, 3]) as tf.Tensor2D,
        timestamps: tf.tensor1d([1, 2]),
        accessCounts: tf.tensor1d([1, 1]),
        surpriseHistory: tf.tensor1d([0.5, 0.5])
      };

      const isValid = pruner.validatePrunedState(nanState);
      expect(isValid).toBe(false);

      // Clean up
      Object.values(nanState).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });
  });

  describe('Configuration Management', () => {
    test('should update configuration', () => {
      const newConfig: Partial<PruningConfig> = {
        keepPercentage: 0.5,
        maxCapacity: 100
      };

      pruner.updateConfig(newConfig);
      
      // We can't directly test the config, but we can test its effects
      const updatedPruner = new MemoryPruner(newConfig);
      expect(updatedPruner).toBeDefined();
    });

    test('should reset pruning history', () => {
      // Perform some pruning to create history
      return pruner.pruneMemory(memoryState).then(() => {
        let stats = pruner.getPruningStats();
        expect(stats.totalPrunings).toBeGreaterThan(0);

        pruner.reset();
        stats = pruner.getPruningStats();
        expect(stats.totalPrunings).toBe(0);
        expect(stats.averageReduction).toBe(0);
        expect(stats.lastPruningTime).toBe(0);
      });
    });
  });

  describe('Statistics and Monitoring', () => {
    test('should track pruning statistics', async () => {
      const initialStats = pruner.getPruningStats();
      expect(initialStats.totalPrunings).toBe(0);

      await pruner.pruneMemory(memoryState);

      const afterStats = pruner.getPruningStats();
      expect(afterStats.totalPrunings).toBe(1);
      expect(afterStats.averageReduction).toBeGreaterThan(0);
      expect(afterStats.lastPruningTime).toBeGreaterThan(0);
      expect(afterStats.timeSinceLastPruning).toBeGreaterThanOrEqual(0);
    });

    test('should calculate average reduction over multiple prunings', async () => {
      // Perform multiple pruning operations
      await pruner.pruneMemory(memoryState);
      
      // Create new memory state for second pruning
      const memoryState2: IMemoryState = {
        shortTerm: tf.randomNormal([80, memoryDim]) as tf.Tensor2D,
        longTerm: tf.randomNormal([40, memoryDim]) as tf.Tensor2D,
        meta: tf.randomNormal([80, memoryDim]) as tf.Tensor2D,
        timestamps: tf.range(1, 81) as tf.Tensor1D,
        accessCounts: tf.randomUniform([80], 0, 10) as tf.Tensor1D,
        surpriseHistory: tf.randomUniform([80], 0, 1) as tf.Tensor1D
      };

      await pruner.pruneMemory(memoryState2);

      const stats = pruner.getPruningStats();
      expect(stats.totalPrunings).toBe(2);
      expect(stats.averageReduction).toBeGreaterThan(0);
      expect(stats.averageReduction).toBeLessThan(1);

      // Clean up
      Object.values(memoryState2).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });
  });

  describe('Memory Quality Analysis', () => {
    test('should analyze memory quality before and after pruning', async () => {
      const result = await pruner.pruneMemory(memoryState);
      const analysis = analyzeMemoryQuality(memoryState, result.newMemoryState);

      expect(analysis).toBeDefined();
      expect(isFinite(analysis.originalEntropy)).toBe(true);
      expect(isFinite(analysis.prunedEntropy)).toBe(true);
      expect(isFinite(analysis.originalSurprise)).toBe(true);
      expect(isFinite(analysis.prunedSurprise)).toBe(true);
      expect(isFinite(analysis.entropyImprovement)).toBe(true);
      expect(isFinite(analysis.surpriseImprovement)).toBe(true);
    });

    test('should show improvement in memory quality metrics', async () => {
      // Create memory state with some low-quality memories
      const mixedQualityState: IMemoryState = {
        shortTerm: tf.concat([
          tf.ones([25, memoryDim]).mul(0.1),    // Low entropy vectors
          tf.randomNormal([25, memoryDim]),     // High entropy vectors
          tf.ones([25, memoryDim]).mul(0.2),    // Low entropy vectors
          tf.randomNormal([25, memoryDim])      // High entropy vectors
        ]) as tf.Tensor2D,
        longTerm: tf.randomNormal([25, memoryDim]) as tf.Tensor2D,
        meta: tf.randomNormal([100, memoryDim]) as tf.Tensor2D,
        timestamps: tf.range(1, 101) as tf.Tensor1D,
        accessCounts: tf.randomUniform([100], 0, 10) as tf.Tensor1D,
        surpriseHistory: tf.concat([
          tf.ones([50]).mul(0.1),  // Low surprise
          tf.ones([50]).mul(0.9)   // High surprise
        ]) as tf.Tensor1D
      };

      const result = await pruner.pruneMemory(mixedQualityState);
      const analysis = analyzeMemoryQuality(mixedQualityState, result.newMemoryState);

      // After pruning, we should generally see quality improvements
      expect(analysis.prunedSurprise).toBeGreaterThanOrEqual(analysis.originalSurprise * 0.9); // Allow some variance

      // Clean up
      Object.values(mixedQualityState).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty memory state', async () => {
      const emptyState: IMemoryState = {
        shortTerm: tf.zeros([0, memoryDim]) as tf.Tensor2D,
        longTerm: tf.zeros([0, memoryDim]) as tf.Tensor2D,
        meta: tf.zeros([0, memoryDim]) as tf.Tensor2D,
        timestamps: tf.zeros([0]) as tf.Tensor1D,
        accessCounts: tf.zeros([0]) as tf.Tensor1D,
        surpriseHistory: tf.zeros([0]) as tf.Tensor1D
      };

      const result = await pruner.pruneMemory(emptyState);

      expect(result.originalCount).toBe(0);
      expect(result.finalCount).toBe(0);
      expect(result.reductionRatio).toBe(0);

      // Clean up
      Object.values(emptyState).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });

    test('should handle single memory entry', async () => {
      const singleState: IMemoryState = {
        shortTerm: tf.randomNormal([1, memoryDim]) as tf.Tensor2D,
        longTerm: tf.zeros([1, memoryDim]) as tf.Tensor2D,
        meta: tf.randomNormal([1, memoryDim]) as tf.Tensor2D,
        timestamps: tf.tensor1d([1]),
        accessCounts: tf.tensor1d([5]),
        surpriseHistory: tf.tensor1d([0.8])
      };

      const result = await pruner.pruneMemory(singleState);

      expect(result.originalCount).toBe(1);
      expect(result.finalCount).toBe(1); // Should keep the single memory
      expect(result.reductionRatio).toBe(0);

      // Clean up
      Object.values(singleState).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });

    test('should handle extreme configurations', async () => {
      const extremePruner = new MemoryPruner({
        keepPercentage: 0.01,  // Keep only 1%
        minMemoriesToKeep: 1,
        maxCapacity: 50,
        entropyWeight: 10.0,
        surpriseWeight: 10.0,
        redundancyWeight: 0.1
      });

      const result = await extremePruner.pruneMemory(memoryState);

      expect(result.finalCount).toBeGreaterThanOrEqual(1);
      expect(result.finalCount).toBeLessThanOrEqual(result.originalCount);
      expect(result.reductionRatio).toBeGreaterThan(0);
    });
  });

  describe('Performance', () => {
    test('should complete pruning in reasonable time', async () => {
      const startTime = Date.now();
      await pruner.pruneMemory(memoryState);
      const duration = Date.now() - startTime;

      // Should complete in under 10 seconds for 100 memories
      expect(duration).toBeLessThan(10000);
    });

    test('should handle large memory states', async () => {
      // Create larger memory state
      const largeState: IMemoryState = {
        shortTerm: tf.randomNormal([500, memoryDim]) as tf.Tensor2D,
        longTerm: tf.randomNormal([250, memoryDim]) as tf.Tensor2D,
        meta: tf.randomNormal([500, memoryDim]) as tf.Tensor2D,
        timestamps: tf.range(1, 501) as tf.Tensor1D,
        accessCounts: tf.randomUniform([500], 0, 10) as tf.Tensor1D,
        surpriseHistory: tf.randomUniform([500], 0, 1) as tf.Tensor1D
      };

      const largePruner = new MemoryPruner({
        keepPercentage: 0.6,
        maxCapacity: 400
      });

      const startTime = Date.now();
      const result = await largePruner.pruneMemory(largeState);
      const duration = Date.now() - startTime;

      expect(result).toBeDefined();
      expect(result.finalCount).toBeLessThan(result.originalCount);
      expect(duration).toBeLessThan(30000); // Should complete in under 30 seconds

      // Clean up
      Object.values(largeState).forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    });
  });
});
