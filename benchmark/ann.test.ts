import * as tf from '@tensorflow/tfjs-node';
import { HNSW } from '../src/ann';

describe('ANN Benchmark Tests', () => {
  let testData: tf.Tensor2D[];
  let queryVector: tf.Tensor1D;
  let hnsw: HNSW;

  beforeEach(() => {
    // Generate test data for benchmarking
    testData = [];
    for (let i = 0; i < 5000; i++) {
      testData.push(tf.randomNormal([768]) as tf.Tensor2D);
    }
    queryVector = tf.randomNormal([768]);
    hnsw = new HNSW();
  });

  afterEach(() => {
    // Clean up tensors
    testData.forEach(tensor => tensor.dispose());
    queryVector.dispose();
  });

  test('HNSW vs Brute Force - Time Comparison', async () => {
    const topK = 10;

    // Benchmark brute force search
    const bruteForceStart = performance.now();
    const bruteForceResults = bruteForceSimilaritySearch(queryVector, testData, topK);
    const bruteForceTime = performance.now() - bruteForceStart;

    // Build HNSW index
    hnsw.buildIndex(testData);

    // Benchmark HNSW search
    const hnswStart = performance.now();
    const hnswResults = hnsw.search(queryVector, topK);
    const hnswTime = performance.now() - hnswStart;

    console.log(`Brute Force Time: ${bruteForceTime}ms`);
    console.log(`HNSW Time: ${hnswTime}ms`);
    console.log(`Speedup: ${bruteForceTime / hnswTime}x`);

    // HNSW should be significantly faster for large datasets
    expect(hnswTime).toBeLessThan(bruteForceTime);

    // Clean up results
    bruteForceResults.forEach(tensor => tensor.dispose());
    hnswResults.forEach(tensor => tensor.dispose());
  });

  test('HNSW vs Brute Force - Recall Comparison', async () => {
    const topK = 10;

    // Get ground truth from brute force
    const groundTruth = bruteForceSimilaritySearch(queryVector, testData, topK);
    
    // Build HNSW index and search
    hnsw.buildIndex(testData);
    const hnswResults = hnsw.search(queryVector, topK);

    // Calculate recall (intersection of results)
    const recall = calculateRecall(groundTruth, hnswResults);
    
    console.log(`Recall: ${recall * 100}%`);

    // HNSW should have reasonable recall (>80% for this test)
    expect(recall).toBeGreaterThan(0.8);

    // Clean up
    groundTruth.forEach(tensor => tensor.dispose());
    hnswResults.forEach(tensor => tensor.dispose());
  });

  test('needsRebuild logic', () => {
    // Test rebuild conditions
    expect(hnsw.needsRebuild(true, 3000)).toBe(true);
    expect(hnsw.needsRebuild(true, 1000)).toBe(false);
    expect(hnsw.needsRebuild(false, 3000)).toBe(false);
  });
});

function bruteForceSimilaritySearch(
  query: tf.Tensor1D, 
  data: tf.Tensor2D[], 
  topK: number
): tf.Tensor2D[] {
  const similarities: { index: number; similarity: number }[] = [];

  // Calculate similarities with all vectors
  data.forEach((vector, index) => {
    const similarity = tf.dot(query, vector.squeeze()).dataSync()[0];
    similarities.push({ index, similarity });
  });

  // Sort by similarity and return top K
  similarities.sort((a, b) => b.similarity - a.similarity);
  
  return similarities
    .slice(0, topK)
    .map(result => data[result.index]);
}

function calculateRecall(groundTruth: tf.Tensor2D[], hnswResults: tf.Tensor2D[]): number {
  const groundTruthHashes = new Set(
    groundTruth.map(tensor => tensor.dataSync().join(','))
  );
  
  const matches = hnswResults.filter(tensor =>
    groundTruthHashes.has(tensor.dataSync().join(','))
  ).length;
  
  return matches / groundTruth.length;
}
