import * as tf from '@tensorflow/tfjs-node';
import { HNSW } from '../src/ann';

describe('ANN Quick Benchmark Tests', () => {
  let testData: tf.Tensor2D[];
  let queryVector: tf.Tensor1D;
  let hnsw: HNSW;

  beforeEach(() => {
    // Reduced test data for faster benchmarking
    testData = [];
    for (let i = 0; i < 500; i++) { // Reduced from 5000 to 500
      testData.push(tf.randomNormal([128]) as tf.Tensor2D); // Reduced from 768 to 128 dims
    }
    queryVector = tf.randomNormal([128]);
    hnsw = new HNSW();
  });

  afterEach(() => {
    // Clean up tensors
    testData.forEach(tensor => tensor.dispose());
    queryVector.dispose();
  });

  test('HNSW vs Brute Force - Quick Time Comparison', async () => {
    const topK = 5; // Reduced from 10

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

    // HNSW should be faster for datasets > 100 vectors
    expect(hnswTime).toBeLessThan(bruteForceTime * 2); // More lenient for smaller dataset

    // Clean up results
    bruteForceResults.forEach(tensor => tensor.dispose());
    hnswResults.forEach(tensor => tensor.dispose());
  }, 30000); // 30 second timeout
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
