import * as tf from '@tensorflow/tfjs-node';
import { HNSW } from './dist/ann.js';

async function testANN() {
  console.log('Testing ANN implementation...');
  
  try {
    // Create test data
    const testData = [];
    for (let i = 0; i < 100; i++) {
      testData.push(tf.randomNormal([512]));
    }
    
    // Create query vector
    const query = tf.randomNormal([512]);
    
    // Test HNSW
    const hnsw = new HNSW();
    
    console.log('Building HNSW index...');
    await hnsw.buildIndex(testData);
    
    console.log('Performing search...');
    const results = await hnsw.search(query, 5);
    
    console.log(`Found ${results.length} results`);
    
    // Test needsRebuild
    console.log('Testing needsRebuild logic...');
    console.log('needsRebuild(true, 3000):', hnsw.needsRebuild(true, 3000)); // Should be true (memory changed + slot > 2000)
    console.log('needsRebuild(true, 1000):', hnsw.needsRebuild(true, 1000)); // Should be false (slot <= 2000)
    console.log('needsRebuild(false, 3000):', hnsw.needsRebuild(false, 3000)); // Should be false (no memory change)
    console.log('needsRebuild(false, 100):', hnsw.needsRebuild(false, 100)); // Should be false (no memory change + small slot)
    
    // Clean up
    testData.forEach(tensor => tensor.dispose());
    query.dispose();
    results.forEach(tensor => tensor.dispose());
    hnsw.dispose();
    
    console.log('ANN test completed successfully!');
  } catch (error) {
    console.error('ANN test failed:', error);
  }
}

testANN();
