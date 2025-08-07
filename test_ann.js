import { HNSW } from './standalone-ann.js';

let hnsw;
let testData;

async function testANN() {
  console.log('Testing ANN implementation...');
  hnsw = new HNSW();
  try {
    // Create test data
    testData = [];
    for (let i = 0; i < 100; i++) {
      testData.push(Array(512).fill(Math.random()));
    }

    // Create query vector
    const query = Array(512).fill(Math.random());

    // Test HNSW


    console.log('Building HNSW index...');
    await hnsw.buildIndex(testData);

    console.log('Performing search...');
    const results = await hnsw.search(query, 5);

    console.log(`Found ${results.length} results`);
    // Log distances of the results
    console.log('Distances of the results:');
    for (const result of results) {
      const distance = hnsw.computeDistance(query, result);
      console.log(distance);
    }

    // Test needsRebuild
    console.log('Testing needsRebuild logic...');
    console.log('needsRebuild(true, 3000):', hnsw.needsRebuild(true, 3000)); // Should be true (memory changed + slot > 2000)
    console.log('needsRebuild(true, 1000):', hnsw.needsRebuild(true, 1000)); // Should be false (slot <= 2000)
    console.log('needsRebuild(false, 3000):', hnsw.needsRebuild(false, 3000)); // Should be false (no memory change)
    console.log('needsRebuild(false, 100):', hnsw.needsRebuild(false, 100)); // Should be false (no memory change + small slot)

    // Clean up - No need to dispose as we are not using tensors
    console.log('ANN test completed successfully!');
    return;
  } catch (error) {
    console.error('ANN test failed:', error);
    throw error;
  }
}

testANN().then(() => {
  console.log('Test completed');
  hnsw.setLastSlotCount(testData.length); // Update lastSlotCount after the test
}).catch(function(err) {
  console.error("Error in testANN:", err);
});
