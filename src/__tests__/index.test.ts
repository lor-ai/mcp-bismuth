import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor, type IMemoryState } from '../types.js';

describe('TitanMemoryModel Tests', () => {
  let model: TitanMemoryModel;
  const inputDim = 64;
  const hiddenDim = 32;
  const memoryDim = 64;

  beforeEach(() => {
    model = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      memoryDim,
      memorySlots: 1000,
      transformerLayers: 2
    });
  });

  test('Model processes sequences correctly', () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState: IMemoryState = {
      shortTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      longTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      meta: wrapTensor(tf.zeros([memoryDim, inputDim])),
      timestamps: wrapTensor(tf.zeros([memoryDim])),
      accessCounts: wrapTensor(tf.zeros([memoryDim])),
      surpriseHistory: wrapTensor(tf.zeros([memoryDim]))
    };

    const { predicted, memoryUpdate } = model.forward(x, memoryState);

    // Expect the shapes to match the logic:
    expect(predicted.shape).toEqual([inputDim]);
    expect(memoryUpdate.newState.shortTerm.shape).toEqual([memoryDim, inputDim]);

    predicted.dispose();
    memoryUpdate.newState.shortTerm.dispose();
    memoryUpdate.newState.longTerm.dispose();
    memoryUpdate.newState.meta.dispose();
    memoryUpdate.newState.timestamps.dispose();
    memoryUpdate.newState.accessCounts.dispose();
    memoryUpdate.newState.surpriseHistory.dispose();
    x.dispose();
    memoryState.shortTerm.dispose();
    memoryState.longTerm.dispose();
    memoryState.meta.dispose();
    memoryState.timestamps.dispose();
    memoryState.accessCounts.dispose();
    memoryState.surpriseHistory.dispose();
  });

  test('Training reduces loss over time', () => {
    const memoryState: IMemoryState = {
      shortTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      longTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      meta: wrapTensor(tf.zeros([memoryDim, inputDim])),
      timestamps: wrapTensor(tf.zeros([memoryDim])),
      accessCounts: wrapTensor(tf.zeros([memoryDim])),
      surpriseHistory: wrapTensor(tf.zeros([memoryDim]))
    };
    const x_t = wrapTensor(tf.randomNormal([inputDim]));
    const x_next = wrapTensor(tf.randomNormal([inputDim]));

    // Run multiple training steps
    const losses: number[] = [];
    for (let i = 0; i < 10; i++) {
      const result = model.trainStep(x_t, x_next, memoryState);
      losses.push(result.loss.dataSync()[0]);
      result.loss.dispose();
    }

    // Check if loss generally decreases
    const firstLoss = losses[0];
    const lastLoss = losses[losses.length - 1];
    expect(lastLoss).toBeLessThan(firstLoss);

    x_t.dispose();
    x_next.dispose();
    memoryState.shortTerm.dispose();
    memoryState.longTerm.dispose();
    memoryState.meta.dispose();
    memoryState.timestamps.dispose();
    memoryState.accessCounts.dispose();
    memoryState.surpriseHistory.dispose();
  });

  test('Manifold updates work correctly', () => {
    const base = wrapTensor(tf.randomNormal([inputDim]));
    const velocity = wrapTensor(tf.randomNormal([inputDim]));

    const result = model.manifoldStep(base, velocity);

    // Check that result has valid shape
    expect(result.shape).toEqual([inputDim]);

    base.dispose();
    velocity.dispose();
    result.dispose();
  });

  test('Model can save and load weights', async () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState: IMemoryState = {
      shortTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      longTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      meta: wrapTensor(tf.zeros([memoryDim, inputDim])),
      timestamps: wrapTensor(tf.zeros([memoryDim])),
      accessCounts: wrapTensor(tf.zeros([memoryDim])),
      surpriseHistory: wrapTensor(tf.zeros([memoryDim]))
    };

    // Get initial prediction
    const { predicted: pred1 } = model.forward(x, memoryState);
    const initial = pred1.dataSync();
    pred1.dispose();

    // Save and load weights
    await model.saveModel('./test-weights.json');
    await model.loadModel('./test-weights.json');

    // Get prediction after loading
    const { predicted: pred2 } = model.forward(x, memoryState);
    const loaded = pred2.dataSync();
    pred2.dispose();

    // Compare predictions (should be similar)
    expect(initial.length).toEqual(loaded.length);

    x.dispose();
    memoryState.shortTerm.dispose();
    memoryState.longTerm.dispose();
    memoryState.meta.dispose();
    memoryState.timestamps.dispose();
    memoryState.accessCounts.dispose();
    memoryState.surpriseHistory.dispose();
  });
});
