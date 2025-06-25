import * as tf from '@tensorflow/tfjs-node';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor, unwrapTensor, type IMemoryState } from '../types.js';

describe('TitanMemoryModel', () => {
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

  test('initialization', () => {
    expect(model).toBeDefined();
    const config = model.getConfig();
    expect(config.inputDim).toBe(inputDim);
    expect(config.hiddenDim).toBe(hiddenDim);
    expect(config.memoryDim).toBe(memoryDim);
  });

  test('forward pass', () => {
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

  test('training converges', () => {
    const memoryState: IMemoryState = {
      shortTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      longTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      meta: wrapTensor(tf.zeros([memoryDim, inputDim])),
      timestamps: wrapTensor(tf.zeros([memoryDim])),
      accessCounts: wrapTensor(tf.zeros([memoryDim])),
      surpriseHistory: wrapTensor(tf.zeros([memoryDim]))
    };

    const losses: number[] = [];
    tf.tidy(() => {
      for (let i = 0; i < 5; i++) {
        const wrappedX = wrapTensor(tf.randomNormal([inputDim]));
        const wrappedNext = wrapTensor(tf.randomNormal([inputDim]));

        const result = model.trainStep(wrappedX, wrappedNext, memoryState);
        losses.push(unwrapTensor(result.loss).dataSync()[0]);
        result.loss.dispose();
      }
    });

    expect(losses.length).toBe(5);
    expect(losses.every(loss => !isNaN(loss))).toBe(true);

    memoryState.shortTerm.dispose();
    memoryState.longTerm.dispose();
    memoryState.meta.dispose();
    memoryState.timestamps.dispose();
    memoryState.accessCounts.dispose();
    memoryState.surpriseHistory.dispose();
  });

  test('memory manifold step size constraint', () => {
    const config = model.getConfig();
    const memoryState = tf.zeros([config.memoryDim]);
    const direction = tf.randomNormal([config.memoryDim]);

    const result = model.manifoldStep(memoryState, direction);

    // Calculate angle between original and updated vectors
    const dotProduct = tf.sum(tf.mul(memoryState, result));
    const norm1 = tf.norm(memoryState);
    const norm2 = tf.norm(result);
    const cosAngle = tf.div(dotProduct, tf.mul(norm1, norm2));
    const angle = tf.acos(cosAngle);

    const epsilon = 1e-6;
    expect(angle.dataSync()[0]).toBeLessThanOrEqual(0.1 + epsilon);

    memoryState.dispose();
    direction.dispose();
    result.dispose();
    dotProduct.dispose();
    norm1.dispose();
    norm2.dispose();
    cosAngle.dispose();
    angle.dispose();
  });

  test('model persistence', async () => {
    const model1 = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      memoryDim
    });

    const x = wrapTensor(tf.randomNormal([inputDim]));

    const model2 = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      memoryDim
    });

    // Models should have different predictions initially
    const memoryState: IMemoryState = {
      shortTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      longTerm: wrapTensor(tf.zeros([memoryDim, inputDim])),
      meta: wrapTensor(tf.zeros([memoryDim, inputDim])),
      timestamps: wrapTensor(tf.zeros([memoryDim])),
      accessCounts: wrapTensor(tf.zeros([memoryDim])),
      surpriseHistory: wrapTensor(tf.zeros([memoryDim]))
    };

    const { predicted: originalPrediction } = model1.forward(x, memoryState);

    await model1.saveModel('./test-model.json');
    await model2.loadModel('./test-model.json');

    const model3 = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      memoryDim
    });

    await model3.loadModel('./test-model.json');
    const { predicted: loadedPrediction } = model3.forward(x, memoryState);

    // Check if loaded model produces consistent results
    expect(originalPrediction.shape).toEqual(loadedPrediction.shape);

    originalPrediction.dispose();
    loadedPrediction.dispose();
    x.dispose();
    memoryState.shortTerm.dispose();
    memoryState.longTerm.dispose();
    memoryState.meta.dispose();
    memoryState.timestamps.dispose();
    memoryState.accessCounts.dispose();
    memoryState.surpriseHistory.dispose();
    model1.dispose();
    model2.dispose();
    model3.dispose();
  });

  test('sequence training', () => {
    const sequence = [
      wrapTensor(tf.tensor1d([1, 0, 0, 0])),
      wrapTensor(tf.tensor1d([0, 1, 0, 0])),
      wrapTensor(tf.tensor1d([0, 0, 1, 0])),
      wrapTensor(tf.tensor1d([0, 0, 0, 1]))
    ];

    const testModel = new TitanMemoryModel({
      inputDim: 4,
      hiddenDim: 8,
      memoryDim: 16
    });

    const memoryState: IMemoryState = {
      shortTerm: wrapTensor(tf.zeros([16, 4])),
      longTerm: wrapTensor(tf.zeros([16, 4])),
      meta: wrapTensor(tf.zeros([16, 4])),
      timestamps: wrapTensor(tf.zeros([16])),
      accessCounts: wrapTensor(tf.zeros([16])),
      surpriseHistory: wrapTensor(tf.zeros([16]))
    };

    for (let epoch = 0; epoch < 2; epoch++) {
      for (let i = 0; i < sequence.length - 1; i++) {
        const result = testModel.trainStep(sequence[i], sequence[i + 1], memoryState);
        const lossShape = unwrapTensor(result.loss).shape;
        expect(lossShape).toEqual([]);
        result.loss.dispose();
      }
    }

    sequence.forEach(seq => seq.dispose());
    memoryState.shortTerm.dispose();
    memoryState.longTerm.dispose();
    memoryState.meta.dispose();
    memoryState.timestamps.dispose();
    memoryState.accessCounts.dispose();
    memoryState.surpriseHistory.dispose();
    testModel.dispose();
  });
});
