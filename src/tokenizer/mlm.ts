/**
 * @fileoverview Masked Language Model (MLM) head for self-supervised pretraining
 * Implements BERT-style masked language modeling with configurable masking strategies
 */

import * as tf from '@tensorflow/tfjs-node';
import type { BPETokenizer } from './bpe.js';
import type { TokenEmbedding } from './embedding.js';

export interface MLMConfig {
  hiddenSize: number;
  vocabSize: number;
  maskProbability?: number;
  replaceProbability?: number;
  randomProbability?: number;
  maxSequenceLength?: number;
  labelSmoothingEpsilon?: number;
}

export interface MLMOutput {
  logits: tf.Tensor;
  loss: tf.Tensor;
  maskedPositions: tf.Tensor;
  accuracy: tf.Tensor;
}

export interface MLMBatch {
  inputIds: tf.Tensor2D;
  attentionMask: tf.Tensor2D;
  labels: tf.Tensor2D;
  maskedPositions: tf.Tensor2D;
}

export class MaskedLanguageModelHead {
  private config: MLMConfig;
  private projectionLayer!: tf.LayersModel;
  private outputLayer!: tf.LayersModel;
  private isInitialized = false;
  private tokenizer: BPETokenizer;
  private embedding: TokenEmbedding;

  constructor(
    config: MLMConfig,
    tokenizer: BPETokenizer,
    embedding: TokenEmbedding
  ) {
    this.config = {
      maskProbability: 0.15,
      replaceProbability: 0.8,  // 80% of masked tokens become [MASK]
      randomProbability: 0.1,   // 10% become random tokens
      maxSequenceLength: 512,
      labelSmoothingEpsilon: 0.1,
      ...config
    };

    this.tokenizer = tokenizer;
    this.embedding = embedding;
    this.initializeLayers();
  }

  /**
   * Initialize the MLM head layers
   */
  private initializeLayers(): void {
    // Projection layer to transform hidden states
    this.projectionLayer = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [this.config.hiddenSize],
          units: this.config.hiddenSize,
          activation: 'gelu',
          kernelInitializer: 'glorotNormal',
          name: 'mlm_projection'
        }),
        tf.layers.layerNormalization({
          name: 'mlm_layer_norm'
        })
      ],
      name: 'mlm_projection_layer'
    });

    // Output layer to predict vocabulary
    this.outputLayer = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [this.config.hiddenSize],
          units: this.config.vocabSize,
          activation: 'linear',
          kernelInitializer: 'glorotNormal',
          useBias: true,
          name: 'mlm_output'
        })
      ],
      name: 'mlm_output_layer'
    });

    this.isInitialized = true;
    console.log(`Initialized MLM head for vocab size ${this.config.vocabSize}`);
  }

  /**
   * Create masks for MLM training
   * @param inputIds Input token IDs
   * @returns Object with masked inputs and labels
   */
  public createMasks(inputIds: tf.Tensor2D): {
    maskedInputIds: tf.Tensor2D;
    labels: tf.Tensor2D;
    maskedPositions: tf.Tensor2D;
  } {
    return tf.tidy(() => {
      const [batchSize, seqLength] = inputIds.shape;
      
      // Special token IDs
      const padId = this.tokenizer.getSpecialTokenId('PAD');
      const maskId = this.tokenizer.getSpecialTokenId('MASK');
      const clsId = this.tokenizer.getSpecialTokenId('CLS');
      const sepId = this.tokenizer.getSpecialTokenId('SEP');
      
      // Create attention mask (1 for real tokens, 0 for padding)
      const attentionMask = tf.notEqual(inputIds, tf.scalar(padId));
      
      // Don't mask special tokens
      const specialTokenMask = tf.logicalOr(
        tf.logicalOr(
          tf.equal(inputIds, tf.scalar(clsId)),
          tf.equal(inputIds, tf.scalar(sepId))
        ),
        tf.equal(inputIds, tf.scalar(padId))
      );

      // Create random mask for masking probability
      const randomMask = tf.randomUniform([batchSize, seqLength]);
      const shouldMask = tf.logicalAnd(
        tf.less(randomMask, tf.scalar(this.config.maskProbability!)),
        tf.logicalNot(specialTokenMask)
      );

      // Create labels (copy of original input IDs)
      const labels = tf.where(
        shouldMask,
        inputIds,
        tf.fill([batchSize, seqLength], -100) // -100 will be ignored in loss calculation
      );

      // Create masked input IDs
      let maskedInputIds = tf.clone(inputIds);

      // 80% of masked tokens become [MASK]
      const maskTokens = tf.logicalAnd(
        shouldMask,
        tf.less(tf.randomUniform([batchSize, seqLength]), tf.scalar(this.config.replaceProbability!))
      );
      maskedInputIds = tf.where(
        maskTokens,
        tf.fill([batchSize, seqLength], maskId),
        maskedInputIds
      );

      // 10% of masked tokens become random tokens
      const randomTokens = tf.logicalAnd(
        tf.logicalAnd(shouldMask, tf.logicalNot(maskTokens)),
        tf.less(
          tf.randomUniform([batchSize, seqLength]), 
          tf.scalar(this.config.randomProbability! / (1 - this.config.replaceProbability!))
        )
      );
      const randomIds = tf.randomUniform(
        [batchSize, seqLength], 
        0, 
        this.config.vocabSize, 
        'int32'
      );
      maskedInputIds = tf.where(randomTokens, randomIds, maskedInputIds) as tf.Tensor2D;

      // Convert boolean mask to float for positions
      const maskedPositions = tf.cast(shouldMask, 'float32');

      return {
        maskedInputIds: maskedInputIds as tf.Tensor2D,
        labels: labels as tf.Tensor2D,
        maskedPositions: maskedPositions as tf.Tensor2D
      };
    });
  }

  /**
   * Forward pass through MLM head
   * @param hiddenStates Hidden states from transformer [batch_size, seq_length, hidden_size]
   * @param labels Target token IDs for loss calculation
   * @returns MLM output with logits and loss
   */
  public forward(
    hiddenStates: tf.Tensor3D,
    labels?: tf.Tensor2D
  ): MLMOutput {
    if (!this.isInitialized) {
      throw new Error('MLM head not initialized');
    }

    return tf.tidy(() => {
      // Project hidden states
      const projected = this.projectionLayer.predict(hiddenStates) as tf.Tensor3D;
      
      // Get logits
      const logits = this.outputLayer.predict(projected) as tf.Tensor3D;

      let loss: tf.Tensor;
      let accuracy: tf.Tensor;
      let maskedPositions: tf.Tensor;

      if (labels) {
        // Calculate loss only for masked positions (labels != -100)
        const validMask = tf.notEqual(labels, tf.scalar(-100));
        maskedPositions = tf.cast(validMask, 'float32');

        // Apply label smoothing
        const smoothedLabels = this.applyLabelSmoothing(labels, validMask as tf.Tensor2D);

        // Calculate cross-entropy loss
        const logSoftmax = tf.logSoftmax(logits);
        const losses = tf.neg(tf.sum(tf.mul(smoothedLabels, logSoftmax), -1));
        
        // Only consider valid positions
        const maskedLosses = tf.mul(losses, tf.cast(validMask, 'float32'));
        loss = tf.div(
          tf.sum(maskedLosses),
          tf.maximum(tf.sum(tf.cast(validMask, 'float32')), tf.scalar(1))
        );

        // Calculate accuracy
        const predictions = tf.argMax(logits, -1);
        const correct = tf.cast(
          tf.equal(predictions, tf.cast(labels, predictions.dtype)),
          'float32'
        );
        const maskedCorrect = tf.mul(correct, tf.cast(validMask, 'float32'));
        accuracy = tf.div(
          tf.sum(maskedCorrect),
          tf.maximum(tf.sum(tf.cast(validMask, 'float32')), tf.scalar(1))
        );
      } else {
        loss = tf.scalar(0);
        accuracy = tf.scalar(0);
        maskedPositions = tf.zeros([logits.shape[0], logits.shape[1]]);
      }

      return {
        logits: logits.squeeze() as tf.Tensor,
        loss,
        maskedPositions,
        accuracy
      };
    });
  }

  /**
   * Apply label smoothing to reduce overconfidence
   * @param labels Original labels
   * @param validMask Mask for valid positions
   * @returns Smoothed label distribution
   */
  private applyLabelSmoothing(
    labels: tf.Tensor2D,
    validMask: tf.Tensor2D
  ): tf.Tensor3D {
    return tf.tidy(() => {
      const epsilon = this.config.labelSmoothingEpsilon!;
      const vocabSize = this.config.vocabSize;

      // Convert labels to one-hot
      const oneHot = tf.oneHot(tf.cast(labels, 'int32'), vocabSize);

      // Apply label smoothing
      const smoothed = tf.add(
        tf.mul(oneHot, tf.scalar(1 - epsilon)),
        tf.scalar(epsilon / vocabSize)
      );

      // Zero out invalid positions
      const expandedMask = tf.expandDims(tf.cast(validMask, 'float32'), -1);
      return tf.mul(smoothed, expandedMask);
    });
  }

  /**
   * Prepare a batch for MLM training
   * @param texts Array of text strings
   * @param maxLength Maximum sequence length
   * @returns Prepared batch with masks and labels
   */
  public async prepareBatch(
    texts: string[],
    maxLength = this.config.maxSequenceLength!
  ): Promise<MLMBatch> {
    // Tokenize and encode texts
    const encodedTexts = texts.map(text => {
      const tokens = this.tokenizer.encode(text);
      
      // Add special tokens
      const clsId = this.tokenizer.getSpecialTokenId('CLS');
      const sepId = this.tokenizer.getSpecialTokenId('SEP');
      const padId = this.tokenizer.getSpecialTokenId('PAD');
      
      let sequence = [clsId, ...tokens, sepId];
      
      // Truncate if too long
      if (sequence.length > maxLength) {
        sequence = sequence.slice(0, maxLength - 1).concat([sepId]);
      }
      
      // Pad if too short
      while (sequence.length < maxLength) {
        sequence.push(padId);
      }
      
      return sequence;
    });

    // Convert to tensors
    const inputIds = tf.tensor2d(encodedTexts, [texts.length, maxLength], 'int32');
    
    // Create masks and labels
    const { maskedInputIds, labels, maskedPositions } = this.createMasks(inputIds);
    
    // Create attention mask
    const padId = this.tokenizer.getSpecialTokenId('PAD');
    const attentionMask = tf.cast(tf.notEqual(inputIds, tf.scalar(padId)), 'float32');

    // Clean up original inputIds
    inputIds.dispose();

    return {
      inputIds: maskedInputIds,
      attentionMask: attentionMask as tf.Tensor2D,
      labels: labels,
      maskedPositions: maskedPositions
    };
  }

  /**
   * Train on a batch of data
   * @param batch MLM batch
   * @param hiddenStates Hidden states from transformer
   * @param optimizer TensorFlow optimizer
   * @returns Training metrics
   */
  public trainStep(
    batch: MLMBatch,
    hiddenStates: tf.Tensor3D,
    optimizer: tf.Optimizer
  ): {
    loss: number;
    accuracy: number;
    maskedTokens: number;
  } {
    const { value: loss, grads } = tf.variableGrads(() => {
      const output = this.forward(hiddenStates, batch.labels);
      return output.loss as tf.Scalar;
    });

    // Apply gradients
    const mlmVars = [
      ...this.projectionLayer.getWeights(),
      ...this.outputLayer.getWeights()
    ];
    optimizer.applyGradients(grads);

    // Calculate metrics
    const output = this.forward(hiddenStates, batch.labels);
    const lossValue = loss.dataSync()[0];
    const accuracyValue = output.accuracy.dataSync()[0];
    const maskedTokens = tf.sum(batch.maskedPositions).dataSync()[0];

    // Cleanup
    loss.dispose();
    Object.values(grads).forEach(grad => grad.dispose());
    output.logits.dispose();
    output.loss.dispose();
    output.accuracy.dispose();
    output.maskedPositions.dispose();

    return {
      loss: lossValue,
      accuracy: accuracyValue,
      maskedTokens
    };
  }

  /**
   * Predict masked tokens
   * @param maskedInputIds Masked input token IDs
   * @param hiddenStates Hidden states from transformer
   * @param topK Number of top predictions to return
   * @returns Top predictions for each position
   */
  public predict(
    maskedInputIds: tf.Tensor2D,
    hiddenStates: tf.Tensor3D,
    topK = 5
  ): {
    predictions: number[][][];
    probabilities: number[][][];
  } {
    const output = this.forward(hiddenStates);
    
    return tf.tidy(() => {
      const probabilities = tf.softmax(output.logits as tf.Tensor3D);
      const { values, indices } = tf.topk(probabilities, topK);
      
      const predictionsArray = indices.arraySync() as number[][][];
      const probabilitiesArray = values.arraySync() as number[][][];
      
      return {
        predictions: predictionsArray,
        probabilities: probabilitiesArray
      };
    });
  }

  /**
   * Get model weights for saving/loading
   */
  public getWeights(): tf.Tensor[] {
    return [
      ...this.projectionLayer.getWeights(),
      ...this.outputLayer.getWeights()
    ];
  }

  /**
   * Set model weights
   */
  public setWeights(weights: tf.Tensor[]): void {
    const projectionWeights = weights.slice(0, this.projectionLayer.getWeights().length);
    const outputWeights = weights.slice(this.projectionLayer.getWeights().length);
    
    this.projectionLayer.setWeights(projectionWeights);
    this.outputLayer.setWeights(outputWeights);
  }

  /**
   * Get configuration
   */
  public getConfig(): MLMConfig {
    return { ...this.config };
  }

  /**
   * Dispose of resources
   */
  public dispose(): void {
    if (this.projectionLayer) {
      this.projectionLayer.dispose();
    }
    if (this.outputLayer) {
      this.outputLayer.dispose();
    }
    this.isInitialized = false;
  }
}
