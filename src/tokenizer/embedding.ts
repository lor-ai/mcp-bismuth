/**
 * @fileoverview Token embedding layer with learnable parameters
 * Provides dense vector representations for tokens
 */

import * as tf from '@tensorflow/tfjs-node';

export interface EmbeddingConfig {
  vocabSize: number;
  embeddingDim: number;
  paddingIdx?: number;
  maxNorm?: number;
  normType?: number;
  scaleGradByFreq?: boolean;
  sparse?: boolean;
}

export class TokenEmbedding {
  private config: EmbeddingConfig;
  private embeddingMatrix!: tf.Variable;
  private isInitialized = false;

  constructor(config: EmbeddingConfig) {
    this.config = {
      ...config
    };

    // Validate embedding dimension is in the 256-512 range as specified
    if (this.config.embeddingDim < 256 || this.config.embeddingDim > 512) {
      console.warn(`Embedding dimension ${this.config.embeddingDim} is outside recommended range 256-512`);
    }

    // Initialize the embedding matrix
    this.initializeWeights();
  }

  /**
   * Initialize the embedding matrix with random weights
   */
  private initializeWeights(): void {
    // Use Xavier/Glorot initialization for better gradient flow
    const limit = Math.sqrt(6.0 / (this.config.vocabSize + this.config.embeddingDim));
    
    // Create random matrix with uniform distribution in [-limit, limit]
    const weights = tf.randomUniform(
      [this.config.vocabSize, this.config.embeddingDim],
      -limit,
      limit,
      'float32'
    );

    // Handle padding index if specified
    if (this.config.paddingIdx !== undefined) {
      const paddingRow = tf.zeros([1, this.config.embeddingDim]);
      const mask = tf.oneHot(this.config.paddingIdx, this.config.vocabSize);
      const maskedWeights = tf.sub(
        weights,
        tf.matMul(
          mask.expandDims(1),
          weights.slice([this.config.paddingIdx, 0], [1, this.config.embeddingDim])
        )
      );
      
      this.embeddingMatrix = tf.variable(maskedWeights, true, `embedding_matrix_${Math.random().toString(36).substr(2, 9)}`);
      paddingRow.dispose();
      mask.dispose();
      maskedWeights.dispose();
    } else {
      this.embeddingMatrix = tf.variable(weights, true, `embedding_matrix_${Math.random().toString(36).substr(2, 9)}`);
    }

    weights.dispose();
    this.isInitialized = true;
    
    console.log(`Initialized embedding matrix: [${this.config.vocabSize}, ${this.config.embeddingDim}]`);
  }

  /**
   * Forward pass: convert token IDs to embeddings
   * @param tokenIds Token IDs tensor of shape [batch_size, sequence_length] or [sequence_length]
   * @returns Embedding tensor of shape [batch_size, sequence_length, embedding_dim] or [sequence_length, embedding_dim]
   */
  public forward(tokenIds: tf.Tensor): tf.Tensor {
    if (!this.isInitialized) {
      throw new Error('Embedding layer not initialized');
    }

    return tf.tidy(() => {
      // Ensure token IDs are integers
      const intTokenIds = tf.cast(tokenIds, 'int32');

      // Validate token IDs are within vocabulary range
      const maxTokenId = tf.max(intTokenIds);
      const minTokenId = tf.min(intTokenIds);
      
      if (maxTokenId.dataSync()[0] >= this.config.vocabSize) {
        throw new Error(`Token ID ${maxTokenId.dataSync()[0]} exceeds vocabulary size ${this.config.vocabSize}`);
      }
      
      if (minTokenId.dataSync()[0] < 0) {
        throw new Error(`Token ID ${minTokenId.dataSync()[0]} is negative`);
      }

      // Gather embeddings
      const embeddings = tf.gather(this.embeddingMatrix, intTokenIds);

      // Apply max norm constraint if specified
      if (this.config.maxNorm !== undefined) {
        const norms = tf.norm(embeddings, 2, -1, true);
        const clamped = tf.clipByValue(norms, 0, this.config.maxNorm);
        const normalized = tf.div(embeddings, tf.maximum(norms, tf.scalar(1e-8)));
        return tf.mul(normalized, clamped);
      }

      return embeddings;
    });
  }

  /**
   * Get embeddings for specific token IDs (utility method)
   * @param tokenIds Array of token IDs
   * @returns Promise resolving to embedding matrix
   */
  public async embed(tokenIds: number[]): Promise<tf.Tensor2D> {
    const tokenTensor = tf.tensor1d(tokenIds, 'int32');
    const embeddings = this.forward(tokenTensor) as tf.Tensor2D;
    tokenTensor.dispose();
    return embeddings;
  }

  /**
   * Get embedding for a single token
   * @param tokenId Single token ID
   * @returns Promise resolving to embedding vector
   */
  public async embedSingle(tokenId: number): Promise<tf.Tensor1D> {
    const embeddings = await this.embed([tokenId]);
    const singleEmbedding = embeddings.slice([0, 0], [1, this.config.embeddingDim]).squeeze([0]) as tf.Tensor1D;
    embeddings.dispose();
    return singleEmbedding;
  }

  /**
   * Update embeddings using gradients (for training)
   * @param gradients Gradients tensor with same shape as embedding matrix
   * @param learningRate Learning rate for the update
   */
  public updateWeights(gradients: tf.Tensor, learningRate = 0.001): void {
    if (!this.isInitialized) {
      throw new Error('Embedding layer not initialized');
    }

    tf.tidy(() => {
      const scaledGradients = tf.mul(gradients, tf.scalar(learningRate));
      const newWeights = tf.sub(this.embeddingMatrix, scaledGradients);
      
      // Handle padding index constraint
      if (this.config.paddingIdx !== undefined) {
        const paddingMask = tf.oneHot(this.config.paddingIdx, this.config.vocabSize);
        const maskedWeights = tf.sub(
          newWeights,
          tf.matMul(
            paddingMask.expandDims(1),
            newWeights.slice([this.config.paddingIdx, 0], [1, this.config.embeddingDim])
          )
        );
        this.embeddingMatrix.assign(maskedWeights);
      } else {
        this.embeddingMatrix.assign(newWeights);
      }
    });
  }

  /**
   * Get the embedding matrix (read-only access)
   * @returns Copy of the embedding matrix
   */
  public getWeights(): tf.Tensor2D {
    return this.embeddingMatrix.clone() as tf.Tensor2D;
  }

  /**
   * Set the embedding matrix (for loading pretrained weights)
   * @param weights New embedding matrix
   */
  public setWeights(weights: tf.Tensor2D): void {
    if (weights.shape[0] !== this.config.vocabSize || weights.shape[1] !== this.config.embeddingDim) {
      throw new Error(
        `Weight shape [${weights.shape}] doesn't match expected [${this.config.vocabSize}, ${this.config.embeddingDim}]`
      );
    }

    this.embeddingMatrix.assign(weights);
  }

  /**
   * Save embedding weights to file
   * @param filePath Path to save the weights
   */
  public async saveWeights(filePath: string): Promise<void> {
    // Create a simple model to save the variable
    const model = tf.sequential({
      layers: [
        tf.layers.dense({
          units: this.config.embeddingDim,
          inputShape: [this.config.vocabSize],
          weights: [this.embeddingMatrix, tf.zeros([this.config.embeddingDim])]
        })
      ]
    });
    
    await model.save(`file://${filePath}`);
    model.dispose();
    console.log(`Saved embedding weights to ${filePath}`);
  }

  /**
   * Load embedding weights from file
   * @param filePath Path to load the weights from
   */
  public async loadWeights(filePath: string): Promise<void> {
    const handler = tf.io.fileSystem(`file://${filePath}`);
    const modelArtifacts = await handler.load();
    
    if (modelArtifacts?.weightData) {
      const weightData = new Float32Array(modelArtifacts.weightData as ArrayBuffer);
      const weights = tf.tensor2d(
        weightData,
        [this.config.vocabSize, this.config.embeddingDim],
        'float32'
      );
      
      this.setWeights(weights);
      weights.dispose();
      console.log(`Loaded embedding weights from ${filePath}`);
    } else {
      throw new Error(`Could not load weights from ${filePath}`);
    }
  }

  /**
   * Get embedding statistics
   */
  public getStats(): {
    vocabSize: number;
    embeddingDim: number;
    totalParams: number;
    meanNorm: number;
    stdNorm: number;
  } {
    return tf.tidy(() => {
      const norms = tf.norm(this.embeddingMatrix, 2, 1);
      const meanNorm = tf.mean(norms).dataSync()[0];
      const stdNorm = tf.sqrt(tf.mean(tf.square(tf.sub(norms, meanNorm)))).dataSync()[0];

      return {
        vocabSize: this.config.vocabSize,
        embeddingDim: this.config.embeddingDim,
        totalParams: this.config.vocabSize * this.config.embeddingDim,
        meanNorm,
        stdNorm
      };
    });
  }

  /**
   * Find the most similar tokens to a given embedding
   * @param queryEmbedding Query embedding vector
   * @param topK Number of top similar tokens to return
   * @returns Object with token IDs and similarity scores
   */
  public findSimilar(queryEmbedding: tf.Tensor1D, topK = 10): {
    tokenIds: number[];
    similarities: number[];
  } {
    return tf.tidy(() => {
      // Normalize query embedding
      const normalizedQuery = tf.div(queryEmbedding, tf.norm(queryEmbedding));
      
      // Normalize all embeddings
      const normalizedEmbeddings = tf.div(
        this.embeddingMatrix,
        tf.norm(this.embeddingMatrix, 2, 1, true)
      );

      // Compute cosine similarities
      const similarities = tf.matMul(normalizedEmbeddings, normalizedQuery.expandDims(1)).squeeze([1]);
      
      // Get top-k
      const { values, indices } = tf.topk(similarities, topK);
      
      return {
        tokenIds: Array.from(indices.dataSync()),
        similarities: Array.from(values.dataSync())
      };
    });
  }

  /**
   * Get configuration
   */
  public getConfig(): EmbeddingConfig {
    return { ...this.config };
  }

  /**
   * Dispose of resources
   */
  public dispose(): void {
    if (this.embeddingMatrix && !this.embeddingMatrix.isDisposed) {
      this.embeddingMatrix.dispose();
    }
    this.isInitialized = false;
  }
}
