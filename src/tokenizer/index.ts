/**
 * @fileoverview Main tokenizer module integrating BPE, embeddings, and MLM
 * Provides unified interface for text encoding with backward compatibility
 */

import * as tf from '@tensorflow/tfjs-node';
import { BPETokenizer, type BPEConfig } from './bpe.js';
import { TokenEmbedding, type EmbeddingConfig } from './embedding.js';
import { MaskedLanguageModelHead, type MLMConfig } from './mlm.js';

export interface TokenizerConfig {
  // BPE configuration
  vocabSize?: number;
  mergesFile?: string;
  specialTokens?: string[];
  useCharFallback?: boolean;

  // Embedding configuration
  embeddingDim?: number;
  paddingIdx?: number;
  maxNorm?: number;

  // MLM configuration
  hiddenSize?: number;
  maskProbability?: number;
  maxSequenceLength?: number;
  labelSmoothingEpsilon?: number;

  // Backward compatibility
  useLegacyCharMode?: boolean;

  // Learning
  enableBootstrapping?: boolean;
  maxMergesPerText?: number;
}

export interface TokenizationResult {
  tokenIds: number[];
  embeddings: tf.Tensor2D;
  attentionMask: tf.Tensor1D;
  metadata: {
    originalLength: number;
    tokenizedLength: number;
    truncated: boolean;
    usedLegacyMode: boolean;
  };
}

export interface PretrainingBatch {
  inputIds: tf.Tensor2D;
  embeddings: tf.Tensor3D;
  attentionMask: tf.Tensor2D;
  labels: tf.Tensor2D;
  maskedPositions: tf.Tensor2D;
}

/**
 * Advanced tokenizer with BPE, embeddings, and self-supervised learning
 */
export class AdvancedTokenizer {
  private config: TokenizerConfig;
  private bpeTokenizer!: BPETokenizer;
  private embedding!: TokenEmbedding;
  private mlmHead: MaskedLanguageModelHead | null = null;
  private isInitialized = false;
  private bootstrapCounter = 0;

  constructor(config: TokenizerConfig = {}) {
    this.config = {
      vocabSize: 32000,
      embeddingDim: 256,
      hiddenSize: 768,
      maxSequenceLength: 512,
      useLegacyCharMode: false,
      enableBootstrapping: true,
      maxMergesPerText: 100,
      useCharFallback: true,
      maskProbability: 0.15,
      labelSmoothingEpsilon: 0.1,
      ...config
    };

    this.initializeComponents();
  }

  /**
   * Initialize all tokenizer components
   */
  private initializeComponents(): void {
    // Initialize BPE tokenizer
    const bpeConfig: BPEConfig = {
      vocabSize: this.config.vocabSize!,
      mergesFile: this.config.mergesFile,
      specialTokens: this.config.specialTokens,
      useCharFallback: this.config.useCharFallback
    };
    this.bpeTokenizer = new BPETokenizer(bpeConfig);

    // Initialize embeddings
    const embeddingConfig: EmbeddingConfig = {
      vocabSize: this.config.vocabSize!,
      embeddingDim: this.config.embeddingDim!,
      paddingIdx: this.bpeTokenizer.getSpecialTokenId('PAD'),
      maxNorm: this.config.maxNorm
    };
    this.embedding = new TokenEmbedding(embeddingConfig);

    console.log('Advanced tokenizer components initialized');
  }

  /**
   * Initialize the tokenizer (load existing merges, etc.)
   */
  public async initialize(): Promise<void> {
    await this.bpeTokenizer.loadMerges();

    // Initialize MLM head if needed
    if (this.config.hiddenSize) {
      const mlmConfig: MLMConfig = {
        hiddenSize: this.config.hiddenSize,
        vocabSize: this.config.vocabSize!,
        maskProbability: this.config.maskProbability,
        maxSequenceLength: this.config.maxSequenceLength,
        labelSmoothingEpsilon: this.config.labelSmoothingEpsilon
      };
      this.mlmHead = new MaskedLanguageModelHead(mlmConfig, this.bpeTokenizer, this.embedding);
    }

    this.isInitialized = true;
    console.log('Advanced tokenizer initialized successfully');
  }

  /**
   * Encode text to tokens and embeddings
   * @param text Input text
   * @param options Encoding options
   * @returns Tokenization result with embeddings
   */
  public async encode(
    text: string,
    options: {
      maxLength?: number;
      padding?: boolean;
      truncation?: boolean;
      addSpecialTokens?: boolean;
      returnTensors?: boolean;
    } = {}
  ): Promise<TokenizationResult> {
    const {
      maxLength = this.config.maxSequenceLength!,
      padding = true,
      truncation = true,
      addSpecialTokens = true,
      returnTensors = true
    } = options;

    // Handle legacy character mode fallback
    if (this.config.useLegacyCharMode) {
      return this.encodeLegacyChar(text, maxLength);
    }

    // Bootstrap learning if enabled
    if (this.config.enableBootstrapping && this.bootstrapCounter < 1000) {
      await this.bootstrapFromText(text);
      this.bootstrapCounter++;
    }

    // Tokenize with BPE
    let tokenIds = this.bpeTokenizer.encode(text);
    const originalLength = tokenIds.length;

    // Add special tokens
    if (addSpecialTokens) {
      const clsId = this.bpeTokenizer.getSpecialTokenId('CLS');
      const sepId = this.bpeTokenizer.getSpecialTokenId('SEP');
      tokenIds = [clsId, ...tokenIds, sepId];
    }

    let truncated = false;

    // Handle truncation
    if (truncation && tokenIds.length > maxLength) {
      if (addSpecialTokens) {
        const sepId = this.bpeTokenizer.getSpecialTokenId('SEP');
        tokenIds = tokenIds.slice(0, maxLength - 1).concat([sepId]);
      } else {
        tokenIds = tokenIds.slice(0, maxLength);
      }
      truncated = true;
    }

    // Handle padding
    const padId = this.bpeTokenizer.getSpecialTokenId('PAD');
    const attentionMask: number[] = [];

    for (let i = 0; i < tokenIds.length; i++) {
      attentionMask.push(tokenIds[i] === padId ? 0 : 1);
    }

    if (padding && tokenIds.length < maxLength) {
      while (tokenIds.length < maxLength) {
        tokenIds.push(padId);
        attentionMask.push(0);
      }
    }

    // Get embeddings
    const embeddings = await this.embedding.embed(tokenIds);
    const attentionMaskTensor = tf.tensor1d(attentionMask, 'float32');

    return {
      tokenIds,
      embeddings: embeddings,
      attentionMask: attentionMaskTensor,
      metadata: {
        originalLength,
        tokenizedLength: tokenIds.length,
        truncated,
        usedLegacyMode: false
      }
    };
  }

  /**
   * Legacy character-based encoding for backward compatibility
   */
  private async encodeLegacyChar(text: string, maxLength: number): Promise<TokenizationResult> {
    const charCodes = text.split('').map(char => char.charCodeAt(0) % 256);
    const tokenIds = charCodes.slice(0, maxLength);

    // Pad to maxLength
    while (tokenIds.length < maxLength) {
      tokenIds.push(0);
    }

    const attentionMask = tokenIds.map(id => id === 0 ? 0 : 1);

    // Create simple embeddings for character mode
    const embeddings = tf.randomNormal([tokenIds.length, this.config.embeddingDim!]) as tf.Tensor2D;
    const attentionMaskTensor = tf.tensor1d(attentionMask, 'float32');

    return {
      tokenIds,
      embeddings,
      attentionMask: attentionMaskTensor,
      metadata: {
        originalLength: text.length,
        tokenizedLength: tokenIds.length,
        truncated: text.length > maxLength,
        usedLegacyMode: true
      }
    };
  }

  /**
   * Bootstrap BPE learning from incoming text
   */
  private async bootstrapFromText(text: string): Promise<void> {
    if (text.length < 50) { return; } // Skip very short texts

    try {
      await this.bpeTokenizer.learnFromText(text, this.config.maxMergesPerText);

      // Update vocabulary size in embedding if needed
      const newVocabSize = this.bpeTokenizer.getVocabSize();
      if (newVocabSize !== this.embedding.getConfig().vocabSize) {
        console.log(`Vocabulary expanded to ${newVocabSize} tokens`);
        // Note: In a production system, you'd want to expand the embedding matrix
        // This is a simplified version
      }
    } catch (error) {
      console.warn('Error during bootstrap learning:', error);
    }
  }

  /**
   * Decode token IDs back to text
   * @param tokenIds Array of token IDs
   * @returns Decoded text
   */
  public decode(tokenIds: number[]): string {
    if (this.config.useLegacyCharMode) {
      return tokenIds
        .filter(id => id !== 0)
        .map(id => String.fromCharCode(id))
        .join('');
    }

    return this.bpeTokenizer.decode(tokenIds);
  }

  /**
   * Prepare a batch for MLM pretraining
   * @param texts Array of texts
   * @param maxLength Maximum sequence length
   * @returns Pretraining batch
   */
  public async preparePretrainingBatch(
    texts: string[],
    maxLength = this.config.maxSequenceLength!
  ): Promise<PretrainingBatch> {
    if (!this.mlmHead) {
      throw new Error('MLM head not initialized. Set hiddenSize in config.');
    }

    // Get MLM batch
    const mlmBatch = await this.mlmHead.prepareBatch(texts, maxLength);

    // Get embeddings for the inputs
    const embeddings = this.embedding.forward(mlmBatch.inputIds) as tf.Tensor3D;

    return {
      inputIds: mlmBatch.inputIds,
      embeddings,
      attentionMask: mlmBatch.attentionMask,
      labels: mlmBatch.labels,
      maskedPositions: mlmBatch.maskedPositions
    };
  }

  /**
   * Perform MLM training step
   * @param batch Pretraining batch
   * @param hiddenStates Hidden states from transformer
   * @param optimizer Optimizer
   * @returns Training metrics
   */
  public trainMLMStep(
    batch: PretrainingBatch,
    hiddenStates: tf.Tensor3D,
    optimizer: tf.Optimizer
  ): { loss: number; accuracy: number; maskedTokens: number } {
    if (!this.mlmHead) {
      throw new Error('MLM head not initialized');
    }

    return this.mlmHead.trainStep(
      {
        inputIds: batch.inputIds,
        attentionMask: batch.attentionMask,
        labels: batch.labels,
        maskedPositions: batch.maskedPositions
      },
      hiddenStates,
      optimizer
    );
  }

  /**
   * Get tokenizer statistics
   */
  public getStats(): {
    bpe: { vocabSize: number; mergesCount: number; specialTokensCount: number };
    embedding: { vocabSize: number; embeddingDim: number; totalParams: number; meanNorm: number; stdNorm: number };
    bootstrapCount: number;
    mode: 'BPE' | 'Legacy';
  } {
    return {
      bpe: this.bpeTokenizer.getStats(),
      embedding: this.embedding.getStats(),
      bootstrapCount: this.bootstrapCounter,
      mode: this.config.useLegacyCharMode ? 'Legacy' : 'BPE'
    };
  }

  /**
   * Save tokenizer state
   * @param directory Directory to save to
   */
  public async save(directory: string): Promise<void> {
    // BPE merges are automatically saved

    // Save embedding weights
    await this.embedding.saveWeights(`${directory}/embedding_weights`);

    // Save MLM weights if present
    if (this.mlmHead) {
      const mlmWeights = this.mlmHead.getWeights();
      // Create a simple model to save the weights
      const model = tf.sequential({
        layers: [
          tf.layers.dense({
            units: 1,
            weights: mlmWeights
          })
        ]
      });

      await model.save(`file://${directory}/mlm_model`);
      model.dispose();
    }

    console.log(`Tokenizer saved to ${directory}`);
  }

  /**
   * Load tokenizer state
   * @param directory Directory to load from
   */
  public async load(directory: string): Promise<void> {
    try {
      // Load embedding weights
      await this.embedding.loadWeights(`${directory}/embedding_weights`);

      // Load MLM weights if present
      if (this.mlmHead) {
        // This is simplified - in practice you'd save/load the full model
        console.log('MLM weights loading not fully implemented in this example');
      }

      console.log(`Tokenizer loaded from ${directory}`);
    } catch (error) {
      console.warn('Error loading tokenizer:', error);
    }
  }

  /**
   * Switch between BPE and legacy modes
   * @param useLegacyMode Whether to use legacy character mode
   */
  public setLegacyMode(useLegacyMode: boolean): void {
    this.config.useLegacyCharMode = useLegacyMode;
    console.log(`Switched to ${useLegacyMode ? 'legacy character' : 'BPE'} mode`);
  }

  /**
   * Get embedding layer (for integration with model)
   */
  public getEmbedding(): TokenEmbedding {
    return this.embedding;
  }

  /**
   * Get BPE tokenizer (for direct access)
   */
  public getBPETokenizer(): BPETokenizer {
    return this.bpeTokenizer;
  }

  /**
   * Get MLM head (for training)
   */
  public getMLMHead(): MaskedLanguageModelHead | null {
    return this.mlmHead;
  }

  /**
   * Dispose of resources
   */
  public getConfig(): TokenizerConfig {
    return this.config;
  }

  public dispose(): void {
    this.bpeTokenizer.dispose();
    this.embedding.dispose();
    if (this.mlmHead) {
      this.mlmHead.dispose();
    }
    this.isInitialized = false;
  }
}

// Re-export types and classes
export { BPETokenizer, TokenEmbedding, MaskedLanguageModelHead };
export type { BPEConfig, EmbeddingConfig, MLMConfig };
