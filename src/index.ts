// Import polyfills first
import './utils/polyfills.js';

// Henry's Titan Memory Server
import { z } from "zod";
import * as tf from '@tensorflow/tfjs-node';
import type { TensorContainer } from '@tensorflow/tfjs-core';
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

import type { IMemoryState } from './types.js';
import { wrapTensor, unwrapTensor } from './types.js';
import { TitanMemoryModel } from './model.js';
export { TitanMemoryModel } from './model.js';
export { AdvancedTokenizer, BPETokenizer, TokenEmbedding, MaskedLanguageModelHead } from './tokenizer/index.js';
export type { TokenizerConfig, TokenizationResult, BPEConfig, EmbeddingConfig, MLMConfig } from './tokenizer/index.js';
export { LearnerService, type LearnerConfig } from './learner.js';
import { LearnerService, type LearnerConfig } from './learner.js';
import { VectorProcessor } from './utils.js';
import * as path from 'path';
import { promises as fs } from 'fs';
import * as crypto from 'crypto';

/**
 * Represents a serialized memory state that can be stored and loaded.
 */
interface SerializedMemoryState {
  shortTerm: number[];
  longTerm: number[];
  meta: number[];
  timestamps: number[];
  accessCounts: number[];
  surpriseHistory: number[];
}

/**
 * Statistics about the memory state.
 */
interface MemoryStats {
  shortTermMean: number;
  shortTermStd: number;
  longTermMean: number;
  longTermStd: number;
  capacity: number;
  surpriseScore: number;
  patternDiversity: number;
}

/**
 * Titan Memory Server - A neural memory system that can learn and predict sequences
 * while maintaining state through a memory vector.
 */
export class TitanMemoryServer {
  private server: McpServer;
  private model!: TitanMemoryModel;
  private vectorProcessor: VectorProcessor;
  private memoryState: IMemoryState;
  private learnerService?: LearnerService;
  private tokenizer?: any; // Will be AdvancedTokenizer when available
  private isInitialized = false;
  private autoSaveInterval?: NodeJS.Timeout;
  private readonly memoryPath: string;
  private readonly modelPath: string;
  private readonly weightsPath: string;

  constructor(options: { memoryPath?: string } = {}) {
    this.server = new McpServer({
      name: "Titan Memory",
      version: "1.2.0",
      description: "A neural memory system for LLMs that can learn and predict sequences while maintaining state"
    });
    this.vectorProcessor = VectorProcessor.getInstance();
    this.memoryPath = options.memoryPath ?? path.join(process.cwd(), '.titan_memory');
    this.modelPath = path.join(this.memoryPath, 'model.json');
    this.weightsPath = path.join(this.memoryPath, 'model.weights.bin');
    this.memoryState = this.initializeEmptyState();

    this.registerTools();
  }

  private initializeEmptyState(): IMemoryState {
    return tf.tidy(() => ({
      shortTerm: wrapTensor(tf.tensor2d([], [0, this.model?.getConfig()?.memoryDim ?? 1024])),
      longTerm: wrapTensor(tf.tensor2d([], [0, this.model?.getConfig()?.memoryDim ?? 1024])),
      meta: wrapTensor(tf.tensor2d([], [0, 5])),
      timestamps: wrapTensor(tf.tensor1d([])),
      accessCounts: wrapTensor(tf.tensor1d([])),
      surpriseHistory: wrapTensor(tf.tensor1d([]))
    }));
  }

  private wrapWithMemoryManagement<T extends TensorContainer>(fn: () => T): T {
    return tf.tidy(fn);
  }

  private async wrapWithMemoryManagementAsync<T extends TensorContainer>(fn: () => Promise<T>): Promise<T> {
    tf.engine().startScope();
    try {
      return await fn();
    } finally {
      tf.engine().endScope();
    }
  }

  private encryptTensor(tensor: tf.Tensor): Uint8Array {
    const data = tensor.dataSync();
    const key = crypto.randomBytes(32);
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
    const encrypted = Buffer.concat([cipher.update(Buffer.from(data.buffer)), cipher.final()]);
    return new Uint8Array(Buffer.concat([iv, key, encrypted]));
  }

  private validateMemoryState(state: IMemoryState): boolean {
    return tf.tidy(() => {
      try {
        const validations = [
          state.shortTerm && !unwrapTensor(state.shortTerm).isDisposed,
          state.longTerm && !unwrapTensor(state.longTerm).isDisposed,
          state.meta && !unwrapTensor(state.meta).isDisposed,
          state.timestamps && !unwrapTensor(state.timestamps).isDisposed,
          state.accessCounts && !unwrapTensor(state.accessCounts).isDisposed,
          state.surpriseHistory && !unwrapTensor(state.surpriseHistory).isDisposed
        ];

        return validations.every(Boolean);
      } catch (error) {
        // Silent validation failure
        return false;
      }
    });
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.autoInitialize();
      this.isInitialized = true;
    }
  }

  private registerTools(): void {
    // Help tool
    this.server.tool(
      'help',
      "Get help about available tools",
      {
        tool: z.string().optional().describe("Specific tool name to get help for"),
        category: z.string().optional().describe("Category of tools to explore"),
        showExamples: z.boolean().optional().describe("Include usage examples"),
        verbose: z.boolean().optional().describe("Include detailed descriptions")
      },
      async () => {
        await this.ensureInitialized();
        const helpText = "Available tools:\n" +
          "- help: Get help about available tools\n" +
          "- init_model: Initialize the Titan Memory model\n" +
          "- forward_pass: Perform a forward pass through the model\n" +
          "- train_step: Execute a training step\n" +
          "- get_memory_state: Get current memory state\n" +
          "- manifold_step: Update memory along a manifold direction\n" +
          "- prune_memory: Remove less relevant memories\n" +
          "- save_checkpoint: Save memory state to file\n" +
          "- load_checkpoint: Load memory state from file\n" +
          "- reset_gradients: Reset accumulated gradients";
        return {
          content: [{
            type: "text",
            text: helpText
          }]
        };
      }
    );

    // Init model tool
    this.server.tool(
      'init_model',
      {
        inputDim: z.number().int().positive().default(768).describe("Input dimension size"),
        hiddenDim: z.number().int().positive().default(512).describe("Hidden dimension size"),
        memoryDim: z.number().int().positive().default(1024).describe("Memory dimension size"),
        transformerLayers: z.number().int().positive().default(6).describe("Number of transformer layers"),
        numHeads: z.number().int().positive().default(8).describe("Number of attention heads"),
        ffDimension: z.number().int().positive().default(2048).describe("Feed-forward dimension"),
        dropoutRate: z.number().min(0).max(0.9).default(0.1).describe("Dropout rate"),
        maxSequenceLength: z.number().int().positive().default(512).describe("Maximum sequence length"),
        memorySlots: z.number().int().positive().default(5000).describe("Number of memory slots"),
        similarityThreshold: z.number().min(0).max(1).default(0.65).describe("Similarity threshold"),
        surpriseDecay: z.number().min(0).max(1).default(0.9).describe("Surprise decay rate"),
        pruningInterval: z.number().int().positive().default(1000).describe("Pruning interval"),
        gradientClip: z.number().positive().default(1.0).describe("Gradient clipping value")
      },
      async (params) => {
        try {
          this.model = new TitanMemoryModel();
          const config = {
            inputDim: params.inputDim,
            hiddenDim: params.hiddenDim ?? 512,
            memoryDim: params.memoryDim ?? 1024,
            transformerLayers: params.transformerLayers,
            numHeads: params.numHeads ?? 8,
            ffDimension: params.ffDimension ?? 2048,
            dropoutRate: params.dropoutRate ?? 0.1,
            maxSequenceLength: params.maxSequenceLength ?? 512,
            memorySlots: params.memorySlots,
            similarityThreshold: params.similarityThreshold ?? 0.65,
            surpriseDecay: params.surpriseDecay ?? 0.9,
            pruningInterval: params.pruningInterval ?? 1000,
            gradientClip: params.gradientClip ?? 1.0
          };

          await this.model.initialize(config);
          this.memoryState = this.initializeEmptyState();
          this.isInitialized = true;

          return {
            content: [{
              type: "text",
              text: `Titan Memory Model initialized successfully with configuration: ${JSON.stringify(config, null, 2)}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to initialize model: ${message}`
            }]
          };
        }
      }
    );

    // Memory stats tool
    this.server.tool(
      'memory_stats',
      {},
      async () => {
        await this.ensureInitialized();
        const memoryStats = this.model.get_memory_state();
        return {
          content: [{
            type: "text",
            text: JSON.stringify(memoryStats, null, 2)
          }]
        };
      }
    );

    // Forward pass tool
    this.server.tool(
      'forward_pass',
      "Perform a forward pass through the model with given input",
      {
        x: z.union([z.string(), z.array(z.number())]).describe("Input data (text or number array)"),
        memoryState: z.any().optional().describe("Optional memory state")
      },
      async (params) => {
        await this.ensureInitialized();

        try {
          const input = await this.processInput(params.x);
          const result = this.model.forward(wrapTensor(input), this.memoryState);

          const predicted = Array.from(unwrapTensor(result.predicted).dataSync());
          const memoryUpdate = {
            shortTerm: Array.from(unwrapTensor(result.memoryUpdate.newState.shortTerm).dataSync()),
            longTerm: Array.from(unwrapTensor(result.memoryUpdate.newState.longTerm).dataSync()),
            meta: Array.from(unwrapTensor(result.memoryUpdate.newState.meta).dataSync()),
            timestamps: Array.from(unwrapTensor(result.memoryUpdate.newState.timestamps).dataSync()),
            accessCounts: Array.from(unwrapTensor(result.memoryUpdate.newState.accessCounts).dataSync()),
            surpriseHistory: Array.from(unwrapTensor(result.memoryUpdate.newState.surpriseHistory).dataSync())
          };

          // Update memory state
          this.memoryState = result.memoryUpdate.newState;

          input.dispose();

          return {
            content: [{
              type: "text",
              text: `Forward pass completed. Predicted: [${predicted.slice(0, 10).map(x => x.toFixed(4)).join(', ')}${predicted.length > 10 ? '...' : ''}]`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Forward pass failed: ${message}`
            }]
          };
        }
      }
    );

    // Train step tool
    this.server.tool(
      'train_step',
      "Execute a training step with current and next inputs",
      {
        x_t: z.union([z.string(), z.array(z.number())]).describe("Current input"),
        x_next: z.union([z.string(), z.array(z.number())]).describe("Next expected input")
      },
      async (params) => {
        await this.ensureInitialized();

        try {
          const currentInput = await this.processInput(params.x_t);
          const nextInput = await this.processInput(params.x_next);

          const result = this.model.trainStep(
            wrapTensor(currentInput),
            wrapTensor(nextInput),
            this.memoryState
          );

          const loss = unwrapTensor(result.loss).dataSync()[0];

          currentInput.dispose();
          nextInput.dispose();

          return {
            content: [{
              type: "text",
              text: `Training step completed. Loss: ${loss.toFixed(6)}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Training step failed: ${message}`
            }]
          };
        }
      }
    );

    // Get memory state tool
    this.server.tool(
      'get_memory_state',
      "Get current memory state statistics and information",
      {},
      async () => {
        await this.ensureInitialized();

        try {
          const stats = this.getMemoryStats();
          const health = await this.performHealthCheck('quick');

          return {
            content: [{
              type: "text",
              text: `Memory State:
- Short-term mean: ${stats.shortTermMean.toFixed(4)}
- Long-term mean: ${stats.longTermMean.toFixed(4)}
- Capacity: ${(stats.capacity * 100).toFixed(1)}%
- Surprise score: ${stats.surpriseScore.toFixed(4)}
- Pattern diversity: ${stats.patternDiversity.toFixed(4)}
- Health status: ${health.status || 'unknown'}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to get memory state: ${message}`
            }]
          };
        }
      }
    );

    // Reset gradients tool
    this.server.tool(
      'reset_gradients',
      "Reset accumulated gradients in the model",
      {},
      async () => {
        await this.ensureInitialized();

        try {
          this.model.resetGradients();
          return {
            content: [{
              type: "text",
              text: "Gradients reset successfully"
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to reset gradients: ${message}`
            }]
          };
        }
      }
    );

    // Save checkpoint tool
    this.server.tool(
      'save_checkpoint',
      "Save current memory state to a checkpoint file",
      {
        path: z.string().describe("Path to save the checkpoint")
      },
      async (params) => {
        await this.ensureInitialized();

        try {
          const checkpointData = {
            memoryState: {
              shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).dataSync()),
              longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).dataSync()),
              meta: Array.from(unwrapTensor(this.memoryState.meta).dataSync()),
              timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).dataSync()),
              accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).dataSync()),
              surpriseHistory: Array.from(unwrapTensor(this.memoryState.surpriseHistory).dataSync())
            },
            config: this.model.getConfig(),
            timestamp: Date.now()
          };

          await fs.mkdir(path.dirname(params.path), { recursive: true });
          await fs.writeFile(params.path, JSON.stringify(checkpointData, null, 2));

          return {
            content: [{
              type: "text",
              text: `Checkpoint saved to ${params.path}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to save checkpoint: ${message}`
            }]
          };
        }
      }
    );

    // Load checkpoint tool
    this.server.tool(
      'load_checkpoint',
      "Load memory state from a checkpoint file",
      {
        path: z.string().describe("Path to the checkpoint file")
      },
      async (params) => {
        try {
          const data = await fs.readFile(params.path, 'utf-8');
          const checkpointData = JSON.parse(data) as {
            memoryState?: {
              shortTerm: number[];
              longTerm: number[];
              meta: number[];
              timestamps: number[];
              accessCounts: number[];
              surpriseHistory: number[];
            };
          };

          if (checkpointData.memoryState) {
            const memState = checkpointData.memoryState;
            this.memoryState = tf.tidy(() => ({
              shortTerm: wrapTensor(tf.tensor2d(memState.shortTerm, [memState.shortTerm.length, 1])),
              longTerm: wrapTensor(tf.tensor2d(memState.longTerm, [memState.longTerm.length, 1])),
              meta: wrapTensor(tf.tensor2d(memState.meta, [memState.meta.length, 1])),
              timestamps: wrapTensor(tf.tensor1d(memState.timestamps)),
              accessCounts: wrapTensor(tf.tensor1d(memState.accessCounts)),
              surpriseHistory: wrapTensor(tf.tensor1d(memState.surpriseHistory))
            }));
          }

          return {
            content: [{
              type: "text",
              text: `Checkpoint loaded from ${params.path}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to load checkpoint: ${message}`
            }]
          };
        }
      }
    );

    // Initialize learner service tool
    this.server.tool(
      'init_learner',
      "Initialize the online learning service with specified configuration",
      {
        bufferSize: z.number().int().positive().default(10000).describe("Replay buffer size"),
        batchSize: z.number().int().positive().default(32).describe("Training batch size"),
        updateInterval: z.number().int().positive().default(1000).describe("Update interval in milliseconds"),
        gradientClipValue: z.number().positive().default(1.0).describe("Gradient clipping value"),
        contrastiveWeight: z.number().min(0).max(1).default(0.2).describe("Contrastive learning weight"),
        nextTokenWeight: z.number().min(0).max(1).default(0.4).describe("Next token prediction weight"),
        mlmWeight: z.number().min(0).max(1).default(0.4).describe("Masked language modeling weight"),
        accumulationSteps: z.number().int().positive().default(4).describe("Gradient accumulation steps"),
        learningRate: z.number().positive().default(0.0001).describe("Learning rate"),
        nanGuardThreshold: z.number().positive().default(1e-6).describe("NaN guard threshold")
      },
      async (params) => {
        await this.ensureInitialized();
        
        try {
          // Initialize tokenizer if not already done
          if (!this.tokenizer) {
            // For now, we'll use a mock tokenizer - in practice this would be AdvancedTokenizer
            this.tokenizer = {
              encode: (text: string) => tf.randomNormal([768]),
              decode: (tensor: tf.Tensor) => 'decoded_text',
              getSpecialTokens: () => ({ mask: 103, pad: 0, unk: 1 })
            };
          }
          
          const learnerConfig: Partial<LearnerConfig> = {
            bufferSize: params.bufferSize,
            batchSize: params.batchSize,
            updateInterval: params.updateInterval,
            gradientClipValue: params.gradientClipValue,
            contrastiveWeight: params.contrastiveWeight,
            nextTokenWeight: params.nextTokenWeight,
            mlmWeight: params.mlmWeight,
            accumulationSteps: params.accumulationSteps,
            learningRate: params.learningRate,
            nanGuardThreshold: params.nanGuardThreshold
          };
          
          this.learnerService = new LearnerService(this.model, this.tokenizer, learnerConfig);
          
          return {
            content: [{
              type: "text",
              text: `Learner service initialized successfully with configuration: ${JSON.stringify(learnerConfig, null, 2)}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to initialize learner service: ${message}`
            }]
          };
        }
      }
    );

    // Pause learner tool
    this.server.tool(
      'pause_learner',
      "Pause the online learning loop",
      {},
      async () => {
        try {
          if (!this.learnerService) {
            return {
              content: [{
                type: "text",
                text: "Learner service not initialized. Please run init_learner first."
              }]
            };
          }
          
          this.learnerService.pauseTraining();
          
          return {
            content: [{
              type: "text",
              text: "Online learning loop paused successfully"
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to pause learner: ${message}`
            }]
          };
        }
      }
    );

    // Resume learner tool
    this.server.tool(
      'resume_learner',
      "Resume the online learning loop",
      {},
      async () => {
        try {
          if (!this.learnerService) {
            return {
              content: [{
                type: "text",
                text: "Learner service not initialized. Please run init_learner first."
              }]
            };
          }
          
          this.learnerService.resumeTraining();
          
          return {
            content: [{
              type: "text",
              text: "Online learning loop resumed successfully"
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to resume learner: ${message}`
            }]
          };
        }
      }
    );

    // Get learner stats tool
    this.server.tool(
      'get_learner_stats',
      "Get statistics about the online learning service",
      {},
      async () => {
        try {
          if (!this.learnerService) {
            return {
              content: [{
                type: "text",
                text: "Learner service not initialized. Please run init_learner first."
              }]
            };
          }
          
          const stats = this.learnerService.getTrainingStats();
          
          return {
            content: [{
              type: "text",
              text: `Learner Statistics:
- Buffer size: ${stats.bufferSize}
- Step count: ${stats.stepCount}
- Is running: ${stats.isRunning}
- Average loss: ${stats.averageLoss.toFixed(6)}
- Last loss: ${stats.lastLoss.toFixed(6)}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to get learner stats: ${message}`
            }]
          };
        }
      }
    );

    // Add training sample tool
    this.server.tool(
      'add_training_sample',
      "Add a training sample to the replay buffer",
      {
        input: z.union([z.string(), z.array(z.number())]).describe("Input data (text or number array)"),
        target: z.union([z.string(), z.array(z.number())]).describe("Target data (text or number array)"),
        positive: z.union([z.string(), z.array(z.number())]).optional().describe("Positive sample for contrastive learning"),
        negative: z.union([z.string(), z.array(z.number())]).optional().describe("Negative sample for contrastive learning")
      },
      async (params) => {
        try {
          if (!this.learnerService) {
            return {
              content: [{
                type: "text",
                text: "Learner service not initialized. Please run init_learner first."
              }]
            };
          }
          
          // Convert inputs to tensors if they are arrays
          const input = Array.isArray(params.input) ? tf.tensor1d(params.input) : params.input;
          const target = Array.isArray(params.target) ? tf.tensor1d(params.target) : params.target;
          const positive = params.positive ? (Array.isArray(params.positive) ? tf.tensor1d(params.positive) : params.positive) : undefined;
          const negative = params.negative ? (Array.isArray(params.negative) ? tf.tensor1d(params.negative) : params.negative) : undefined;
          
          this.learnerService.addTrainingSample(input, target, positive, negative);
          
          // Clean up tensor references if we created them
          if (Array.isArray(params.input)) (input as tf.Tensor).dispose();
          if (Array.isArray(params.target)) (target as tf.Tensor).dispose();
          if (positive && Array.isArray(params.positive)) (positive as tf.Tensor).dispose();
          if (negative && Array.isArray(params.negative)) (negative as tf.Tensor).dispose();
          
          const stats = this.learnerService.getTrainingStats();
          
          return {
            content: [{
              type: "text",
              text: `Training sample added successfully. Buffer size: ${stats.bufferSize}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to add training sample: ${message}`
            }]
          };
        }
      }
    );
  }

  private async processInput(input: string | number[]): Promise<tf.Tensor1D> {
    if (typeof input === 'string') {
      return await this.model.encodeText(input);
    } else {
      return tf.tensor1d(input);
    }
  }

  private async autoInitialize(): Promise<void> {
    try {
      // Try to load existing model
      const modelExists = await fs.access(this.modelPath).then(() => true).catch(() => false);

      if (modelExists) {
        this.model = new TitanMemoryModel();
        await this.model.loadModel(this.modelPath);
      } else {
        // Initialize with default config
        this.model = new TitanMemoryModel();
        await this.model.initialize({
          inputDim: 768,
          memorySlots: 5000,
          transformerLayers: 6
        });

        // Ensure directory exists and save
        await fs.mkdir(this.memoryPath, { recursive: true });
        await this.model.save(this.modelPath);
      }

      this.memoryState = this.initializeEmptyState();

      // Try to load existing memory state
      const memoryStateExists = await fs.access(path.join(this.memoryPath, 'memory_state.json')).then(() => true).catch(() => false);
      if (memoryStateExists) {
        await this.loadMemoryState();
      }

      // Setup auto-save
      this.autoSaveInterval = setInterval(async () => {
        try {
          await this.saveMemoryState();
        } catch (error) {
          // Silent auto-save failure
        }
      }, 60000); // Save every minute

    } catch (error) {
      // Silent auto-initialization failure
      // Continue with basic initialization
      this.model = new TitanMemoryModel();
      await this.model.initialize({
        inputDim: 768,
        memorySlots: 5000,
        transformerLayers: 6
      });
      this.memoryState = this.initializeEmptyState();
    }
  }

  private async saveMemoryState(): Promise<void> {
    try {
      const state = {
        shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).dataSync()),
        longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).dataSync()),
        meta: Array.from(unwrapTensor(this.memoryState.meta).dataSync()),
        timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).dataSync()),
        accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).dataSync()),
        surpriseHistory: Array.from(unwrapTensor(this.memoryState.surpriseHistory).dataSync()),
        timestamp: Date.now()
      };

      await fs.writeFile(
        path.join(this.memoryPath, 'memory_state.json'),
        JSON.stringify(state, null, 2)
      );
    } catch (error) {
      // Silent failure
    }
  }

  private async loadMemoryState(): Promise<void> {
    try {
      const data = await fs.readFile(path.join(this.memoryPath, 'memory_state.json'), 'utf-8');
      const state = JSON.parse(data) as {
        shortTerm: number[];
        longTerm: number[];
        meta: number[];
        timestamps: number[];
        accessCounts: number[];
        surpriseHistory: number[];
      };

      this.memoryState = tf.tidy(() => ({
        shortTerm: wrapTensor(tf.tensor2d(state.shortTerm, [state.shortTerm.length, 1])),
        longTerm: wrapTensor(tf.tensor2d(state.longTerm, [state.longTerm.length, 1])),
        meta: wrapTensor(tf.tensor2d(state.meta, [state.meta.length, 1])),
        timestamps: wrapTensor(tf.tensor1d(state.timestamps)),
        accessCounts: wrapTensor(tf.tensor1d(state.accessCounts)),
        surpriseHistory: wrapTensor(tf.tensor1d(state.surpriseHistory))
      }));
    } catch (error) {
      // Silent failure - continue with default state
    }
  }

  public async run(): Promise<void> {
    try {
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
      
      // Setup graceful shutdown
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());
      
      // Server running on stdio
    } catch (error) {
      // Failed to start server
      process.exit(1);
    }
  }

  private async shutdown(): Promise<void> {
    try {
      // Stop learner service if running
      if (this.learnerService) {
        this.learnerService.dispose();
      }
      
      // Clear auto-save interval
      if (this.autoSaveInterval) {
        clearInterval(this.autoSaveInterval);
      }
      
      // Save final state
      await this.saveMemoryState();
      
      // Dispose model
      if (this.model) {
        this.model.dispose();
      }
      
      process.exit(0);
    } catch (error) {
      process.exit(1);
    }
  }

  private getMemoryStats(): MemoryStats {
    return tf.tidy(() => {
      const shortTermData = unwrapTensor(this.memoryState.shortTerm).dataSync();
      const longTermData = unwrapTensor(this.memoryState.longTerm).dataSync();
      const surpriseData = unwrapTensor(this.memoryState.surpriseHistory).dataSync();

      const shortTermMean = shortTermData.length > 0 ? Array.from(shortTermData).reduce((a, b) => a + b, 0) / shortTermData.length : 0;
      const longTermMean = longTermData.length > 0 ? Array.from(longTermData).reduce((a, b) => a + b, 0) / longTermData.length : 0;
      const surpriseScore = surpriseData.length > 0 ? Array.from(surpriseData).reduce((a, b) => a + b, 0) / surpriseData.length : 0;

      return {
        shortTermMean,
        shortTermStd: 0, // Simplified for now
        longTermMean,
        longTermStd: 0, // Simplified for now
        capacity: shortTermData.length / (this.model?.getConfig()?.memorySlots || 5000),
        surpriseScore,
        patternDiversity: 0.5 // Simplified for now
      };
    });
  }

  private async performHealthCheck(checkType: string): Promise<any> {
    const memoryInfo = tf.memory();
    const stats = this.getMemoryStats();

    return {
      status: memoryInfo.numTensors < 1000 ? 'healthy' : 'warning',
      tensors: memoryInfo.numTensors,
      bytes: memoryInfo.numBytes,
      capacity: stats.capacity,
      checkType
    };
  }

  private calculateHealthScore(healthData: any): number {
    // Simple health score calculation
    let score = 1.0;

    if (healthData.tensors > 1000) { score -= 0.3; }
    if (healthData.capacity > 0.8) { score -= 0.2; }

    return Math.max(0, score);
  }

  private generateHealthRecommendations(healthData: any, healthScore: number): string[] {
    const recommendations = [];

    if (healthData.tensors > 1000) {
      recommendations.push("Consider running tensor cleanup - high tensor count detected");
    }

    if (healthData.capacity > 0.8) {
      recommendations.push("Memory capacity is high - consider pruning old memories");
    }

    if (healthScore < 0.7) {
      recommendations.push("Overall health is low - consider running optimization");
    }

    return recommendations;
  }
}

// Create and run server
const server = new TitanMemoryServer();
server.run().catch(() => process.exit(1));
