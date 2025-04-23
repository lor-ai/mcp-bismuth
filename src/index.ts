// Import polyfills first
import './utils/polyfills.js';

// Henry's Titan Memory Server
import { z } from "zod";
import * as tf from '@tensorflow/tfjs-node';
import type { TensorContainer } from '@tensorflow/tfjs-core';
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { RequestHandlerExtra } from "@modelcontextprotocol/sdk/shared/protocol.js";

import { IMemoryState, wrapTensor, unwrapTensor } from './types.js';
import { TitanMemoryModel } from './model.js';
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
 * Standard response format for all MCP tools.
 */
interface ToolResponse {
  [key: string]: unknown;
  content: Array<{
    [key: string]: unknown;
    type: "text" | "error" | "data";
    text: string;
    data?: any;
  }>;
  _meta?: Record<string, unknown>;
  isError?: boolean;
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
    this.memoryPath = options.memoryPath || path.join(process.cwd(), '.titan_memory');
    this.modelPath = path.join(this.memoryPath, 'model.json');
    this.weightsPath = path.join(this.memoryPath, 'model.weights.bin');
    this.memoryState = this.initializeEmptyState();

    this.registerTools();
  }

  private initializeEmptyState(): IMemoryState {
    return tf.tidy(() => ({
      shortTerm: wrapTensor(tf.tensor2d([], [0, this.model?.getConfig()?.memoryDim || 1024])),
      longTerm: wrapTensor(tf.tensor2d([], [0, this.model?.getConfig()?.memoryDim || 1024])),
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
        console.warn('Error validating memory state:', error);
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
      {
        tool: z.string().optional().describe("Specific tool name to get help for"),
        category: z.string().optional().describe("Category of tools to explore"),
        showExamples: z.boolean().optional().describe("Include usage examples"),
        verbose: z.boolean().optional().describe("Include detailed descriptions")
      },
      async (params, extra): Promise<ToolResponse> => {
        await this.ensureInitialized();
        const helpText = "Available tools:\\n" +
          "- help: Get help about available tools\\n" +
          "- init_model: Initialize the Titan Memory model\\n" +
          "- forward_pass: Perform a forward pass through the model\\n" +
          "- train_step: Execute a training step\\n" +
          "- get_memory_state: Get current memory state\\n" +
          "- manifold_step: Update memory along a manifold direction\\n" +
          "- prune_memory: Remove less relevant memories\\n" +
          "- save_checkpoint: Save memory state to file\\n" +
          "- load_checkpoint: Load memory state from file\\n" +
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
      async (params, extra): Promise<ToolResponse> => {
        this.model = new TitanMemoryModel();
        const config = {
          inputDim: params.inputDim,
          hiddenDim: params.hiddenDim || 512,
          memoryDim: params.memoryDim || 1024,
          transformerLayers: params.transformerLayers,
          numHeads: params.numHeads || 8,
          ffDimension: params.ffDimension || 2048,
          dropoutRate: params.dropoutRate || 0.1,
          maxSequenceLength: params.maxSequenceLength || 512,
          memorySlots: params.memorySlots,
          similarityThreshold: params.similarityThreshold || 0.65,
          surpriseDecay: params.surpriseDecay || 0.9,
          pruningInterval: params.pruningInterval || 1000,
          gradientClip: params.gradientClip || 1.0
        };
        await this.model.initialize(config);
        this.memoryState = this.initializeEmptyState();
        this.isInitialized = true;
        return {
          content: [{
            type: "text",
            text: `Model initialized with configuration: ${JSON.stringify(config)}`
          }]
        };
      }
    );

    // Forward pass tool
    this.server.tool(
      'forward_pass',
      {
        x: z.union([
          z.array(z.number()),
          z.string()
        ]).describe("Input vector or text"),
        memoryState: z.object({
          shortTerm: z.array(z.number()).optional(),
          longTerm: z.array(z.number()).optional(),
          meta: z.array(z.number()).optional(),
          timestamps: z.array(z.number()).optional(),
          accessCounts: z.array(z.number()).optional(),
          surpriseHistory: z.array(z.number()).optional()
        }).optional().describe("Memory state to use")
      },
      async (params, extra): Promise<ToolResponse> => {
        await this.ensureInitialized();

        const inputTensor = await this.processInput(params.x);

        let stateToUse = this.memoryState;
        if (params.memoryState) {
          const tensors = tf.tidy(() => ({
            shortTerm: tf.tensor(params.memoryState?.shortTerm || []),
            longTerm: tf.tensor(params.memoryState?.longTerm || []),
            meta: tf.tensor(params.memoryState?.meta || []),
            timestamps: tf.tensor1d(params.memoryState?.timestamps || []),
            accessCounts: tf.tensor1d(params.memoryState?.accessCounts || []),
            surpriseHistory: tf.tensor1d(params.memoryState?.surpriseHistory || [])
          }));
          stateToUse = {
            shortTerm: wrapTensor(tensors.shortTerm),
            longTerm: wrapTensor(tensors.longTerm),
            meta: wrapTensor(tensors.meta),
            timestamps: wrapTensor(tensors.timestamps),
            accessCounts: wrapTensor(tensors.accessCounts),
            surpriseHistory: wrapTensor(tensors.surpriseHistory)
          };
        }

        const result = this.model.forward(wrapTensor(inputTensor), stateToUse);

        this.memoryState = result.memoryUpdate.newState;

        const responseData = tf.tidy(() => ({
          predicted: Array.from(unwrapTensor(result.predicted).dataSync()),
          immediateSurprise: Array.from(unwrapTensor(result.memoryUpdate.surprise.immediate).dataSync()),
          accumulatedSurprise: Array.from(unwrapTensor(result.memoryUpdate.surprise.accumulated).dataSync()),
          totalSurprise: Array.from(unwrapTensor(result.memoryUpdate.surprise.totalSurprise).dataSync())
        }));

        inputTensor.dispose();

        return {
          content: [{
            type: "text",
            text: `Forward pass completed. Predicted: [${responseData.predicted.slice(0, 5)}...], Surprise (I/A/T): [${responseData.immediateSurprise.slice(0, 1)}...]/[${responseData.accumulatedSurprise.slice(0, 1)}...]/[${responseData.totalSurprise.slice(0, 1)}...]`
          }]
        };
      }
    );

    // Train step tool
    this.server.tool(
      'train_step',
      {
        x_t: z.union([
          z.array(z.number()),
          z.string()
        ]).describe("Current input vector or text"),
        x_next: z.union([
          z.array(z.number()),
          z.string()
        ]).describe("Next input vector or text")
      },
      async (params, extra): Promise<ToolResponse> => {
        await this.ensureInitialized();

        const x_t_tensor = await this.processInput(params.x_t);
        const x_next_tensor = await this.processInput(params.x_next);

        const result = this.model.trainStep(
          wrapTensor(x_t_tensor),
          wrapTensor(x_next_tensor),
          this.memoryState
        );

        const responseData = tf.tidy(() => {
          const lossVal = unwrapTensor(result.loss).dataSync()[0];
          const gradShortTerm = Array.from(unwrapTensor(result.gradients.shortTerm).dataSync());
          const gradLongTerm = Array.from(unwrapTensor(result.gradients.longTerm).dataSync());
          const gradMeta = Array.from(unwrapTensor(result.gradients.meta).dataSync());
          return { lossVal, gradShortTerm, gradLongTerm, gradMeta };
        });

        x_t_tensor.dispose();
        x_next_tensor.dispose();

        return {
          content: [{
            type: "text",
            text: `Training step completed. Loss: ${responseData.lossVal.toFixed(4)}. Gradients (Short/Long/Meta): [${responseData.gradShortTerm.length}]/[${responseData.gradLongTerm.length}]/[${responseData.gradMeta.length}]`
          }]
        };
      }
    );

    // Get memory state tool
    this.server.tool(
      'get_memory_state',
      {
        type: z.string().optional().describe("Optional memory type filter")
      },
      async (params, extra): Promise<ToolResponse> => {
        await this.ensureInitialized();

        const state = this.model.getMemorySnapshot();

        const statsResult = tf.tidy(() => {
          const shortTermMean = state.shortTerm.mean().dataSync()[0];
          const shortTermStd = tf.moments(state.shortTerm).variance.sqrt().dataSync()[0];
          const longTermMean = state.longTerm.mean().dataSync()[0];
          const longTermStd = tf.moments(state.longTerm).variance.sqrt().dataSync()[0];

          const surpriseHistory = this.memoryState.surpriseHistory;
          const surpriseScore = surpriseHistory && surpriseHistory.size > 0
            ? surpriseHistory.mean().dataSync()[0]
            : 0;

          const meta = this.memoryState.meta;
          const patternDiversity = meta && meta.size > 0
            ? tf.moments(meta).variance.sqrt().mean().dataSync()[0]
            : 0;

          const memorySlots = this.model.getConfig().memorySlots;
          const timestamps = this.memoryState.timestamps;
          const usedSlots = timestamps ? timestamps.shape[0] : 0;
          const capacity = 1 - (usedSlots / memorySlots);

          const timestampsArray = timestamps ? Array.from(timestamps.dataSync()) : [];
          const accessCountsArray = this.memoryState.accessCounts ? Array.from(this.memoryState.accessCounts.dataSync()) : [];

          return {
            stats: {
              shortTermMean,
              shortTermStd,
              longTermMean,
              longTermStd,
              surpriseScore,
              patternDiversity,
              capacity
            },
            capacity,
            timestamps: timestampsArray,
            accessCounts: accessCountsArray
          };
        });

        Object.values(state).forEach(t => t.dispose());

        const status = statsResult.capacity > 0.3 ? "active" : "pruning";

        return {
          content: [{
            type: "text",
            text: `Memory state retrieved. Status: ${status}, Capacity: ${(statsResult.capacity * 100).toFixed(1)}%, Entries: ${statsResult.timestamps.length}, Avg Surprise: ${statsResult.stats.surpriseScore.toFixed(3)}`
          }]
        };
      }
    );

    // Manifold step tool
    this.server.tool(
      'manifold_step',
      {
        base: z.array(z.number()).describe("Base memory state"),
        velocity: z.array(z.number()).describe("Update direction")
      },
      async (params, extra): Promise<ToolResponse> => {
        await this.ensureInitialized();

        const result = tf.tidy(() => {
          const base = tf.tensor(params.base);
          const velocity = tf.tensor(params.velocity);
          return this.model.manifoldStep(
            wrapTensor(base),
            wrapTensor(velocity)
          );
        });

        const newBaseArray = Array.from(unwrapTensor(result).dataSync());
        result.dispose();

        return {
          content: [{
            type: "text",
            text: `Manifold step completed. New base: [${newBaseArray.slice(0, 5)}...]`
          }]
        };
      }
    );

    // Prune memory tool
    this.server.tool(
      'prune_memory',
      {
        threshold: z.number().min(0).max(1).describe("Pruning threshold (0-1)")
      },
      async (params, extra): Promise<ToolResponse> => {
        await this.ensureInitialized();

        this.memoryState = this.model.pruneMemory(this.memoryState, params.threshold);

        const statsResult = tf.tidy(() => {
          const memorySlots = this.model.getConfig().memorySlots;
          const timestamps = this.memoryState.timestamps;
          const usedSlots = timestamps ? timestamps.shape[0] : 0;
          const capacity = 1 - (usedSlots / memorySlots);
          return { capacity, remainingEntries: usedSlots };
        });

        return {
          content: [{
            type: "text",
            text: `Memory pruned. New capacity: ${(statsResult.capacity * 100).toFixed(1)}%, Remaining entries: ${statsResult.remainingEntries}`
          }]
        };
      }
    );

    // Save checkpoint tool
    this.server.tool(
      'save_checkpoint',
      {
        path: z.string().optional().describe("Checkpoint file path (optional, defaults to internal path)")
      },
      async (params, extra): Promise<ToolResponse> => {
        await this.ensureInitialized();

        const checkpointPath = params.path || this.modelPath;
        await this.model.save(checkpointPath);

        return {
          content: [{
            type: "text",
            text: `Memory state saved to ${checkpointPath}`
          }]
        };
      }
    );

    // Load checkpoint tool
    this.server.tool(
      'load_checkpoint',
      {
        path: z.string().optional().describe("Checkpoint file path (optional, defaults to internal path)")
      },
      async (params, extra): Promise<ToolResponse> => {
        const checkpointPath = params.path || this.modelPath;
        try {
          if (!this.model) {
            this.model = new TitanMemoryModel();
          }
          await this.model.load(checkpointPath);
          this.memoryState = this.model.getMemoryState();
          this.isInitialized = true;

          return {
            content: [{
              type: "text",
              text: `Memory state loaded from ${checkpointPath}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          console.error(`Failed to load checkpoint from ${checkpointPath}:`, error);
          return {
            content: [{
              type: "error",
              text: `Failed to load memory state from ${checkpointPath}: ${message}`
            }],
            isError: true
          };
        }
      }
    );

    // Reset gradients tool
    this.server.tool(
      'reset_gradients',
      {},
      async (params, extra): Promise<ToolResponse> => {
        await this.ensureInitialized();

        this.model.resetGradients();

        return {
          content: [{
            type: "text",
            text: "Gradients reset successfully"
          }]
        };
      }
    );
  }

  private async processInput(input: string | number[]): Promise<tf.Tensor1D> {
    if (typeof input === 'string') {
      return this.model.encodeText(input);
    } else {
      return tf.tidy(() => tf.tensor1d(input));
    }
  }

  private async autoInitialize(): Promise<void> {
    try {
      await tf.ready();
      await tf.setBackend('tensorflow');

      const backend = tf.getBackend();
      if (!backend) {
        throw new Error('Failed to initialize TensorFlow.js backend');
      }

      console.log('TensorFlow backend initialized:', backend);

      await fs.mkdir(this.memoryPath, { recursive: true });

      this.model = new TitanMemoryModel({
        inputDim: 768,
        memorySlots: 5000,
        transformerLayers: 6
      });

      await this.model.initialize();

      try {
        const [modelExists, weightsExist, memoryStateExists] = await Promise.all([
          fs.access(this.modelPath).then(() => true).catch(() => false),
          fs.access(this.weightsPath).then(() => true).catch(() => false),
          fs.access(path.join(this.memoryPath, 'memory_state.json')).then(() => true).catch(() => false)
        ]);

        if (modelExists && weightsExist) {
          console.log('Found existing model and weights, loading...');
          await this.model.loadModel(this.modelPath);

          if (memoryStateExists) {
            console.log('Found existing memory state, loading...');
            const memoryStateJson = await fs.readFile(
              path.join(this.memoryPath, 'memory_state.json'),
              'utf8'
            );
            const memoryState = JSON.parse(memoryStateJson) as SerializedMemoryState;

            this.memoryState = tf.tidy(() => ({
              shortTerm: wrapTensor(tf.tensor1d(memoryState.shortTerm)),
              longTerm: wrapTensor(tf.tensor1d(memoryState.longTerm)),
              meta: wrapTensor(tf.tensor1d(memoryState.meta)),
              timestamps: wrapTensor(tf.tensor1d(memoryState.timestamps)),
              accessCounts: wrapTensor(tf.tensor1d(memoryState.accessCounts)),
              surpriseHistory: wrapTensor(tf.tensor1d(memoryState.surpriseHistory))
            }));
          } else {
            console.log('No saved memory state found, initializing new state');
            this.memoryState = this.initializeEmptyState();
            await this.saveMemoryState();
          }
        } else {
          console.log('No saved model found, initializing new model and state');
          await this.model.saveModel(this.modelPath);
          this.memoryState = this.initializeEmptyState();
          await this.saveMemoryState();
        }
      } catch (loadError) {
        console.error('Error loading saved state:', loadError);
        console.log('Initializing new model and state');
        this.memoryState = this.initializeEmptyState();
        await this.model.saveModel(this.modelPath);
        await this.saveMemoryState();
      }

      if (!this.autoSaveInterval) {
        this.autoSaveInterval = setInterval(async () => {
          try {
            await this.saveMemoryState();
          } catch (error) {
            console.error('Failed to auto-save memory state:', error);
          }
        }, 300000);
      }

      this.isInitialized = true;
    } catch (error: unknown) {
      console.error('Initialization failed:', error instanceof Error ? error.message : error);
      throw error;
    }
  }

  private async saveMemoryState(): Promise<void> {
    tf.engine().startScope();

    try {
      await this.model.saveModel(this.modelPath);

      const memoryState = tf.tidy(() => {
        if (!this.validateMemoryState(this.memoryState)) {
          throw new Error('Invalid memory state during save');
        }

        return {
          shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).clone().dataSync()),
          longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).clone().dataSync()),
          meta: Array.from(unwrapTensor(this.memoryState.meta).clone().dataSync()),
          timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).clone().dataSync()),
          accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).clone().dataSync()),
          surpriseHistory: Array.from(unwrapTensor(this.memoryState.surpriseHistory).clone().dataSync())
        };
      });

      const encryptedState = tf.tidy(() => {
        const tensors = [
          tf.tensor(memoryState.shortTerm),
          tf.tensor(memoryState.longTerm),
          tf.tensor(memoryState.meta),
          tf.tensor(memoryState.timestamps),
          tf.tensor(memoryState.accessCounts),
          tf.tensor(memoryState.surpriseHistory)
        ];

        const encryptedBuffers = tensors.map(tensor => {
          const encrypted = this.encryptTensor(tensor);
          return Buffer.from(encrypted);
        });

        return Buffer.concat(encryptedBuffers);
      });

      await fs.writeFile(this.weightsPath, encryptedState);
    } catch (error) {
      console.error('Failed to save memory state:', error);
      throw error;
    } finally {
      tf.engine().endScope();
    }
  }

  /**
   * Run the Titan Memory Server
   * This method initializes the server and connects to the transport
   */
  public async run(): Promise<void> {
    try {
      console.log('Starting Titan Memory Server...');

      await this.ensureInitialized();

      if (!this.autoSaveInterval) {
        this.autoSaveInterval = setInterval(async () => {
          try {
            await this.model.save(this.modelPath);
            console.log(`Auto-saved memory state to ${this.modelPath}`);
          } catch (error) {
            console.error('Error during auto-save:', error);
          }
        }, 5 * 60 * 1000);
      }

      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      const serverInfo = (this.server as any);
      console.log(`Titan Memory Server v${serverInfo.version || 'N/A'} running`);
      console.log('Available tools:');

      const toolSchemas = (this.server as any).toolSchemas || {};
      const tools = Object.keys(toolSchemas);
      tools.forEach(tool => {
        console.log(`- ${tool}`);
      });

      process.on('SIGINT', async () => {
        console.log('Shutting down Titan Memory Server...');

        try {
          await this.model.save(this.modelPath);
          console.log(`Memory state saved to ${this.modelPath}`);
        } catch (error) {
          console.error('Error saving memory state on exit:', error);
        }

        if (this.autoSaveInterval) {
          clearInterval(this.autoSaveInterval);
        }

        if (this.model) {
          this.model.dispose();
        }

        process.exit(0);
      });
    } catch (error) {
      console.error('Error starting Titan Memory Server:', error);
      throw error;
    }
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  new TitanMemoryServer().run().catch(error => {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(JSON.stringify({
      jsonrpc: "2.0",
      method: "error",
      params: {
        message: `Boot failed: ${errorMessage}`
      }
    }));
    process.exit(1);
  });
}

const HelpParams = z.object({
  tool: z.string().optional(),
  category: z.string().optional(),
  showExamples: z.boolean().optional(),
  verbose: z.boolean().optional(),
  interactive: z.boolean().optional(),
  context: z.record(z.any()).optional()
});

const InitModelParams = z.object({
  inputDim: z.number().int().positive().optional(),
  hiddenDim: z.number().int().positive().optional(),
  memoryDim: z.number().int().positive().optional(),
  transformerLayers: z.number().int().positive().optional(),
  numHeads: z.number().int().positive().optional(),
  ffDimension: z.number().int().positive().optional(),
  dropoutRate: z.number().min(0).max(0.9).optional(),
  maxSequenceLength: z.number().int().positive().optional(),
  memorySlots: z.number().int().positive().optional(),
  similarityThreshold: z.number().min(0).max(1).optional(),
  surpriseDecay: z.number().min(0).max(1).optional(),
  pruningInterval: z.number().int().positive().optional(),
  gradientClip: z.number().positive().optional(),
});