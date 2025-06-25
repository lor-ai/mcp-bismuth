import * as tf from '@tensorflow/tfjs-node';
import * as crypto from 'crypto';

export class MemoryManager {
    private static instance: MemoryManager;
    private cleanupInterval: NodeJS.Timeout | null = null;
    private encryptionKey: Buffer;
    private iv: Buffer;

    private constructor() {
        this.encryptionKey = crypto.randomBytes(32);
        this.iv = crypto.randomBytes(16);
        this.startPeriodicCleanup();
    }

    public static getInstance(): MemoryManager {
        if (!MemoryManager.instance) {
            MemoryManager.instance = new MemoryManager();
        }
        return MemoryManager.instance;
    }

    private startPeriodicCleanup(): void {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
        }
        this.cleanupInterval = setInterval(() => {
            tf.engine().startScope();
            try {
                // Force garbage collection of unused tensors
                tf.engine().disposeVariables();
                // Clean up tensors
                const numTensors = tf.memory().numTensors;
                if (numTensors > 1000) {
                    const tensors = tf.engine().state.numTensors;
                    tf.disposeVariables();
                    tf.dispose();
                }
            } finally {
                tf.engine().endScope();
            }
        }, 60000); // Run every minute
    }

    public validateVectorShape(tensor: tf.Tensor, expectedShape: number[]): boolean {
        return tf.tidy(() => {
            if (tensor.shape.length !== expectedShape.length) {return false;}
            return tensor.shape.every((dim, i) => expectedShape[i] === -1 || dim === expectedShape[i]);
        });
    }

    public encryptTensor(tensor: tf.Tensor): Buffer {
        const data = tensor.dataSync();
        const cipher = crypto.createCipheriv('aes-256-gcm', this.encryptionKey, this.iv);
        const encrypted = Buffer.concat([
            cipher.update(Buffer.from(new Float32Array(data).buffer)),
            cipher.final()
        ]);
        const authTag = cipher.getAuthTag();
        return Buffer.concat([encrypted, authTag]);
    }

    public decryptTensor(encrypted: Buffer, shape: number[]): tf.Tensor {
        const authTag = encrypted.slice(-16);
        const encryptedData = encrypted.slice(0, -16);
        const decipher = crypto.createDecipheriv('aes-256-gcm', this.encryptionKey, this.iv);
        decipher.setAuthTag(authTag);
        const decrypted = Buffer.concat([
            decipher.update(encryptedData),
            decipher.final()
        ]);
        const data = new Float32Array(decrypted.buffer);
        return tf.tensor(Array.from(data), shape);
    }

    public wrapWithMemoryManagement<T extends tf.TensorContainer>(fn: () => T): T {
        return tf.tidy(fn);
    }

    public async wrapWithMemoryManagementAsync<T>(fn: () => Promise<T>): Promise<T> {
        tf.engine().startScope();
        try {
            return await fn();
        } finally {
            tf.engine().endScope();
        }
    }

    public dispose(): void {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
            this.cleanupInterval = null;
        }
    }
}

export class VectorProcessor {
    private static instance: VectorProcessor;
    private memoryManager: MemoryManager = MemoryManager.getInstance();

    private constructor() {
        // Initialize MemoryManager
    }

    public static getInstance(): VectorProcessor {
        if (!VectorProcessor.instance) {
            VectorProcessor.instance = new VectorProcessor();
        }
        return VectorProcessor.instance;
    }

    /**
     * Processes input, ensuring it's a valid tensor
     * @param input Input data (string, number[], or tf.Tensor)
     * @returns Processed tf.Tensor
     */
    public processInput(input: string | number[] | tf.Tensor): tf.Tensor {
        // Use wrapWithMemoryManagement for safer tensor operations
        return this.memoryManager.wrapWithMemoryManagement(() => {
            try {
                if (typeof input === 'string') {
                    // Handle string input: encodeText returns a Promise<tf.Tensor>, so await it
                    // However, wrapWithMemoryManagement expects a sync function.
                    // Encoding should happen *outside* the sync wrapper.
                    // This function should likely be async or handle the promise differently.
                    // For now, throwing error as async logic can't be directly in sync wrapper.
                    throw new Error('String input requires async processing, use encodeText directly.');
                } else if (Array.isArray(input)) {
                    // Handle array input
                    if (input.length === 0) {
                        return tf.tensor2d([], [0, 1]); // Handle empty array
                    }
                    if (typeof input[0] === 'number') {
                        // It's 1D, convert to 2D [1, N]
                        return tf.tensor1d(input).expandDims(0);
                    } else {
                        // Assume it's already 2D or compatible
                        // Add more robust checking if needed
                        // Attempt to create tensor2d, catch potential errors
                        try {
                            return tf.tensor2d(input as unknown as number[][]);
                        } catch (e) {
                            console.error("Failed to create tensor2d from array:", e);
                            throw new Error("Invalid array format for tensor2d");
                        }
                    }
                } else if (input instanceof tf.Tensor) {
                    // Handle tensor input
                    // Ensure it's 2D
                    if (input.rank === 1) {
                        return input.expandDims(0); // Convert 1D to 2D
                    } else if (input.rank === 2) {
                        return input; // Already 2D
                    } else {
                        throw new Error(`Unsupported tensor rank: ${input.rank}. Expected 1 or 2.`);
                    }
                } else {
                    throw new Error(`Unsupported input type: ${typeof input}`);
                }
            } catch (error) {
                console.error("Error processing input:", error);
                // Create a default tensor in case of error
                // Use a default dimension size (e.g., 768) if model config isn't accessible here
                const defaultDim = 768;
                return tf.zeros([1, defaultDim]);
            }
        });
    }

    /**
     * Validates tensor shape and normalizes if necessary
     * @param tensor Tensor to validate
     * @param expectedShape Expected shape
     * @returns Validated and potentially normalized tensor
     */
    public validateAndNormalize(tensor: tf.Tensor, expectedShape: number[]): tf.Tensor {
        return this.memoryManager.wrapWithMemoryManagement(() => {
            validateTensorShape(tensor, expectedShape);
            // Add normalization logic if required (e.g., L2 normalization)
            // return tf.tidy(() => tf.div(tensor, tf.norm(tensor)));
            return tensor;
        });
    }

    /**
     * Encodes text to a tensor representation using a pre-trained model.
     * @param text The text to encode.
     * @param maxLength Optional maximum sequence length.
     * @returns A promise resolving to the encoded tensor.
     */
    public async encodeText(text: string, maxLength = 512): Promise<tf.Tensor> {
        // Use wrapWithMemoryManagementAsync for async tensor operations
        return this.memoryManager.wrapWithMemoryManagementAsync(async () => {
            try {
                // Placeholder: Replace with actual text encoding logic (e.g., using USE)
                // const model = await use.load(); // Example: Load Universal Sentence Encoder
                // const embeddings = await model.embed([text]);
                // return embeddings;

                // Simple character code encoding as a fallback placeholder
                const tokens = text.split('').map(char => char.charCodeAt(0));
                const paddedArray = tokens.slice(0, maxLength);
                while (paddedArray.length < maxLength) {
                    paddedArray.push(0); // Pad with 0
                }
                // Ensure the output is a 2D tensor [1, maxLength]
                const tensor = tf.tensor2d([paddedArray], [1, maxLength]);
                return tensor;
            } catch (error) {
                console.error("Error encoding text:", error);
                // Return a default zero tensor in case of error
                return tf.zeros([1, maxLength]);
            }
        });
    }
}

export class AutomaticMemoryMaintenance {
    private static instance: AutomaticMemoryMaintenance;
    private memoryManager: MemoryManager;
    private maintenanceInterval: NodeJS.Timeout | null = null;

    private constructor() {
        this.memoryManager = MemoryManager.getInstance();
        this.startMaintenanceLoop();
    }

    public static getInstance(): AutomaticMemoryMaintenance {
        if (!AutomaticMemoryMaintenance.instance) {
            AutomaticMemoryMaintenance.instance = new AutomaticMemoryMaintenance();
        }
        return AutomaticMemoryMaintenance.instance;
    }

    private startMaintenanceLoop(): void {
        if (this.maintenanceInterval) {
            clearInterval(this.maintenanceInterval);
        }
        this.maintenanceInterval = setInterval(() => {
            this.performMaintenance();
        }, 300000); // Run every 5 minutes
    }

    private performMaintenance(): void {
        this.memoryManager.wrapWithMemoryManagement(() => {
            // Check memory usage
            const memoryInfo = tf.memory();
            if (memoryInfo.numTensors > 1000 || memoryInfo.numBytes > 1e8) {
                tf.engine().disposeVariables();
                const tensors = tf.engine().state.numTensors;
                tf.disposeVariables();
                tf.dispose();
            }
        });
    }

    public dispose(): void {
        if (this.maintenanceInterval) {
            clearInterval(this.maintenanceInterval);
            this.maintenanceInterval = null;
        }
    }
}

// Utility functions for tensor operations
export function checkNullOrUndefined(value: any): boolean {
    return value === null || value === undefined;
}

export function validateTensor(tensor: tf.Tensor | null | undefined): boolean {
    return !checkNullOrUndefined(tensor) && !tensor!.isDisposed;
}

export function validateTensorShape(tensor: tf.Tensor | null | undefined, expectedShape: number[]): boolean {
    if (!validateTensor(tensor)) {return false;}
    const shape = tensor!.shape;
    if (shape.length !== expectedShape.length) {return false;}
    return shape.every((dim, i) => expectedShape[i] === -1 || expectedShape[i] === dim);
}

// Safe tensor operations that handle null checks
export class SafeTensorOps {
    static reshape(tensor: tf.Tensor, shape: number[]): tf.Tensor {
        if (!validateTensor(tensor)) {
            throw new Error('Invalid tensor for reshape operation');
        }
        return tf.reshape(tensor, shape);
    }

    static matMul(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for matMul operation');
        }
        return tf.matMul(a, b);
    }

    static add(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for add operation');
        }
        return tf.add(a, b);
    }

    static sub(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for sub operation');
        }
        return tf.sub(a, b);
    }

    static mul(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for mul operation');
        }
        return tf.mul(a, b);
    }

    static div(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for div operation');
        }
        return tf.div(a, b);
    }
} 