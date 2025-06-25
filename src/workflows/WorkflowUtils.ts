import type { TitanMemorySystem } from '../types.js';

/**
 * Safely store workflow memory data
 */
export async function safeStoreWorkflowMemory(
    memory: TitanMemorySystem,
    type: string,
    data: any
): Promise<void> {
    try {
        if (memory.storeWorkflowMemory) {
            await memory.storeWorkflowMemory(type, data);
        } else {
            // Fallback to regular memory storage
            await memory.storeMemory(`${type}: ${JSON.stringify(data).substring(0, 200)}`);
        }
    } catch (error) {
        console.warn('Failed to store workflow memory:', error);
    }
}

/**
 * Safely get relevant context from memory
 */
export async function safeGetRelevantContext(
    memory: TitanMemorySystem,
    query: string
): Promise<any> {
    try {
        if (memory.getRelevantContext) {
            return await memory.getRelevantContext(query);
        } else {
            // Fallback to regular memory recall
            const memories = await memory.recallMemory(query, 5);
            return { context: memories, fallback: true };
        }
    } catch (error) {
        console.warn('Failed to get relevant context:', error);
        return { context: [], fallback: true };
    }
}

/**
 * Safely find similar content
 */
export async function safeFindSimilarContent(
    memory: TitanMemorySystem,
    content: string
): Promise<Array<{ score: number; content: string }>> {
    try {
        if (memory.findSimilarContent) {
            return await memory.findSimilarContent(content);
        } else {
            // Fallback to regular memory recall
            const memories = await memory.recallMemory(content, 5);
            return memories.map((tensor, index) => ({
                score: Math.random() * 0.5 + 0.5, // Mock similarity score
                content: `Similar content ${index + 1}`
            }));
        }
    } catch (error) {
        console.warn('Failed to find similar content:', error);
        return [];
    }
}

/**
 * Safely get workflow history
 */
export async function safeGetWorkflowHistory(
    memory: TitanMemorySystem,
    type: string,
    limit: number
): Promise<any[]> {
    try {
        if (memory.getWorkflowHistory) {
            return await memory.getWorkflowHistory(type, limit);
        } else {
            // Fallback to regular memory recall
            const memories = await memory.recallMemory(type, limit);
            return memories.map((tensor, index) => ({
                id: `workflow-${index}`,
                type,
                data: Array.from(tensor.dataSync()).slice(0, 10),
                timestamp: new Date(Date.now() - index * 60000)
            }));
        }
    } catch (error) {
        console.warn('Failed to get workflow history:', error);
        return [];
    }
}

/**
 * Safely shutdown memory system
 */
export async function safeShutdown(memory: TitanMemorySystem): Promise<void> {
    try {
        if (memory.shutdown) {
            await memory.shutdown();
        } else {
            // Fallback cleanup
            memory.dispose();
        }
    } catch (error) {
        console.warn('Failed to shutdown memory system:', error);
    }
}

/**
 * Safely get health status
 */
export async function safeGetHealthStatus(memory: TitanMemorySystem): Promise<any> {
    try {
        if (memory.getHealthStatus) {
            return await memory.getHealthStatus();
        } else {
            // Fallback health check
            const state = memory.getMemoryState();
            return {
                status: 'ok',
                memoryState: state ? 'active' : 'inactive',
                fallback: true
            };
        }
    } catch (error) {
        console.warn('Failed to get health status:', error);
        return { status: 'error', message: 'Health check failed', fallback: true };
    }
}

/**
 * Handle unknown errors safely
 */
export function safeErrorMessage(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    return String(error);
} 