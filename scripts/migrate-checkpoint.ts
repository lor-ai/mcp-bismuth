#!/usr/bin/env npx ts-node

import * as fs from 'fs/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import { TitanMemoryModel } from '../src/model';

interface MigrationOptions {
  sourcePath: string;
  targetPath?: string;
  backup?: boolean;
  verbose?: boolean;
  dryRun?: boolean;
}

interface CheckpointVersion {
  version: string;
  format: string;
  created: string;
  modelHash?: string;
}

class CheckpointMigrator {
  private verbose: boolean = false;

  constructor(private options: MigrationOptions) {
    this.verbose = options.verbose || false;
  }

  /**
   * Migrate a checkpoint to the new format
   */
  async migrate(): Promise<void> {
    try {
      await this.validateSource();
      const version = await this.detectVersion();
      
      this.log(`Detected checkpoint version: ${version.version} (${version.format})`);
      
      if (version.format === 'titan-memory-v2') {
        this.log('Checkpoint is already in the latest format');
        return;
      }

      if (this.options.backup) {
        await this.createBackup();
      }

      await this.performMigration(version);
      
      this.log('Migration completed successfully');
    } catch (error) {
      console.error('Migration failed:', error);
      process.exit(1);
    }
  }

  /**
   * Validate that the source path exists and is readable
   */
  private async validateSource(): Promise<void> {
    try {
      const stats = await fs.stat(this.options.sourcePath);
      if (!stats.isDirectory()) {
        throw new Error('Source path must be a directory');
      }
    } catch (error) {
      throw new Error(`Cannot access source path: ${error}`);
    }
  }

  /**
   * Detect the version of the checkpoint
   */
  private async detectVersion(): Promise<CheckpointVersion> {
    const modelJsonPath = path.join(this.options.sourcePath, 'model.json');
    
    try {
      const modelData = JSON.parse(await fs.readFile(modelJsonPath, 'utf-8'));
      return {
        version: modelData.version || '1.0',
        format: modelData.format || 'titan-memory-v1',
        created: modelData.created || new Date().toISOString(),
        modelHash: modelData.modelHash
      };
    } catch (error) {
      // Legacy format detection
      const weightsPath = path.join(this.options.sourcePath, 'weights.json');
      try {
        await fs.access(weightsPath);
        return {
          version: '0.9',
          format: 'legacy',
          created: new Date().toISOString()
        };
      } catch {
        throw new Error('Unable to detect checkpoint format');
      }
    }
  }

  /**
   * Create a backup of the original checkpoint
   */
  private async createBackup(): Promise<void> {
    const backupPath = `${this.options.sourcePath}.backup.${Date.now()}`;
    this.log(`Creating backup at: ${backupPath}`);
    
    if (!this.options.dryRun) {
      await this.copyDirectory(this.options.sourcePath, backupPath);
    }
  }

  /**
   * Perform the actual migration based on detected version
   */
  private async performMigration(version: CheckpointVersion): Promise<void> {
    const targetPath = this.options.targetPath || this.generateTargetPath(version);
    
    this.log(`Migrating to: ${targetPath}`);
    
    if (this.options.dryRun) {
      this.log('[DRY RUN] Would perform migration operations');
      return;
    }

    switch (version.format) {
      case 'legacy':
        await this.migrateLegacyFormat(targetPath);
        break;
      case 'titan-memory-v1':
        await this.migrateV1ToV2(targetPath);
        break;
      default:
        throw new Error(`Unsupported format: ${version.format}`);
    }
  }

  /**
   * Generate target path with new naming convention
   */
  private generateTargetPath(version: CheckpointVersion): string {
    const modelHash = version.modelHash || this.generateModelHash();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T');
    const datePart = timestamp[0].replace(/-/g, '');
    const timePart = timestamp[1].split('-')[0].replace(/:/g, '');
    
    return path.join(
      'checkpoints',
      modelHash,
      `snapshot-${datePart}-${timePart}`
    );
  }

  /**
   * Generate a model hash based on configuration
   */
  private generateModelHash(): string {
    // Generate hash based on model configuration
    const hash = crypto.createHash('sha256');
    hash.update(JSON.stringify({
      timestamp: Date.now(),
      source: this.options.sourcePath
    }));
    return hash.digest('hex').substring(0, 16);
  }

  /**
   * Migrate from legacy format to v2
   */
  private async migrateLegacyFormat(targetPath: string): Promise<void> {
    this.log('Migrating from legacy format...');
    
    // Create target directory structure
    await fs.mkdir(targetPath, { recursive: true });
    
    // Read legacy weights
    const weightsPath = path.join(this.options.sourcePath, 'weights.json');
    const weights = JSON.parse(await fs.readFile(weightsPath, 'utf-8'));
    
    // Create new model configuration
    const modelConfig = {
      version: '2.0',
      format: 'titan-memory-v2',
      created: new Date().toISOString(),
      modelHash: this.generateModelHash(),
      config: {
        inputSize: 768,
        hiddenSize: 1024,
        memorySlots: 1000,
        learningRate: 0.001,
        // Migration defaults
        enableHierarchicalMemory: false,
        enableQuantization: false
      }
    };
    
    // Save model configuration
    await fs.writeFile(
      path.join(targetPath, 'modelConfig.json'),
      JSON.stringify(modelConfig, null, 2)
    );
    
    // Migrate weights
    await fs.writeFile(
      path.join(targetPath, 'weights.json'),
      JSON.stringify(weights, null, 2)
    );
    
    this.log('Legacy migration completed');
  }

  /**
   * Migrate from v1 to v2 format
   */
  private async migrateV1ToV2(targetPath: string): Promise<void> {
    this.log('Migrating from v1 to v2 format...');
    
    // Create target directory structure
    await fs.mkdir(targetPath, { recursive: true });
    
    // Read existing model.json
    const modelPath = path.join(this.options.sourcePath, 'model.json');
    const modelData = JSON.parse(await fs.readFile(modelPath, 'utf-8'));
    
    // Generate model hash if not present
    const modelHash = modelData.modelHash || this.generateModelHash();
    
    // Create new model configuration
    const modelConfig = {
      version: '2.0',
      format: 'titan-memory-v2',
      created: new Date().toISOString(),
      modelHash,
      config: modelData.config,
      migrated: {
        from: modelData.format || 'titan-memory-v1',
        timestamp: new Date().toISOString(),
        originalPath: this.options.sourcePath
      }
    };
    
    // Save model configuration
    await fs.writeFile(
      path.join(targetPath, 'modelConfig.json'),
      JSON.stringify(modelConfig, null, 2)
    );
    
    // Copy weights and other data
    const filesToCopy = ['encoder', 'decoder'];
    for (const file of filesToCopy) {
      const sourcePath = path.join(this.options.sourcePath, file);
      const targetFilePath = path.join(targetPath, file);
      try {
        await this.copyDirectory(sourcePath, targetFilePath);
      } catch (error) {
        this.log(`Warning: Could not copy ${file}: ${error}`);
      }
    }
    
    // Extract and save memory state
    if (modelData.memoryState) {
      await fs.writeFile(
        path.join(targetPath, 'memoryState.json'),
        JSON.stringify(modelData.memoryState, null, 2)
      );
    }
    
    // Extract and save telemetry
    if (modelData.telemetry) {
      await fs.writeFile(
        path.join(targetPath, 'telemetry.json'),
        JSON.stringify(modelData.telemetry, null, 2)
      );
    }
    
    // Create empty ANN index placeholder
    await fs.writeFile(
      path.join(targetPath, 'annIndex.json'),
      JSON.stringify({ 
        type: 'HNSW',
        built: false,
        nodes: [],
        metadata: { migrated: true }
      }, null, 2)
    );
    
    // Create empty tokenizer merges placeholder
    await fs.writeFile(
      path.join(targetPath, 'tokenizerMerges.json'),
      JSON.stringify({
        merges: [],
        vocab: {},
        metadata: { migrated: true }
      }, null, 2)
    );
    
    this.log('v1 to v2 migration completed');
  }

  /**
   * Copy directory recursively
   */
  private async copyDirectory(source: string, target: string): Promise<void> {
    await fs.mkdir(target, { recursive: true });
    const items = await fs.readdir(source);
    
    for (const item of items) {
      const sourcePath = path.join(source, item);
      const targetPath = path.join(target, item);
      const stats = await fs.stat(sourcePath);
      
      if (stats.isDirectory()) {
        await this.copyDirectory(sourcePath, targetPath);
      } else {
        await fs.copyFile(sourcePath, targetPath);
      }
    }
  }

  /**
   * Log message if verbose mode is enabled
   */
  private log(message: string): void {
    if (this.verbose) {
      console.log(`[MIGRATE] ${message}`);
    }
  }
}

/**
 * CLI interface
 */
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: migrate-checkpoint.ts [options] <source-path>

Options:
  --target <path>     Target path for migrated checkpoint
  --backup           Create backup before migration
  --verbose          Enable verbose logging
  --dry-run          Show what would be done without executing
  --help, -h         Show this help message

Examples:
  # Migrate old checkpoint to new format
  npx ts-node scripts/migrate-checkpoint.ts ./old-checkpoint --backup --verbose
  
  # Dry run migration
  npx ts-node scripts/migrate-checkpoint.ts ./old-checkpoint --dry-run --verbose
  
  # Migrate to specific target
  npx ts-node scripts/migrate-checkpoint.ts ./old-checkpoint --target ./new-checkpoint
`);
    process.exit(0);
  }

  let sourcePath = '';
  const options: MigrationOptions = {
    sourcePath: '',
    backup: false,
    verbose: false,
    dryRun: false
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    
    switch (arg) {
      case '--target':
        options.targetPath = args[++i];
        break;
      case '--backup':
        options.backup = true;
        break;
      case '--verbose':
        options.verbose = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      default:
        if (!arg.startsWith('--')) {
          sourcePath = arg;
        }
        break;
    }
  }

  if (!sourcePath) {
    console.error('Error: Source path is required');
    process.exit(1);
  }

  options.sourcePath = sourcePath;
  
  const migrator = new CheckpointMigrator(options);
  await migrator.migrate();
}

// Run CLI if this file is executed directly
if (require.main === module) {
  main().catch(error => {
    console.error('Unexpected error:', error);
    process.exit(1);
  });
}

export { CheckpointMigrator, MigrationOptions };
