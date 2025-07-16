/**
 * @fileoverview Self-bootstrapping Byte Pair Encoding (BPE) tokenizer
 * Learns merges from incoming text and persists them to merges.json
 */

import * as fs from 'fs/promises';
import * as path from 'path';

export interface BPEMerge {
  pair: [string, string];
  merged: string;
  frequency: number;
}

export interface BPEConfig {
  vocabSize: number;
  mergesFile?: string;
  specialTokens?: string[];
  useCharFallback?: boolean;
}

export class BPETokenizer {
  private merges: Map<string, string> = new Map();
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private mergeFrequencies: Map<string, number> = new Map();
  private config: BPEConfig;
  private mergesFile: string;
  private isDirty = false;
  private saveTimeout: NodeJS.Timeout | null = null;

  // Special tokens
  private readonly SPECIAL_TOKENS = {
    PAD: '<PAD>',
    UNK: '<UNK>',
    CLS: '<CLS>',
    SEP: '<SEP>',
    MASK: '<MASK>'
  };

  constructor(config: BPEConfig) {
    this.config = {
      useCharFallback: true,
      ...config
    };
    
    this.mergesFile = config.mergesFile || path.join(process.cwd(), 'merges.json');
    this.initializeVocab();
  }

  /**
   * Initialize vocabulary with special tokens and characters
   */
  private initializeVocab(): void {
    this.vocab.clear();
    this.reverseVocab.clear();

    // Add special tokens
    let tokenId = 0;
    for (const [name, token] of Object.entries(this.SPECIAL_TOKENS)) {
      this.vocab.set(token, tokenId);
      this.reverseVocab.set(tokenId, token);
      tokenId++;
    }

    // Add user-defined special tokens
    if (this.config.specialTokens) {
      for (const token of this.config.specialTokens) {
        if (!this.vocab.has(token)) {
          this.vocab.set(token, tokenId);
          this.reverseVocab.set(tokenId, token);
          tokenId++;
        }
      }
    }

    // Add basic characters (bytes 0-255)
    for (let i = 0; i < 256; i++) {
      const char = String.fromCharCode(i);
      if (!this.vocab.has(char)) {
        this.vocab.set(char, tokenId);
        this.reverseVocab.set(tokenId, char);
        tokenId++;
      }
    }
  }

  /**
   * Load existing merges from file
   */
  public async loadMerges(): Promise<void> {
    try {
      const mergesContent = await fs.readFile(this.mergesFile, 'utf8');
      const mergesData = JSON.parse(mergesContent);
      
      this.merges.clear();
      this.mergeFrequencies.clear();
      
      if (mergesData.merges) {
        for (const merge of mergesData.merges) {
          const key = `${merge.pair[0]} ${merge.pair[1]}`;
          this.merges.set(key, merge.merged);
          this.mergeFrequencies.set(key, merge.frequency || 1);
          
          // Add merged token to vocab if not present
          if (!this.vocab.has(merge.merged)) {
            const tokenId = this.vocab.size;
            this.vocab.set(merge.merged, tokenId);
            this.reverseVocab.set(tokenId, merge.merged);
          }
        }
      }
      
      console.log(`Loaded ${this.merges.size} BPE merges from ${this.mergesFile}`);
    } catch (error) {
      console.log(`No existing merges file found at ${this.mergesFile}, starting fresh`);
    }
  }

  /**
   * Save merges to file (debounced)
   */
  private async saveMerges(): Promise<void> {
    if (this.saveTimeout) {
      clearTimeout(this.saveTimeout);
    }

    this.saveTimeout = setTimeout(async () => {
      if (!this.isDirty) return;

      try {
        const mergesData = {
          merges: Array.from(this.merges.entries()).map(([key, merged]) => {
            const [first, second] = key.split(' ');
            return {
              pair: [first, second],
              merged,
              frequency: this.mergeFrequencies.get(key) || 1
            };
          }),
          vocabSize: this.vocab.size,
          timestamp: new Date().toISOString()
        };

        await fs.writeFile(this.mergesFile, JSON.stringify(mergesData, null, 2));
        this.isDirty = false;
        console.log(`Saved ${this.merges.size} BPE merges to ${this.mergesFile}`);
      } catch (error) {
        console.error('Error saving merges:', error);
      }
    }, 5000); // Debounce saves by 5 seconds
  }

  /**
   * Get word frequencies from text
   */
  private getWordFrequencies(text: string): Map<string, number> {
    const wordFreq = new Map<string, number>();
    
    // Simple word splitting - could be enhanced with better tokenization
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);

    for (const word of words) {
      // Convert word to character sequence with end marker
      const charSeq = word.split('').join(' ') + ' </w>';
      wordFreq.set(charSeq, (wordFreq.get(charSeq) || 0) + 1);
    }

    return wordFreq;
  }

  /**
   * Get pair frequencies from word frequencies
   */
  private getPairFrequencies(wordFreqs: Map<string, number>): Map<string, number> {
    const pairFreq = new Map<string, number>();

    for (const [word, freq] of wordFreqs) {
      const chars = word.split(' ');
      for (let i = 0; i < chars.length - 1; i++) {
        const pair = `${chars[i]} ${chars[i + 1]}`;
        pairFreq.set(pair, (pairFreq.get(pair) || 0) + freq);
      }
    }

    return pairFreq;
  }

  /**
   * Apply a merge to word frequencies
   */
  private applyMerge(
    wordFreqs: Map<string, number>,
    pair: string,
    merged: string
  ): Map<string, number> {
    const [first, second] = pair.split(' ');
    const newWordFreqs = new Map<string, number>();

    for (const [word, freq] of wordFreqs) {
      const newWord = word.replace(new RegExp(`${first} ${second}`, 'g'), merged);
      newWordFreqs.set(newWord, freq);
    }

    return newWordFreqs;
  }

  /**
   * Learn BPE merges from text
   */
  public async learnFromText(text: string, maxMerges = 1000): Promise<void> {
    const wordFreqs = this.getWordFrequencies(text);
    let currentWordFreqs = new Map(wordFreqs);

    for (let i = 0; i < maxMerges && this.vocab.size < this.config.vocabSize; i++) {
      const pairFreqs = this.getPairFrequencies(currentWordFreqs);
      
      if (pairFreqs.size === 0) break;

      // Find most frequent pair
      let maxFreq = 0;
      let bestPair = '';
      
      for (const [pair, freq] of pairFreqs) {
        if (freq > maxFreq) {
          maxFreq = freq;
          bestPair = pair;
        }
      }

      if (maxFreq < 2) break; // Stop if no pair appears more than once

      // Create merged token
      const [first, second] = bestPair.split(' ');
      const merged = first + second;

      // Add to merges and vocab
      this.merges.set(bestPair, merged);
      this.mergeFrequencies.set(bestPair, maxFreq);

      if (!this.vocab.has(merged)) {
        const tokenId = this.vocab.size;
        this.vocab.set(merged, tokenId);
        this.reverseVocab.set(tokenId, merged);
      }

      // Apply merge to word frequencies
      currentWordFreqs = this.applyMerge(currentWordFreqs, bestPair, merged);
    }

    this.isDirty = true;
    await this.saveMerges();
  }

  /**
   * Apply BPE encoding to a word
   */
  private applyBPE(word: string): string[] {
    if (word.length <= 1) return [word];

    // Start with character sequence
    let tokens = word.split('').map(char => char);
    tokens[tokens.length - 1] += '</w>'; // Add end marker

    while (tokens.length > 1) {
      let bestPair = '';
      let bestMerged = '';
      let bestPos = -1;

      // Find the best merge to apply
      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = `${tokens[i]} ${tokens[i + 1]}`;
        if (this.merges.has(pair)) {
          const merged = this.merges.get(pair)!;
          const freq = this.mergeFrequencies.get(pair) || 0;
          
          if (freq > 0 && (bestPair === '' || freq > (this.mergeFrequencies.get(bestPair) || 0))) {
            bestPair = pair;
            bestMerged = merged;
            bestPos = i;
          }
        }
      }

      if (bestPair === '') break;

      // Apply the merge
      const newTokens = [
        ...tokens.slice(0, bestPos),
        bestMerged,
        ...tokens.slice(bestPos + 2)
      ];
      tokens = newTokens;
    }

    return tokens;
  }

  /**
   * Encode text to token IDs
   */
  public encode(text: string): number[] {
    const tokens: number[] = [];
    
    // Simple word splitting
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);

    for (const word of words) {
      const bpeTokens = this.applyBPE(word);
      
      for (const token of bpeTokens) {
        const tokenId = this.vocab.get(token);
        if (tokenId !== undefined) {
          tokens.push(tokenId);
        } else if (this.config.useCharFallback) {
          // Fallback to character encoding
          for (const char of token) {
            const charId = this.vocab.get(char);
            if (charId !== undefined) {
              tokens.push(charId);
            } else {
              tokens.push(this.vocab.get(this.SPECIAL_TOKENS.UNK)!);
            }
          }
        } else {
          tokens.push(this.vocab.get(this.SPECIAL_TOKENS.UNK)!);
        }
      }
    }

    return tokens;
  }

  /**
   * Decode token IDs to text
   */
  public decode(tokenIds: number[]): string {
    const tokens = tokenIds
      .map(id => this.reverseVocab.get(id))
      .filter(token => token !== undefined)
      .map(token => token!);

    // Join tokens and clean up
    return tokens
      .join('')
      .replace(/\<\/w\>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * Get vocabulary size
   */
  public getVocabSize(): number {
    return this.vocab.size;
  }

  /**
   * Get special token ID
   */
  public getSpecialTokenId(name: keyof typeof this.SPECIAL_TOKENS): number {
    const token = this.SPECIAL_TOKENS[name];
    return this.vocab.get(token) || 0;
  }

  /**
   * Get statistics about the tokenizer
   */
  public getStats(): {
    vocabSize: number;
    mergesCount: number;
    specialTokensCount: number;
  } {
    return {
      vocabSize: this.vocab.size,
      mergesCount: this.merges.size,
      specialTokensCount: Object.keys(this.SPECIAL_TOKENS).length
    };
  }

  /**
   * Clean up resources
   */
  public getMerges(): Array<[string, string]> {
    return Array.from(this.merges.entries());
  }

  public getVocab(): Record<string, number> {
    return Object.fromEntries(this.vocab.entries());
  }

  public dispose(): void {
    if (this.saveTimeout) {
      clearTimeout(this.saveTimeout);
      this.saveTimeout = null;
    }
  }
}
