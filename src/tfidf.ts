import * as tf from '@tensorflow/tfjs-node';

/**
 * TF-IDF Vectorizer and Cosine Similarity Computation
 */
export class TfIdfVectorizer {
  private documentTermMatrix: tf.Tensor2D | null = null;
  private vocabulary: string[] = [];
  private idfVector: tf.Tensor1D | null = null;

  /**
   * Fit the vectorizer with the given documents
   * @param documents An array of strings, each string is a document
   */
  public fit(documents: string[]): void {
    const tokenizedDocuments = documents.map(doc => this.tokenize(doc));
    this.buildVocabulary(tokenizedDocuments);
    this.buildDocumentTermMatrix(tokenizedDocuments);
    this.computeIdfVector();
  }

  /**
   * Transform documents into TF-IDF vectors
   * @param documents An array of strings
   * @returns Tensor containing TF-IDF vectors
   */
  public transform(documents: string[]): tf.Tensor2D {
    const tokenizedDocuments = documents.map(doc => this.tokenize(doc));
    const termFrequencyMatrix = this.buildTermFrequencyMatrix(tokenizedDocuments);
    return tf.mul(termFrequencyMatrix, this.idfVector!);
  }

  /**
   * Compute cosine similarity between two vectors
   * @param vectorA Tensor1D
   * @param vectorB Tensor1D
   * @returns Cosine similarity
   */
  public static cosineSimilarity(vectorA: tf.Tensor1D, vectorB: tf.Tensor1D): number {
    const dotProduct = tf.dot(vectorA, vectorB).dataSync()[0];
    const normA = tf.norm(vectorA).dataSync()[0];
    const normB = tf.norm(vectorB).dataSync()[0];
    return dotProduct / (normA * normB);
  }

  /**
   * Tokenize input text into words
   * @param text Input text
   * @returns An array of words
   */
  private tokenize(text: string): string[] {
    return text.toLowerCase().match(/\b(\w+)\b/g) || [];
  }

  /**
   * Build the vocabulary from tokenized documents
   * @param tokenizedDocuments An array of array of words
   */
  private buildVocabulary(tokenizedDocuments: string[][]): void {
    const vocabSet: Set<string> = new Set();
    tokenizedDocuments.flat().forEach(word => vocabSet.add(word));
    this.vocabulary = Array.from(vocabSet).sort();
  }

  /**
   * Build the document-term matrix
   * @param tokenizedDocuments Tokenized documents
   */
  private buildDocumentTermMatrix(tokenizedDocuments: string[][]): void {
    const termFrequencyMatrix = this.buildTermFrequencyMatrix(tokenizedDocuments);
    this.documentTermMatrix = termFrequencyMatrix;
  }

  /**
   * Build term frequency matrix
   * @param tokenizedDocuments Tokenized documents
   * @returns Term frequency matrix
   */
  private buildTermFrequencyMatrix(tokenizedDocuments: string[][]): tf.Tensor2D {
    const matrixData = tokenizedDocuments.map(doc => {
      const vector = new Array(this.vocabulary.length).fill(0);
      doc.forEach(word => {
        const index = this.vocabulary.indexOf(word);
        if (index !== -1) vector[index] += 1;
      });
      return vector;
    });
    return tf.tensor2d(matrixData);
  }

  /**
   * Compute the inverse document frequency vector
   */
  private computeIdfVector(): void {
    const docCount = this.documentTermMatrix!.shape[0];
    const df = tf.sum(tf.cast(this.documentTermMatrix!.greater(0), 'float32'), 0);
    this.idfVector = tf.log(tf.div(tf.scalar(docCount), tf.add(df, 1))) as tf.Tensor1D;
  }
}

