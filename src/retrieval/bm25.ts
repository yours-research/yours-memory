import { tokenize } from '../utils/text.js';

export interface Bm25Document {
  id: string;
  text: string;
  boostText?: string;
}

export interface Bm25Hit {
  id: string;
  score: number;
  rank: number;
}

export interface Bm25Options {
  k1?: number;
  b?: number;
}

interface Posting {
  tf: number;
}

/**
 * Pure TypeScript BM25 inverted index matching FTS5 tokenization semantics.
 * Used directly for InMemory/IndexedDB backends and standalone lexical scoring.
 */
export class Bm25Index {
  private readonly k1: number;
  private readonly b: number;
  private readonly docLengths = new Map<string, number>();
  private readonly docTerms = new Map<string, Map<string, number>>();
  private readonly inverted = new Map<string, Map<string, Posting>>();
  private totalLength = 0;

  constructor(options: Bm25Options = {}) {
    this.k1 = options.k1 ?? 1.2;
    this.b = options.b ?? 0.75;
  }

  get size(): number {
    return this.docLengths.size;
  }

  upsert(doc: Bm25Document): void {
    if (this.docLengths.has(doc.id)) {
      this.remove(doc.id);
    }

    const baseTokens = tokenize(doc.text, {
      removeStopwords: true,
      stem: true,
      splitIdentifiers: true,
    });
    const boostTokens = doc.boostText
      ? tokenize(doc.boostText, {
          removeStopwords: true,
          stem: true,
          splitIdentifiers: true,
        })
      : [];

    const tfMap = new Map<string, number>();
    for (const token of baseTokens) {
      tfMap.set(token, (tfMap.get(token) ?? 0) + 1);
    }
    // Give title/tags slight lexical reinforcement matching multi-column FTS5 weighting.
    for (const token of boostTokens) {
      tfMap.set(token, (tfMap.get(token) ?? 0) + 1.5);
    }

    const length = baseTokens.length + boostTokens.length;
    this.docLengths.set(doc.id, length);
    this.docTerms.set(doc.id, tfMap);
    this.totalLength += length;

    for (const [term, tf] of tfMap) {
      let postings = this.inverted.get(term);
      if (!postings) {
        postings = new Map();
        this.inverted.set(term, postings);
      }
      postings.set(doc.id, { tf });
    }
  }

  remove(id: string): boolean {
    const tfMap = this.docTerms.get(id);
    const len = this.docLengths.get(id);
    if (!tfMap || len === undefined) return false;

    for (const term of tfMap.keys()) {
      const postings = this.inverted.get(term);
      if (postings) {
        postings.delete(id);
        if (postings.size === 0) {
          this.inverted.delete(term);
        }
      }
    }

    this.docTerms.delete(id);
    this.docLengths.delete(id);
    this.totalLength = Math.max(0, this.totalLength - len);
    return true;
  }

  clear(): void {
    this.docLengths.clear();
    this.docTerms.clear();
    this.inverted.clear();
    this.totalLength = 0;
  }

  search(query: string, limit = 20, filterFn?: (id: string) => boolean): Bm25Hit[] {
    const N = this.docLengths.size;
    if (N === 0 || limit <= 0) return [];

    const qTokens = tokenize(query, {
      removeStopwords: true,
      stem: true,
      splitIdentifiers: true,
    });
    const terms =
      qTokens.length > 0
        ? Array.from(new Set(qTokens))
        : Array.from(
            new Set(
              tokenize(query, {
                removeStopwords: false,
                stem: true,
                splitIdentifiers: false,
              })
            )
          );

    if (terms.length === 0) return [];

    const avgDl = this.totalLength / N || 1;
    const scores = new Map<string, number>();

    for (const term of terms) {
      const postings = this.inverted.get(term);
      if (!postings || postings.size === 0) continue;

      const df = postings.size;
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));

      for (const [docId, posting] of postings) {
        if (filterFn && !filterFn(docId)) continue;
        const dl = this.docLengths.get(docId) ?? 0;
        const tf = posting.tf;
        const denom = tf + this.k1 * (1 - this.b + (this.b * dl) / avgDl);
        const termScore = idf * ((tf * (this.k1 + 1)) / (denom || 1));
        scores.set(docId, (scores.get(docId) ?? 0) + termScore);
      }
    }

    const sorted = Array.from(scores.entries())
      .filter(([, score]) => score > 0)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, limit);

    return sorted.map(([id, score], idx) => ({
      id,
      score,
      rank: idx + 1,
    }));
  }
}
