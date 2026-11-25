import type { MemoryRecord, MemoryRecordInput } from '../contracts.js';
import { cosineSimilarity } from '../utils/math.js';
import { canonicalizeText } from '../utils/text.js';

export interface DeduplicationMatch {
  record: MemoryRecord;
  reason: 'exact-canonical' | 'near-duplicate-cosine';
  similarity: number;
}

export interface DeduplicationCheckOptions {
  /** Cosine similarity threshold above which same-tier records merge instead of duplicating (default 0.94). */
  nearDuplicateThreshold?: number;
}

/**
 * Checks whether an incoming memory record duplicates an active record in the same tier.
 * Respects taint boundaries: tainted inputs never collapse into untainted `profile` records.
 */
export function findDuplicateRecord(
  input: MemoryRecordInput,
  incomingVector: Float32Array,
  candidates: Array<{ record: MemoryRecord; vector: Float32Array | null }>,
  options: DeduplicationCheckOptions = {}
): DeduplicationMatch | null {
  const threshold = options.nearDuplicateThreshold ?? 0.94;
  const incomingCanonical = canonicalizeText(input.content);
  const incomingTainted = Boolean(input.tainted);

  let bestNear: DeduplicationMatch | null = null;

  for (const { record, vector } of candidates) {
    if (record.supersededById) continue;
    if (record.tier !== input.tier) continue;
    if (record.tier === 'profile' && !record.tainted && incomingTainted) {
      continue;
    }

    if (canonicalizeText(record.content) === incomingCanonical) {
      return {
        record,
        reason: 'exact-canonical',
        similarity: 1,
      };
    }

    if (vector) {
      const sim = cosineSimilarity(incomingVector, vector);
      if (sim >= threshold && (!bestNear || sim > bestNear.similarity)) {
        bestNear = {
          record,
          reason: 'near-duplicate-cosine',
          similarity: sim,
        };
      }
    }
  }

  return bestNear;
}
