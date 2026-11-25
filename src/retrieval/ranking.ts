import type { MemoryHit, MemoryRecord, RankingConfig, TierDecayConfig } from '../contracts.js';
import type { LexicalSearchResult } from '../storage/types.js';
import type { HnswSearchResult } from './hnsw.js';

const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;

export const DEFAULT_RANKING_CONFIG: RankingConfig = {
  rrfK: 60,
  lexicalWeight: 1.0,
  vectorWeight: 1.0,
  recencyMix: 0.35,
  importanceBoost: 0.25,
  accessBoost: 0.08,
  taintedPenalty: 0.85,
  decayByTier: {
    // Durable user facts/preferences resist decay strongly.
    profile: { halfLifeMs: 180 * DAY_MS, floor: 0.65 },
    // Opt-in indexed web pages / PDF chunks decay slowly.
    document: { halfLifeMs: 60 * DAY_MS, floor: 0.25 },
    // Consolidated session summaries decay moderately.
    episodic: { halfLifeMs: 30 * DAY_MS, floor: 0.15 },
    // Ephemeral working tab/goal graph decays quickly.
    working: { halfLifeMs: 12 * HOUR_MS, floor: 0.05 },
  },
};

/**
 * Exponential half-life temporal salience decay bounded by tier-specific floor.
 * Uses `max(lastAccessedAt, updatedAt)` so frequently recalled memories stay salient.
 */
export function computeRecencyWeight(
  record: MemoryRecord,
  tierConfig: TierDecayConfig,
  nowMs: number
): number {
  const referenceMs = Math.max(record.lastAccessedAt, record.updatedAt, record.createdAt);
  const ageMs = Math.max(0, nowMs - referenceMs);
  if (tierConfig.halfLifeMs <= 0) return tierConfig.floor;
  const decay = Math.pow(0.5, ageMs / tierConfig.halfLifeMs);
  return Math.max(tierConfig.floor, Math.min(1, decay));
}

export interface HybridFusionInput {
  recordsById: Map<string, MemoryRecord>;
  lexicalHits: LexicalSearchResult[];
  vectorHits: HnswSearchResult[];
  nowMs: number;
  limit: number;
  minScore: number;
  config?: Partial<RankingConfig>;
}

/**
 * Combine BM25 lexical ranks and HNSW semantic ranks via Reciprocal Rank Fusion (RRF),
 * weighted by temporal salience decay, record importance, access reinforcement, and taint.
 */
export function fuseHybridRanks(input: HybridFusionInput): MemoryHit[] {
  const cfg: RankingConfig = {
    ...DEFAULT_RANKING_CONFIG,
    ...input.config,
    decayByTier: {
      ...DEFAULT_RANKING_CONFIG.decayByTier,
      ...input.config?.decayByTier,
    },
  };

  const bm25ById = new Map<string, LexicalSearchResult>();
  for (const hit of input.lexicalHits) {
    bm25ById.set(hit.id, hit);
  }

  const vecById = new Map<string, HnswSearchResult>();
  for (const hit of input.vectorHits) {
    vecById.set(hit.id, hit);
  }

  const candidateIds = new Set<string>([...bm25ById.keys(), ...vecById.keys()]);
  const hits: MemoryHit[] = [];

  for (const id of candidateIds) {
    const record = input.recordsById.get(id);
    if (!record || record.supersededById) continue;

    const bm25 = bm25ById.get(id);
    const vec = vecById.get(id);

    // Reciprocal Rank Fusion score normalized against theoretical dual-rank #1 maximum
    const bm25Rrf = bm25 ? cfg.lexicalWeight / (cfg.rrfK + bm25.rank) : 0;
    const vecRrf = vec ? cfg.vectorWeight / (cfg.rrfK + vec.rank) : 0;
    const maxRrf = (cfg.lexicalWeight + cfg.vectorWeight) / (cfg.rrfK + 1);
    const normalizedRrf = maxRrf > 0 ? (bm25Rrf + vecRrf) / maxRrf : 0;

    // Incorporate semantic cosine similarity signal when present to break ties cleanly
    const simSignal = vec ? Math.max(0, vec.similarity) * 0.15 : 0;

    const tierDecay = cfg.decayByTier[record.tier];
    const recencyWeight = computeRecencyWeight(record, tierDecay, input.nowMs);

    const importance = Math.max(0, Math.min(1, record.importance ?? 0.5));
    const accessReinforcement = 1 + cfg.accessBoost * Math.log1p(Math.max(0, record.accessCount - 1));
    const salienceMultiplier =
      (1 - cfg.recencyMix + cfg.recencyMix * recencyWeight) *
      (1 + cfg.importanceBoost * (importance - 0.5)) *
      accessReinforcement *
      (record.tainted ? cfg.taintedPenalty : 1);

    const finalScore = (normalizedRrf + simSignal) * salienceMultiplier;

    if (finalScore < input.minScore) continue;

    hits.push({
      record,
      score: finalScore,
      bm25Rank: bm25?.rank,
      vectorRank: vec?.rank,
      recencyWeight,
    });
  }

  hits.sort((a, b) => b.score - a.score || a.record.id.localeCompare(b.record.id));
  return hits.slice(0, Math.max(0, input.limit));
}
