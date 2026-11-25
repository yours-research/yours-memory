import { describe, expect, it } from 'vitest';
import type { MemoryRecord } from '../src/contracts.js';
import { Bm25Index, computeRecencyWeight, DEFAULT_RANKING_CONFIG, fuseHybridRanks } from '../src/index.js';

function makeRecord(partial: Partial<MemoryRecord> & { id: string; tier: MemoryRecord['tier']; content: string }): MemoryRecord {
  const now = 1_700_000_000_000;
  return {
    createdAt: now,
    updatedAt: now,
    lastAccessedAt: now,
    accessCount: 1,
    importance: 0.5,
    ...partial,
  };
}

describe('BM25 lexical scoring & RRF temporal ranking', () => {
  it('splits camelCase and snake_case identifiers cleanly in BM25', () => {
    const bm25 = new Bm25Index();
    bm25.upsert({ id: 'a', text: 'OPFS sync_access_handle persistence in DedicatedWorker' });
    bm25.upsert({ id: 'b', text: 'DOM getElementById lookup performance notes' });

    const hits = bm25.search('sync access handle');
    expect(hits[0]?.id).toBe('a');
    expect(hits[0]?.rank).toBe(1);

    const domHits = bm25.search('element lookup');
    expect(domHits[0]?.id).toBe('b');
  });

  it('applies tier-specific half-lives: profile resists decay while working decays rapidly', () => {
    const t0 = 1_700_000_000_000;
    const sevenDaysLater = t0 + 7 * 86_400_000;

    const profileRec = makeRecord({ id: 'p1', tier: 'profile', content: 'User prefers Rust', lastAccessedAt: t0, updatedAt: t0, createdAt: t0 });
    const workingRec = makeRecord({ id: 'w1', tier: 'working', content: 'Active tab is Rust docs', lastAccessedAt: t0, updatedAt: t0, createdAt: t0 });

    const pWeight = computeRecencyWeight(profileRec, DEFAULT_RANKING_CONFIG.decayByTier.profile, sevenDaysLater);
    const wWeight = computeRecencyWeight(workingRec, DEFAULT_RANKING_CONFIG.decayByTier.working, sevenDaysLater);

    expect(pWeight).toBeGreaterThan(0.95);
    expect(wWeight).toBeLessThanOrEqual(0.06);
  });

  it('combines BM25 + vector ranks via RRF and boosts important / frequently accessed records', () => {
    const now = 1_700_000_000_000;
    const records = new Map<string, MemoryRecord>([
      [
        'dual_hit',
        makeRecord({
          id: 'dual_hit',
          tier: 'profile',
          content: 'Prefers local WebGPU inference',
          importance: 0.9,
          accessCount: 5,
          lastAccessedAt: now,
        }),
      ],
      [
        'lexical_only',
        makeRecord({
          id: 'lexical_only',
          tier: 'working',
          content: 'WebGPU tab opened',
          importance: 0.2,
          accessCount: 1,
          lastAccessedAt: now - 3 * 86_400_000,
          updatedAt: now - 3 * 86_400_000,
          createdAt: now - 3 * 86_400_000,
        }),
      ],
      [
        'vector_tainted',
        makeRecord({
          id: 'vector_tainted',
          tier: 'document',
          content: 'Untrusted web page snippet about GPU',
          tainted: true,
          lastAccessedAt: now,
        }),
      ],
    ]);

    const hits = fuseHybridRanks({
      recordsById: records,
      lexicalHits: [
        { id: 'lexical_only', score: 3.1, rank: 1 },
        { id: 'dual_hit', score: 2.8, rank: 2 },
      ],
      vectorHits: [
        { id: 'dual_hit', similarity: 0.92, distance: 0.08, rank: 1 },
        { id: 'vector_tainted', similarity: 0.85, distance: 0.15, rank: 2 },
      ],
      nowMs: now,
      limit: 10,
      minScore: 0,
    });

    expect(hits.map((h) => h.record.id)).toEqual(['dual_hit', 'vector_tainted', 'lexical_only']);
    expect(hits[0]?.bm25Rank).toBe(2);
    expect(hits[0]?.vectorRank).toBe(1);
    expect(hits[0]?.recencyWeight).toBeCloseTo(1, 3);
  });
});
