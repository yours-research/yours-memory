import type { MemoryRecord, MemoryRecordInput } from '../contracts.js';
import { canonicalizeText, tokenize } from '../utils/text.js';

export interface AssertionSlot {
  subject: string;
  predicate: string;
  value: string;
  polarity: 'positive' | 'negative';
}

const NEGATION_TOKENS = new Set(['not', 'no', 'never', 'dislike', 'hate', 'avoid', 'stop', 'dont', "don't"]);

const SLOT_PATTERNS: Array<{
  regex: RegExp;
  extract: (m: RegExpMatchArray) => AssertionSlot | null;
}> = [
  {
    // e.g. "My favorite editor is Vim" / "My preferred theme is dark mode"
    regex: /\bmy\s+(?:favorite|preferred|default|primary|current)\s+([a-z0-9 _-]{2,32}?)\s+(?:is|are)\s+(.+)/i,
    extract: (m) => {
      const attr = canonicalizeText(m[1] ?? '');
      const val = canonicalizeText(m[2] ?? '');
      if (!attr || !val) return null;
      return { subject: 'user', predicate: `attr:${attr}`, value: val, polarity: 'positive' };
    },
  },
  {
    // e.g. "My name is Taylor" / "My timezone is EST"
    regex: /\bmy\s+(name|timezone|time zone|location|city|role|job|editor|ide|browser|theme|language|package manager)\s+is\s+(.+)/i,
    extract: (m) => {
      const attr = canonicalizeText(m[1] ?? '');
      const val = canonicalizeText(m[2] ?? '');
      if (!attr || !val) return null;
      return { subject: 'user', predicate: `attr:${attr}`, value: val, polarity: 'positive' };
    },
  },
  {
    // e.g. "I don't like X" / "I dislike X" / "I prefer X" / "I use X" / "I live in X" / "User prefers X"
    regex: /\b(?:i|user)\s+(do not|don't|never|no longer)?\s*(prefer|use|like|love|dislike|hate|avoid|live in|work at|work on|want|choose)\s+(.+)/i,
    extract: (m) => {
      const neg = Boolean(m[1]);
      let verb = canonicalizeText(m[2] ?? '');
      const obj = canonicalizeText(m[3] ?? '');
      if (!verb || !obj) return null;

      let polarity: 'positive' | 'negative' = neg ? 'negative' : 'positive';
      if (verb === 'dislike' || verb === 'hate' || verb === 'avoid') {
        polarity = 'negative';
        verb = 'like';
      } else if (verb === 'prefer' || verb === 'love' || verb === 'like') {
        verb = 'preference';
      }
      return { subject: 'user', predicate: `verb:${verb}`, value: obj, polarity };
    },
  },
  {
    // e.g. "Prefers dark mode" / "Uses pnpm instead of npm"
    regex: /^(?:prefers|uses|likes|dislikes|avoids)\s+(.+)/i,
    extract: (m) => {
      const raw = m[0];
      const lower = raw.toLowerCase();
      const neg = lower.startsWith('dislikes') || lower.startsWith('avoids');
      const predicate = lower.startsWith('uses') ? 'verb:use' : 'verb:preference';
      const val = canonicalizeText(m[1] ?? '');
      if (!val) return null;
      return { subject: 'user', predicate, value: val, polarity: neg ? 'negative' : 'positive' };
    },
  },
];

const ANTONYM_PAIRS: ReadonlyArray<readonly [string, string]> = [
  ['dark', 'light'],
  ['tabs', 'spaces'],
  ['vim', 'vscode'],
  ['vim', 'emacs'],
  ['npm', 'pnpm'],
  ['npm', 'yarn'],
  ['pnpm', 'yarn'],
  ['bun', 'npm'],
  ['typescript', 'javascript'],
  ['macos', 'windows'],
  ['macos', 'linux'],
  ['windows', 'linux'],
  ['chrome', 'firefox'],
  ['chrome', 'safari'],
  ['metric', 'imperial'],
  ['concise', 'verbose'],
  ['morning', 'evening'],
];

export function extractAssertionSlot(content: string): AssertionSlot | null {
  const cleaned = content.trim();
  for (const pat of SLOT_PATTERNS) {
    const match = cleaned.match(pat.regex);
    if (match) {
      const slot = pat.extract(match);
      if (slot) return slot;
    }
  }
  return null;
}

function hasPolarityFlip(a: string, b: string): boolean {
  const tokensA = tokenize(a, { removeStopwords: false, stem: true });
  const tokensB = tokenize(b, { removeStopwords: false, stem: true });
  const negA = tokensA.some((t) => NEGATION_TOKENS.has(t));
  const negB = tokensB.some((t) => NEGATION_TOKENS.has(t));
  if (negA === negB) return false;

  const contentA = new Set(tokensA.filter((t) => !NEGATION_TOKENS.has(t)));
  const contentB = new Set(tokensB.filter((t) => !NEGATION_TOKENS.has(t)));
  let overlap = 0;
  for (const t of contentA) {
    if (contentB.has(t)) overlap++;
  }
  const minSize = Math.min(contentA.size, contentB.size);
  return minSize > 0 && overlap / minSize >= 0.5;
}

function hasAntonymConflict(a: string, b: string): boolean {
  const setA = new Set(tokenize(a, { removeStopwords: true, stem: false }));
  const setB = new Set(tokenize(b, { removeStopwords: true, stem: false }));
  for (const [x, y] of ANTONYM_PAIRS) {
    if ((setA.has(x) && setB.has(y)) || (setA.has(y) && setB.has(x))) {
      return true;
    }
  }
  return false;
}

export interface ContradictionMatch {
  record: MemoryRecord;
  reason: 'explicit-supersede' | 'slot-update' | 'polarity-flip' | 'antonym-conflict';
}

/**
 * Detects active records in the same tier that are contradicted by `input`, enabling
 * non-destructive superseding (`supersededById` + provenance edge) instead of silent overwrite.
 */
export function findContradictions(
  input: MemoryRecordInput,
  activeRecords: MemoryRecord[]
): ContradictionMatch[] {
  const matches: ContradictionMatch[] = [];
  const seenIds = new Set<string>();

  if (input.supersedesId) {
    const explicit = activeRecords.find((r) => r.id === input.supersedesId && !r.supersededById);
    if (explicit) {
      matches.push({ record: explicit, reason: 'explicit-supersede' });
      seenIds.add(explicit.id);
    }
  }

  // Automatic contradiction detection focuses on mutable facts/preferences in `profile` and `working`.
  if (input.tier !== 'profile' && input.tier !== 'working') {
    return matches;
  }

  const incomingSlot = extractAssertionSlot(input.content);
  const incomingCanon = canonicalizeText(input.content);

  for (const existing of activeRecords) {
    if (existing.supersededById || seenIds.has(existing.id)) continue;
    if (existing.tier !== input.tier) continue;
    if (canonicalizeText(existing.content) === incomingCanon) continue;

    // 1. Explicit negation flip ("I like TypeScript" vs "I don't like TypeScript")
    if (hasPolarityFlip(existing.content, input.content)) {
      matches.push({ record: existing, reason: 'polarity-flip' });
      seenIds.add(existing.id);
      continue;
    }

    // 2. Structured slot comparison ("My favorite editor is Vim" vs "My favorite editor is VS Code")
    const existingSlot = extractAssertionSlot(existing.content);
    if (incomingSlot && existingSlot && incomingSlot.predicate === existingSlot.predicate) {
      if (incomingSlot.predicate.startsWith('attr:')) {
        if (incomingSlot.value !== existingSlot.value || incomingSlot.polarity !== existingSlot.polarity) {
          matches.push({ record: existing, reason: 'slot-update' });
          seenIds.add(existing.id);
          continue;
        }
      } else if (incomingSlot.predicate === 'verb:preference' || incomingSlot.predicate === 'verb:use') {
        const sameTarget =
          incomingSlot.value === existingSlot.value ||
          incomingSlot.value.includes(existingSlot.value) ||
          existingSlot.value.includes(incomingSlot.value);
        if (sameTarget && incomingSlot.polarity !== existingSlot.polarity) {
          matches.push({ record: existing, reason: 'polarity-flip' });
          seenIds.add(existing.id);
          continue;
        }
        if (hasAntonymConflict(existingSlot.value, incomingSlot.value)) {
          matches.push({ record: existing, reason: 'antonym-conflict' });
          seenIds.add(existing.id);
          continue;
        }
      } else if (incomingSlot.value !== existingSlot.value) {
        matches.push({ record: existing, reason: 'slot-update' });
        seenIds.add(existing.id);
        continue;
      }
    }

    // 3. Direct antonym swap with overlapping context tokens
    if (hasAntonymConflict(existing.content, input.content)) {
      const tokA = new Set(tokenize(existing.content, { removeStopwords: true, stem: true }));
      const tokB = new Set(tokenize(input.content, { removeStopwords: true, stem: true }));
      let shared = 0;
      for (const t of tokA) {
        if (tokB.has(t)) shared++;
      }
      if (shared >= 1) {
        matches.push({ record: existing, reason: 'antonym-conflict' });
        seenIds.add(existing.id);
      }
    }
  }

  return matches;
}
