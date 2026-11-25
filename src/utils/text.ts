/** FTS5-compatible unicode61 + code-aware tokenizer and canonical text normalization. */

const DEFAULT_STOPWORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'for',
  'from',
  'has',
  'he',
  'in',
  'is',
  'it',
  'its',
  'of',
  'on',
  'or',
  'that',
  'the',
  'to',
  'was',
  'were',
  'will',
  'with',
]);

/** Canonical string normalization used for exact deduplication and comparison. */
export function canonicalizeText(text: string): string {
  return text
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[\u200B-\u200D\uFEFF]/g, '')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Lightweight suffix normalization so simple plurals/inflections align across BM25 queries
 * while preserving exact code identifiers.
 */
export function normalizeToken(token: string): string {
  const lower = token.toLowerCase();
  if (lower.length <= 3) return lower;
  if (lower.endsWith('ies') && lower.length > 4) {
    return `${lower.slice(0, -3)}y`;
  }
  if (lower.endsWith('ing') && lower.length > 5) {
    return lower.slice(0, -3);
  }
  if (lower.endsWith('ed') && lower.length > 4) {
    return lower.slice(0, -2);
  }
  if (lower.endsWith('s') && !lower.endsWith('ss') && lower.length > 3) {
    return lower.slice(0, -1);
  }
  return lower;
}

export interface TokenizeOptions {
  removeStopwords?: boolean;
  stem?: boolean;
  splitIdentifiers?: boolean;
}

/**
 * Tokenize text into normalized FTS5/BM25 terms.
 * Handles camelCase (`getElementById` -> `get`, `element`, `by`, `id`, `getelementbyid`)
 * and snake_case (`sync_access_handle` -> `sync`, `access`, `handle`).
 */
export function tokenize(text: string, options: TokenizeOptions = {}): string[] {
  const { removeStopwords = true, stem = false, splitIdentifiers = true } = options;
  const cleaned = text.normalize('NFKC').replace(/[\u200B-\u200D\uFEFF]/g, '');
  const rawMatches = cleaned.match(/[\p{L}\p{N}_]+/gu);
  if (!rawMatches) return [];

  const tokens: string[] = [];
  for (const raw of rawMatches) {
    const parts: string[] = [];
    if (splitIdentifiers) {
      const snakeParts = raw.split('_').filter(Boolean);
      for (const sp of snakeParts) {
        const camelSplit = sp
          .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
          .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
          .split(' ');
        for (const cp of camelSplit) {
          if (cp) parts.push(cp);
        }
      }
      if (parts.length > 1) {
        parts.push(raw.replace(/_/g, ''));
      }
    } else {
      parts.push(raw);
    }

    for (const part of parts) {
      const lower = part.toLowerCase();
      if (removeStopwords && DEFAULT_STOPWORDS.has(lower)) {
        continue;
      }
      tokens.push(stem ? normalizeToken(lower) : lower);
    }
  }
  return tokens;
}

/** Build an FTS5-safe MATCH query string from freeform user query text. */
export function buildFts5MatchQuery(query: string): string {
  const terms = tokenize(query, { removeStopwords: true, stem: false, splitIdentifiers: true });
  if (terms.length === 0) {
    const fallback = tokenize(query, { removeStopwords: false, stem: false, splitIdentifiers: false });
    if (fallback.length === 0) return '';
    return fallback.map((t) => `"${t.replace(/"/g, '""')}"`).join(' OR ');
  }
  const unique = Array.from(new Set(terms));
  return unique.map((t) => `"${t.replace(/"/g, '""')}"*`).join(' OR ');
}
