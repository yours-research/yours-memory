import type { HarnessMessage, MemoryRecordInput } from '../contracts.js';
import { extractAssertionSlot } from './contradiction.js';

const DURABLE_PROFILE_CUES = [
  /\bmy\s+(?:name|favorite|preferred|default|primary|timezone|location|city|editor|ide|browser|theme|language|package manager)\b/i,
  /\bi\s+(?:always\s+)?(?:prefer|use|like|love|dislike|hate|avoid|live in|work at)\b/i,
  /\bremember\s+(?:that\s+)?/i,
];

export interface ConsolidationPlan {
  profileFacts: MemoryRecordInput[];
  episodicSummary: MemoryRecordInput | null;
}

/**
 * Extracts durable user profile facts (strictly from untainted user messages)
 * and builds an episodic session summary for `threadId`.
 */
export function planConsolidation(threadId: string, messages: HarnessMessage[]): ConsolidationPlan {
  const profileFacts: MemoryRecordInput[] = [];
  const seenProfileText = new Set<string>();
  const summaryLines: string[] = [];
  let hasTaintedContribution = false;

  for (const msg of messages) {
    const trimmed = msg.content.trim();
    if (!trimmed) continue;

    if (msg.tainted || msg.role === 'tool') {
      hasTaintedContribution = true;
      // Never promote tainted or tool messages directly into untainted profile facts.
      if (msg.role === 'tool') {
        const snippet = trimmed.length > 160 ? `${trimmed.slice(0, 157)}...` : trimmed;
        summaryLines.push(`[tool:${msg.toolName ?? 'unknown'}] ${snippet}`);
      } else {
        const snippet = trimmed.length > 180 ? `${trimmed.slice(0, 177)}...` : trimmed;
        summaryLines.push(`[${msg.role}:tainted] ${snippet}`);
      }
      continue;
    }

    if (msg.role === 'user') {
      const slot = extractAssertionSlot(trimmed);
      const hasCue = slot !== null || DURABLE_PROFILE_CUES.some((re) => re.test(trimmed));
      if (hasCue) {
        const cleanedFact = trimmed.replace(/^remember\s+(?:that\s+)?/i, '').trim();
        const key = cleanedFact.toLowerCase();
        if (cleanedFact && !seenProfileText.has(key)) {
          seenProfileText.add(key);
          profileFacts.push({
            tier: 'profile',
            content: cleanedFact,
            threadId,
            tags: ['consolidated', 'profile-fact'],
            importance: 0.8,
            tainted: false,
          });
        }
      }
      summaryLines.push(`[user] ${trimmed}`);
    } else if (msg.role === 'assistant') {
      const snippet = trimmed.length > 220 ? `${trimmed.slice(0, 217)}...` : trimmed;
      summaryLines.push(`[assistant] ${snippet}`);
    }
  }

  const episodicSummary: MemoryRecordInput | null =
    summaryLines.length > 0
      ? {
          tier: 'episodic',
          content: `Session ${threadId} summary:\n${summaryLines.slice(-12).join('\n')}`,
          threadId,
          tags: ['consolidated', 'session-summary'],
          importance: 0.65,
          tainted: hasTaintedContribution,
        }
      : null;

  return { profileFacts, episodicSummary };
}
