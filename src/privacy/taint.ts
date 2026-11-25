import type { MemoryRecord, MemoryRecordInput } from '../contracts.js';
import { TaintViolationError } from '../errors.js';

/**
 * Enforces the IDPI taint boundary:
 * Untrusted (`tainted: true`) web/tool content cannot silently overwrite or supersede
 * a trusted (`tainted: false`) `profile` record unless explicitly confirmed by the user.
 */
export function assertTaintMutationAllowed(
  targetRecord: MemoryRecord,
  incomingTainted: boolean,
  confirmTaintedProfileMutation = false
): void {
  if (
    targetRecord.tier === 'profile' &&
    !targetRecord.tainted &&
    incomingTainted &&
    !confirmTaintedProfileMutation
  ) {
    throw new TaintViolationError(targetRecord.id);
  }
}

export function resolveTaintFlag(input: MemoryRecordInput): boolean {
  return Boolean(input.tainted);
}
