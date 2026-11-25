export class MemoryError extends Error {
  readonly code: string;

  constructor(code: string, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'MemoryError';
    this.code = code;
  }
}

export class SensitiveDomainBlockedError extends MemoryError {
  readonly url: string;
  readonly rule: string;

  constructor(url: string, rule: string, reason: string) {
    super('SENSITIVE_DOMAIN_BLOCKED', `Blocked memory capture from "${url}" (${reason}; rule: ${rule})`);
    this.name = 'SensitiveDomainBlockedError';
    this.url = url;
    this.rule = rule;
  }
}

export class TaintViolationError extends MemoryError {
  readonly targetRecordId: string;

  constructor(targetRecordId: string, message?: string) {
    super(
      'TAINT_VIOLATION',
      message ??
        `Untainted profile record "${targetRecordId}" cannot be silently superseded by tainted content without explicit confirmation.`
    );
    this.name = 'TaintViolationError';
    this.targetRecordId = targetRecordId;
  }
}

export class EncryptionError extends MemoryError {
  constructor(message: string, options?: ErrorOptions) {
    super('ENCRYPTION_ERROR', message, options);
    this.name = 'EncryptionError';
  }
}

export class WorkerCancelledError extends MemoryError {
  constructor(message = 'Worker operation was cancelled') {
    super('WORKER_CANCELLED', message);
    this.name = 'WorkerCancelledError';
  }
}
