import type { BlockedDomainRule, DomainRuleKind } from '../contracts.js';
import { generateId } from '../utils/id.js';

export interface DomainBlockCheck {
  blocked: boolean;
  reason?: string;
  rule?: string;
}

interface CompiledRule {
  rule: BlockedDomainRule;
  test: (normalizedUrl: string, host: string, pathAndQuery: string) => boolean;
}

const BUILTIN_RULES: Array<Omit<BlockedDomainRule, 'createdAt'>> = [
  {
    id: 'builtin_scheme_chrome',
    pattern: 'chrome://*',
    kind: 'glob',
    reason: 'Browser internal chrome:// pages are excluded from automatic capture',
    builtin: true,
  },
  {
    id: 'builtin_scheme_extension',
    pattern: 'chrome-extension://*',
    kind: 'glob',
    reason: 'Extension internal pages are excluded from automatic capture',
    builtin: true,
  },
  {
    id: 'builtin_scheme_edge',
    pattern: 'edge://*',
    kind: 'glob',
    reason: 'Browser internal edge:// pages are excluded from automatic capture',
    builtin: true,
  },
  {
    id: 'builtin_scheme_about',
    pattern: 'about:*',
    kind: 'glob',
    reason: 'Browser internal about: pages are excluded from automatic capture',
    builtin: true,
  },
  {
    id: 'builtin_scheme_devtools',
    pattern: 'devtools://*',
    kind: 'glob',
    reason: 'DevTools pages are excluded from automatic capture',
    builtin: true,
  },
  {
    id: 'builtin_localhost_secrets',
    pattern: '^https?://(localhost|127\\.0\\.0\\.1|\\[::1\\])(:\\d+)?/(admin|auth|oauth|secret|secrets|token|vault|env)(/|$|\\?)',
    kind: 'regex',
    reason: 'Localhost admin, auth, and secret endpoints are blocked',
    builtin: true,
  },
  {
    id: 'builtin_query_secrets',
    pattern: '[?&](access_token|id_token|refresh_token|api_key|apikey|client_secret|password|passwd|secret|session_id|token)=',
    kind: 'regex',
    reason: 'URL contains sensitive credential or token query parameters',
    builtin: true,
  },
  {
    id: 'builtin_password_managers',
    pattern: '(^|\\.)(1password\\.com|bitwarden\\.com|lastpass\\.com|dashlane\\.com|keepersecurity\\.com|proton\\.me|pass\\.proton\\.me)$',
    kind: 'regex',
    reason: 'Password manager and encrypted vault domains are blocked',
    builtin: true,
  },
  {
    id: 'builtin_auth_providers',
    pattern: '(^|\\.)(accounts\\.google\\.com|login\\.microsoftonline\\.com|login\\.live\\.com|appleid\\.apple\\.com|okta\\.com|auth0\\.com|duosecurity\\.com)$',
    kind: 'regex',
    reason: 'Authentication and SSO identity providers are blocked',
    builtin: true,
  },
  {
    id: 'builtin_banking_major',
    pattern: '(^|\\.)(chase\\.com|bankofamerica\\.com|wellsfargo\\.com|citibank\\.com|citi\\.com|capitalone\\.com|paypal\\.com|venmo\\.com|stripe\\.com|coinbase\\.com|schwab\\.com|fidelity\\.com|vanguard\\.com|americanexpress\\.com)$',
    kind: 'regex',
    reason: 'Banking and financial institution domains are blocked',
    builtin: true,
  },
  {
    id: 'builtin_banking_host_keywords',
    pattern: '(^|\\.)([a-z0-9-]*chasebank[a-z0-9-]*|[a-z0-9-]*banking[a-z0-9-]*|onlinebanking|mybank)\\.',
    kind: 'regex',
    reason: 'Banking and financial portals are excluded from automatic capture',
    builtin: true,
  },
  {
    id: 'builtin_sensitive_financial_tax_paths',
    pattern: '^https?://[^/]+/(accounts/(checking|savings)|banking|billing/cards|tax-forms?|w-?2|payroll|credentials)([/?#]|$)',
    kind: 'regex',
    reason: 'Sensitive banking, payroll, tax, and credential URL paths are blocked',
    builtin: true,
  },
  {
    id: 'builtin_generic_auth_bank_subdomains',
    pattern: '^(auth|signin|login|sso|oauth|bank|banking|onlinebanking|vault|checkout)\\.',
    kind: 'regex',
    reason: 'Sensitive authentication, banking, or checkout subdomain is blocked',
    builtin: true,
  },
];

function globToRegExp(glob: string): RegExp {
  const escaped = glob
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*/g, '.*')
    .replace(/\?/g, '.');
  return new RegExp(`^${escaped}$`, 'i');
}

function compileRule(rule: BlockedDomainRule): CompiledRule {
  if (rule.kind === 'exact') {
    const target = rule.pattern.toLowerCase();
    return {
      rule,
      test: (fullUrl, host) => fullUrl === target || host === target,
    };
  }
  if (rule.kind === 'glob') {
    const regex = globToRegExp(rule.pattern);
    return {
      rule,
      test: (fullUrl, host) => regex.test(fullUrl) || regex.test(host),
    };
  }
  const regex = new RegExp(rule.pattern, 'i');
  return {
    rule,
    test: (fullUrl, host, pathAndQuery) =>
      regex.test(fullUrl) || regex.test(host) || regex.test(pathAndQuery),
  };
}

function parseUrlParts(rawUrl: string): { normalizedUrl: string; host: string; pathAndQuery: string } {
  const trimmed = rawUrl.trim();
  try {
    const u = new URL(trimmed);
    return {
      normalizedUrl: u.href.toLowerCase(),
      host: u.hostname.toLowerCase(),
      pathAndQuery: `${u.pathname}${u.search}${u.hash}`.toLowerCase(),
    };
  } catch {
    const lower = trimmed.toLowerCase();
    const withoutScheme = lower.replace(/^[a-z][a-z0-9+.-]*:\/\//, '');
    const slashIdx = withoutScheme.indexOf('/');
    const hostPort = slashIdx === -1 ? withoutScheme : withoutScheme.slice(0, slashIdx);
    const host = hostPort.split(':')[0] ?? hostPort;
    const pathAndQuery = slashIdx === -1 ? '' : withoutScheme.slice(slashIdx);
    return {
      normalizedUrl: lower,
      host,
      pathAndQuery,
    };
  }
}

/**
 * Sensitive-domain and secret-URL filter preventing automatic capture from
 * banking, auth/SSO, password managers, browser internals, and localhost secrets.
 */
export class SensitiveDomainFilter {
  private compiled: CompiledRule[] = [];

  constructor(options: { includeBuiltins?: boolean; initialRules?: BlockedDomainRule[] } = {}) {
    const includeBuiltins = options.includeBuiltins ?? true;
    if (includeBuiltins) {
      const now = 0;
      for (const b of BUILTIN_RULES) {
        this.compiled.push(compileRule({ ...b, createdAt: now }));
      }
    }
    if (options.initialRules) {
      for (const r of options.initialRules) {
        this.addRuleObject(r);
      }
    }
  }

  addRuleObject(rule: BlockedDomainRule): BlockedDomainRule {
    const existingIdx = this.compiled.findIndex((c) => c.rule.id === rule.id);
    const entry = compileRule(rule);
    if (existingIdx >= 0) {
      this.compiled[existingIdx] = entry;
    } else {
      this.compiled.push(entry);
    }
    return { ...rule };
  }

  addRule(pattern: string, kind: DomainRuleKind = 'glob', reason = 'User-configured privacy rule'): BlockedDomainRule {
    // Validate regex eagerly
    if (kind === 'regex') {
      new RegExp(pattern, 'i');
    }
    const rule: BlockedDomainRule = {
      id: generateId('blk'),
      pattern,
      kind,
      reason,
      builtin: false,
      createdAt: Date.now(),
    };
    return this.addRuleObject(rule);
  }

  removeRule(id: string): boolean {
    const idx = this.compiled.findIndex((c) => c.rule.id === id);
    if (idx === -1) return false;
    if (this.compiled[idx]?.rule.builtin) {
      return false;
    }
    this.compiled.splice(idx, 1);
    return true;
  }

  listRules(): BlockedDomainRule[] {
    return this.compiled.map((c) => ({ ...c.rule }));
  }

  listUserRules(): BlockedDomainRule[] {
    return this.compiled.filter((c) => !c.rule.builtin).map((c) => ({ ...c.rule }));
  }

  check(url?: string): DomainBlockCheck {
    if (!url || !url.trim()) {
      return { blocked: false };
    }
    const { normalizedUrl, host, pathAndQuery } = parseUrlParts(url);
    for (const entry of this.compiled) {
      if (entry.test(normalizedUrl, host, pathAndQuery)) {
        return {
          blocked: true,
          reason: entry.rule.reason,
          rule: entry.rule.pattern,
        };
      }
    }
    return { blocked: false };
  }
}

interface ContentPolicyRule {
  pattern: string;
  regex: RegExp;
  reason: string;
}

const SENSITIVE_CONTENT_RULES: ContentPolicyRule[] = [
  {
    pattern: 'content:ssn',
    regex: /\b\d{3}-\d{2}-\d{4}\b/,
    reason: 'Content appears to contain a US Social Security Number (SSN)',
  },
  {
    pattern: 'content:banking_account',
    regex: /\b(?:routing\s*(?:number|num|no|#)?\s*[:#=-]?\s*\d{9}|(?:checking|savings)\s+account\s+balance|iban\s*[:#]?\s*[a-z]{2}\d{2}[a-z0-9]{11,})\b/i,
    reason: 'Content appears to contain sensitive bank account or routing details',
  },
  {
    pattern: 'content:credential_token',
    regex: /(?:-----BEGIN (?:[A-Z0-9 ]* )?PRIVATE KEY-----|\b(?:secret_live_|sk_live_|rk_live_|pk_live_|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})|\b(?:access_token|refresh_token|id_token|client_secret|api_key)\s*[:=]\s*['"]?[A-Za-z0-9._-]{8,})/i,
    reason: 'Content appears to contain an API token, secret key, or OAuth credential',
  },
  {
    pattern: 'content:password_secret',
    regex: /\b(?:(?:recovery|master|backup)\s+password(?:\s+hint)?|password\s*[:=]\s*\S{4,})\b/i,
    reason: 'Content appears to contain sensitive password or recovery information',
  },
];

export function checkSensitiveContent(content?: string): DomainBlockCheck {
  if (!content || !content.trim()) {
    return { blocked: false };
  }
  for (const rule of SENSITIVE_CONTENT_RULES) {
    if (rule.regex.test(content)) {
      return {
        blocked: true,
        reason: rule.reason,
        rule: rule.pattern,
      };
    }
  }
  return { blocked: false };
}
