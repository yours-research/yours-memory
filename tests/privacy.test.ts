import { describe, expect, it } from 'vitest';
import {
  EncryptionError,
  HybridMemoryEngine,
  InMemoryStorageBackend,
  SensitiveDomainBlockedError,
  TaintViolationError,
} from '../src/index.js';

describe('Privacy: sensitive-domain blocklist, taint guard, AES-GCM export/import & wipeAll', () => {
  it('blocks built-in sensitive URLs and supports custom glob/regex rules', async () => {
    const engine = new HybridMemoryEngine({ storage: new InMemoryStorageBackend() });
    await engine.init();

    const blockedUrls = [
      'chrome://settings/passwords',
      'chrome-extension://abcdef/options.html',
      'https://secure.chase.com/web/auth/dashboard',
      'https://my.1password.com/vaults',
      'https://accounts.google.com/v3/signin',
      'http://localhost:8080/admin/secrets',
      'https://example.org/oauth/callback?access_token=supersecret',
    ];

    for (const url of blockedUrls) {
      await expect(
        engine.remember({ tier: 'document', content: 'Page text', sourceUrl: url })
      ).rejects.toBeInstanceOf(SensitiveDomainBlockedError);
    }

    // Allowed public docs URL works
    const allowed = await engine.remember({
      tier: 'document',
      content: 'WebGPU compute shader guide',
      sourceUrl: 'https://gpuweb.github.io/gpuweb/',
      tainted: true,
    });
    expect(allowed.id).toBeTruthy();

    // Custom glob & regex rules
    const customRule = await engine.addBlockedDomain('*.internal.corp', 'glob', 'Corporate intranet');
    await expect(
      engine.remember({
        tier: 'document',
        content: 'HR doc',
        sourceUrl: 'https://wiki.internal.corp/handbook',
      })
    ).rejects.toBeInstanceOf(SensitiveDomainBlockedError);

    expect(await engine.removeBlockedDomain(customRule.id)).toBe(true);
    // Removing now allows capture
    const unblocked = await engine.remember({
      tier: 'document',
      content: 'Public wiki page',
      sourceUrl: 'https://wiki.internal.corp/public',
    });
    expect(unblocked.id).toBeTruthy();
  });

  it('prevents tainted content from silently superseding trusted profile facts', async () => {
    const engine = new HybridMemoryEngine({ storage: new InMemoryStorageBackend() });
    await engine.init();

    const trusted = await engine.remember({
      tier: 'profile',
      content: 'My preferred browser is Chrome',
      tainted: false,
    });

    await expect(
      engine.remember({
        tier: 'profile',
        content: 'My preferred browser is Firefox',
        tainted: true,
        sourceUrl: 'https://untrusted.example.com/post',
      })
    ).rejects.toBeInstanceOf(TaintViolationError);

    // Original trusted profile fact stays intact
    const active = await engine.inspect({ tier: 'profile' });
    expect(active).toHaveLength(1);
    expect(active[0]?.id).toBe(trusted.id);

    // Explicit user confirmation allows superseding
    const confirmed = await engine.remember(
      {
        tier: 'profile',
        content: 'My preferred browser is Firefox',
        tainted: true,
        sourceUrl: 'https://untrusted.example.com/post',
      },
      { confirmTaintedProfileMutation: true }
    );
    expect(confirmed.supersedesId).toBe(trusted.id);
  });

  it('roundtrips AES-256-GCM encrypted backup, rejects wrong passphrase, and wipes cleanly', async () => {
    const engine1 = new HybridMemoryEngine({
      storage: new InMemoryStorageBackend(),
      pbkdf2Iterations: 5_000, // Fast test iteration count; default is 210_000
    });
    await engine1.init();

    await engine1.remember({ tier: 'profile', content: 'I prefer local-first inference', tags: ['ai'] });
    await engine1.remember({ tier: 'working', content: 'Debugging HNSW serialization bug' });
    await engine1.addBlockedDomain('^https://secret\\.example\\.net', 'regex', 'Custom secret');

    const encrypted = await engine1.exportEncrypted('correct-horse-battery-staple');
    expect(encrypted).toBeInstanceOf(Uint8Array);
    expect(new TextDecoder().decode(encrypted)).not.toContain('local-first inference');

    // Wrong passphrase fails cleanly
    const engine2 = new HybridMemoryEngine({ storage: new InMemoryStorageBackend() });
    await engine2.init();
    await expect(engine2.importEncrypted(encrypted, 'wrong-passphrase')).rejects.toBeInstanceOf(
      EncryptionError
    );

    // Correct passphrase restores records, HNSW index, and custom blocklist rules
    const { imported } = await engine2.importEncrypted(encrypted, 'correct-horse-battery-staple');
    expect(imported).toBe(2);

    const hits = await engine2.recall({ query: 'local-first inference' });
    expect(hits[0]?.record.content).toContain('local-first inference');
    expect(engine2.isUrlBlocked('https://secret.example.net/dashboard').blocked).toBe(true);

    // One-call wipeAll clears records, index, and custom rules
    await engine2.wipeAll();
    expect(await engine2.inspect({ includeSuperseded: true })).toHaveLength(0);
    expect((await engine2.getStats()).vectorCount).toBe(0);
    expect(engine2.isUrlBlocked('https://secret.example.net/dashboard').blocked).toBe(false);
  });

  it('blocks banking subdomains, sensitive financial/tax paths, and PII/credential content', async () => {
    const engine = new HybridMemoryEngine({ storage: new InMemoryStorageBackend() });
    await engine.init();

    const blockedInputs = [
      {
        tier: 'document' as const,
        content: 'Checking account balance $14,820.55 routing number 021000021.',
        sourceUrl: 'https://online.chasebank.example.com/accounts/checking',
      },
      {
        tier: 'document' as const,
        content: 'Employee SSN 123-45-6789 filed for W-2.',
        sourceUrl: 'https://hr.example.org/tax-forms',
      },
      {
        tier: 'working' as const,
        content: 'access_token=secret_live_abc123456789 should never persist',
      },
      {
        tier: 'episodic' as const,
        content: 'User entered recovery password hint during sign-in.',
      },
    ];

    for (const input of blockedInputs) {
      await expect(engine.remember(input)).rejects.toBeInstanceOf(SensitiveDomainBlockedError);
    }

    const benign = await engine.remember({
      tier: 'profile',
      content: 'User prefers dark mode and monospace code editors.',
      sourceUrl: 'https://docs.example.org/preferences/theme',
    });
    expect(benign.id).toBeTruthy();
  });

  it('respects setClock so recent episodic memories outrank stale ones on equal relevance', async () => {
    const engine = new HybridMemoryEngine({ storage: new InMemoryStorageBackend() });
    await engine.init();

    const nowMs = Date.UTC(2026, 10, 26, 12, 0, 0);
    const dayMs = 86_400_000;

    engine.setClock(nowMs - 95 * dayMs);
    const oldMem = await engine.remember({
      tier: 'episodic',
      content: 'Sprint planning meeting discussed WebGPU kernel tuning and shader caching.',
      importance: 0.65,
    });

    engine.setClock(nowMs - 1 * dayMs);
    const recentMem = await engine.remember({
      tier: 'episodic',
      content: 'Sprint planning meeting discussed WebGPU kernel tuning and shader caching.',
      importance: 0.65,
    });

    engine.setClock(nowMs);
    const hits = await engine.recall({
      query: 'Sprint planning meeting WebGPU kernel tuning',
      limit: 5,
    });
    expect(hits[0]?.record.id).toBe(recentMem.id);
    expect(hits.some((h) => h.record.id === oldMem.id)).toBe(true);
  });
});
