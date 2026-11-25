# Contributing to `@yours/memory`

Thank you for contributing to `@yours/memory`! Please review [docs/ECOSYSTEM_CONTRACTS.md](./docs/ECOSYSTEM_CONTRACTS.md) before making changes so shared contracts across `yours-research` packages stay aligned.

## Prerequisites

- **Node.js** `>= 20.0.0`
- **npm** `>= 10.0.0`

## Development Workflow

```bash
# Install dev dependencies
npm install

# Run strict TypeScript typechecking
npm run typecheck

# Compile ESM + .d.ts artifacts into dist/
npm run build

# Run the full Vitest suite
npm test
```

## Standards

1. **Contract Fidelity:** Never break the `MemoryProvider`, `MemoryRecord`, `MemoryQuery`, `MemoryHit`, or `MemoryTier` signatures defined in `docs/ECOSYSTEM_CONTRACTS.md`.
2. **Zero Network By Default:** Do not introduce `fetch`, `XMLHttpRequest`, or external telemetry/CDN calls into `@yours/memory`.
3. **Strict TypeScript & Clean Diffs:** Maintain `"strict": true` with zero `@ts-ignore` or unsafe casts.
4. **Tests Required:** Every new ranking rule, storage migration, privacy filter, or lifecycle heuristic must include deterministic Vitest coverage under `tests/`.
5. **License:** All contributions are licensed under **Apache-2.0**.
