/** Generate a collision-resistant unique ID using WebCrypto when available. */
export function generateId(prefix = 'mem'): string {
  const gCrypto = globalThis.crypto;
  if (gCrypto && typeof gCrypto.randomUUID === 'function') {
    return `${prefix}_${gCrypto.randomUUID().replace(/-/g, '')}`;
  }
  const timePart = Date.now().toString(36);
  const randA = Math.floor(Math.random() * 0xffffffff)
    .toString(36)
    .padStart(7, '0');
  const randB = Math.floor(Math.random() * 0xffffffff)
    .toString(36)
    .padStart(7, '0');
  return `${prefix}_${timePart}${randA}${randB}`;
}
