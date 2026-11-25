import { EncryptionError } from '../errors.js';

const MAGIC = new Uint8Array([0x59, 0x4d, 0x45, 0x4d]); // 'YMEM'
const FORMAT_VERSION = 1;
const SALT_BYTES = 16;
const IV_BYTES = 12;
const HEADER_BYTES = 4 + 1 + 4 + SALT_BYTES + IV_BYTES; // 37 bytes
const DEFAULT_PBKDF2_ITERATIONS = 210_000;

function getSubtleCrypto(): SubtleCrypto {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new EncryptionError('WebCrypto SubtleCrypto is unavailable in this environment');
  }
  return subtle;
}

async function deriveAesKey(
  passphrase: string,
  salt: Uint8Array,
  iterations: number,
  usage: KeyUsage[]
): Promise<CryptoKey> {
  if (!passphrase) {
    throw new EncryptionError('Passphrase must not be empty');
  }
  const subtle = getSubtleCrypto();
  const encoder = new TextEncoder();
  const keyMaterial = await subtle.importKey('raw', encoder.encode(passphrase), 'PBKDF2', false, [
    'deriveKey',
  ]);
  const saltBuf = new Uint8Array(salt).buffer;
  return subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: saltBuf,
      iterations,
      hash: 'SHA-256',
    },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    usage
  );
}

/**
 * Encrypt a plaintext payload using PBKDF2-SHA256 + AES-256-GCM.
 * Format: [4B 'YMEM'][1B version=1][4B u32le iterations][16B salt][12B iv][ciphertext+tag]
 */
export async function encryptPayload(
  plaintext: Uint8Array,
  passphrase: string,
  iterations = DEFAULT_PBKDF2_ITERATIONS
): Promise<Uint8Array> {
  const subtle = getSubtleCrypto();
  const salt = globalThis.crypto.getRandomValues(new Uint8Array(SALT_BYTES));
  const iv = globalThis.crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const key = await deriveAesKey(passphrase, salt, iterations, ['encrypt']);

  const plainCopy = new Uint8Array(plaintext);
  const cipherBuffer = await subtle.encrypt(
    { name: 'AES-GCM', iv: new Uint8Array(iv).buffer },
    key,
    plainCopy.buffer
  );
  const cipherBytes = new Uint8Array(cipherBuffer);

  const out = new Uint8Array(HEADER_BYTES + cipherBytes.byteLength);
  const view = new DataView(out.buffer);
  out.set(MAGIC, 0);
  view.setUint8(4, FORMAT_VERSION);
  view.setUint32(5, iterations, true);
  out.set(salt, 9);
  out.set(iv, 9 + SALT_BYTES);
  out.set(cipherBytes, HEADER_BYTES);
  return out;
}

/** Decrypt a payload previously created by `encryptPayload`. */
export async function decryptPayload(blob: Uint8Array, passphrase: string): Promise<Uint8Array> {
  if (blob.byteLength <= HEADER_BYTES) {
    throw new EncryptionError('Encrypted backup is truncated or malformed');
  }
  for (let i = 0; i < MAGIC.length; i++) {
    if (blob[i] !== MAGIC[i]) {
      throw new EncryptionError('Invalid encrypted backup header');
    }
  }
  const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const version = view.getUint8(4);
  if (version !== FORMAT_VERSION) {
    throw new EncryptionError(`Unsupported encrypted backup version: ${version}`);
  }
  const iterations = view.getUint32(5, true);
  if (iterations < 1000) {
    throw new EncryptionError('Invalid PBKDF2 iteration count in backup header');
  }
  const salt = blob.subarray(9, 9 + SALT_BYTES);
  const iv = blob.subarray(9 + SALT_BYTES, HEADER_BYTES);
  const ciphertext = blob.subarray(HEADER_BYTES);

  const subtle = getSubtleCrypto();
  const key = await deriveAesKey(passphrase, salt, iterations, ['decrypt']);

  try {
    const cipherCopy = new Uint8Array(ciphertext);
    const plainBuffer = await subtle.decrypt(
      { name: 'AES-GCM', iv: new Uint8Array(iv).buffer },
      key,
      cipherCopy.buffer
    );
    return new Uint8Array(plainBuffer);
  } catch (err) {
    throw new EncryptionError('Decryption failed: invalid passphrase or corrupted backup', {
      cause: err,
    });
  }
}
