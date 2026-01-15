"""
Token Processing for KV-Bench.

This module provides token processing utilities for KV cache management,
including tokenization simulation, chunk management, and LMCache-compatible
key generation.

The chunking strategy follows LMCache conventions where tokens are divided
into fixed-size chunks, and each chunk is independently cached using a
content-based hash key.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class TokenChunk:
    """A chunk of tokens with metadata.

    Attributes:
        tokens: List of token IDs in this chunk.
        chunk_index: Index of this chunk in the sequence.
        chunk_hash: SHA-256 hash of the chunk content.
        start_position: Starting position in the original sequence.
    """

    tokens: list[int]
    chunk_index: int
    chunk_hash: str
    start_position: int

    @property
    def size(self) -> int:
        """Return the number of tokens in this chunk."""
        return len(self.tokens)

    @property
    def is_full(self) -> bool:
        """Check if this is a full-size chunk (not a partial final chunk)."""
        # This is set by the TokenProcessor based on chunk_size
        return True  # Will be overridden by processor


class TokenProcessor:
    """Processor for tokenization simulation and chunk management.

    This class provides utilities for:
    - Simulating tokenization (without a real tokenizer)
    - Chunking token sequences for prefix caching
    - Computing content-based hashes for cache keys

    The chunking follows LMCache conventions where:
    - Tokens are divided into fixed-size chunks
    - Each chunk is independently hashed
    - Partial final chunks are handled appropriately

    Attributes:
        chunk_size: Number of tokens per chunk (default: 256).
    """

    def __init__(self, chunk_size: int = 256):
        """Initialize the token processor.

        Args:
            chunk_size: Number of tokens per chunk (default: 256).
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self.chunk_size = chunk_size

    def simulate_tokenize(self, text: str, chars_per_token: float = 4.0) -> list[int]:
        """Simulate tokenization by generating pseudo-random token IDs.

        This method simulates tokenization without using a real tokenizer.
        It generates deterministic token IDs based on the input text content.

        The number of tokens is estimated based on the character-to-token ratio,
        which is approximately 4 characters per token for English text.

        Args:
            text: Input text to tokenize.
            chars_per_token: Average characters per token (default: 4.0).

        Returns:
            List of simulated token IDs.
        """
        if not text:
            return []

        # Estimate number of tokens
        num_tokens = max(1, int(len(text) / chars_per_token))

        # Generate deterministic token IDs based on text content
        tokens = []
        text_bytes = text.encode("utf-8")

        for i in range(num_tokens):
            # Create a hash combining the text and position
            hash_input = text_bytes + i.to_bytes(4, "little")
            hash_digest = hashlib.sha256(hash_input).digest()
            # Use first 4 bytes as token ID (mod vocab size, using 128K as default)
            token_id = int.from_bytes(hash_digest[:4], "little") % 128256
            tokens.append(token_id)

        return tokens

    def chunk_tokens(self, tokens: list[int]) -> list[list[int]]:
        """Divide tokens into fixed-size chunks.

        Args:
            tokens: List of token IDs.

        Returns:
            List of token chunks (each chunk is a list of token IDs).
        """
        if not tokens:
            return []

        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i : i + self.chunk_size]
            chunks.append(chunk)

        return chunks

    def chunk_tokens_with_metadata(self, tokens: list[int]) -> list[TokenChunk]:
        """Divide tokens into chunks with full metadata.

        Args:
            tokens: List of token IDs.

        Returns:
            List of TokenChunk objects with hashes and positions.
        """
        if not tokens:
            return []

        chunks = []
        for i, start in enumerate(range(0, len(tokens), self.chunk_size)):
            chunk_tokens = tokens[start : start + self.chunk_size]
            chunk_hash = self.compute_chunk_hash(chunk_tokens)

            chunk = TokenChunk(
                tokens=chunk_tokens,
                chunk_index=i,
                chunk_hash=chunk_hash,
                start_position=start,
            )
            chunks.append(chunk)

        return chunks

    def compute_chunk_hash(self, tokens: list[int]) -> str:
        """Compute SHA-256 hash of a token chunk.

        The hash is computed from the byte representation of the token IDs,
        providing a content-addressable identifier for the chunk.

        Args:
            tokens: List of token IDs.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        if not tokens:
            return hashlib.sha256(b"").hexdigest()

        # Convert tokens to bytes (4 bytes per token, little-endian)
        token_bytes = b"".join(t.to_bytes(4, "little") for t in tokens)
        return hashlib.sha256(token_bytes).hexdigest()

    def find_prefix_match(
        self, tokens: list[int], cached_hashes: set[str]
    ) -> tuple[int, list[str]]:
        """Find how many prefix chunks match the cache.

        This method checks each chunk from the beginning to find the
        longest prefix that exists in the cache.

        Args:
            tokens: List of token IDs.
            cached_hashes: Set of cached chunk hashes.

        Returns:
            Tuple of (number of matching tokens, list of matching hashes).
        """
        if not tokens or not cached_hashes:
            return 0, []

        chunks = self.chunk_tokens_with_metadata(tokens)
        matched_tokens = 0
        matched_hashes = []

        for chunk in chunks:
            if chunk.chunk_hash in cached_hashes:
                matched_tokens += chunk.size
                matched_hashes.append(chunk.chunk_hash)
            else:
                # Stop at first miss (prefix caching)
                break

        return matched_tokens, matched_hashes


@dataclass
class CacheKeyConfig:
    """Configuration for cache key generation.

    Attributes:
        format_version: Key format version string.
        model_name: Name of the model.
        world_size: Total number of workers (tensor parallelism).
        layer_start: Starting layer index for this shard.
        layer_end: Ending layer index for this shard.
    """

    format_version: str = "v1"
    model_name: str = "llama"
    world_size: int = 1
    layer_start: int = 0
    layer_end: int = 0


class CacheKeyGenerator:
    """Generator for LMCache-compatible cache keys.

    LMCache uses a specific key format for distributed KV cache storage:
    format@model@world_size@worker_id@layer_start-layer_end@hash@suffix

    This allows for:
    - Sharding KV cache across multiple workers
    - Layer-wise storage for large models
    - Content-based addressing via chunk hashes

    Attributes:
        config: Configuration for key generation.
    """

    # Key format components
    DEFAULT_FORMAT = "lmcache"
    DEFAULT_SUFFIX = "kv"

    def __init__(
        self,
        model_name: str = "llama",
        world_size: int = 1,
        format_version: str = "v1",
    ):
        """Initialize the cache key generator.

        Args:
            model_name: Name of the model (default: "llama").
            world_size: Total number of workers (default: 1).
            format_version: Key format version (default: "v1").
        """
        self.config = CacheKeyConfig(
            format_version=format_version,
            model_name=model_name,
            world_size=world_size,
        )

    def make_key(
        self,
        chunk_hash: str,
        worker_id: int = 0,
        layer_start: int = 0,
        layer_end: int | None = None,
        suffix: str = "kv",
    ) -> str:
        """Generate a cache key for a chunk.

        Key format: format@model@world_size@worker_id@layer_start-layer_end@hash@suffix

        Args:
            chunk_hash: SHA-256 hash of the token chunk.
            worker_id: Worker ID for this shard (default: 0).
            layer_start: Starting layer index (default: 0).
            layer_end: Ending layer index (default: None, meaning all layers).
            suffix: Key suffix (default: "kv").

        Returns:
            LMCache-compatible cache key string.
        """
        if layer_end is None:
            layer_end = layer_start

        components = [
            self.DEFAULT_FORMAT,
            self.config.model_name,
            str(self.config.world_size),
            str(worker_id),
            f"{layer_start}-{layer_end}",
            chunk_hash,
            suffix,
        ]
        return "@".join(components)

    def make_keys_for_all_workers(
        self,
        chunk_hash: str,
        layer_start: int = 0,
        layer_end: int | None = None,
        suffix: str = "kv",
    ) -> list[str]:
        """Generate cache keys for all workers.

        This is useful when storing or retrieving KV cache that needs
        to be sharded across multiple workers.

        Args:
            chunk_hash: SHA-256 hash of the token chunk.
            layer_start: Starting layer index (default: 0).
            layer_end: Ending layer index (default: None).
            suffix: Key suffix (default: "kv").

        Returns:
            List of cache keys, one per worker.
        """
        return [
            self.make_key(chunk_hash, worker_id, layer_start, layer_end, suffix)
            for worker_id in range(self.config.world_size)
        ]

    def make_keys_for_layers(
        self,
        chunk_hash: str,
        num_layers: int,
        layers_per_shard: int = 1,
        worker_id: int = 0,
        suffix: str = "kv",
    ) -> list[str]:
        """Generate cache keys for layer-sharded storage.

        This is useful when storing KV cache with layer-wise sharding,
        where different layers are stored in different keys.

        Args:
            chunk_hash: SHA-256 hash of the token chunk.
            num_layers: Total number of model layers.
            layers_per_shard: Number of layers per shard (default: 1).
            worker_id: Worker ID (default: 0).
            suffix: Key suffix (default: "kv").

        Returns:
            List of cache keys, one per layer shard.
        """
        keys = []
        for layer_start in range(0, num_layers, layers_per_shard):
            layer_end = min(layer_start + layers_per_shard - 1, num_layers - 1)
            key = self.make_key(chunk_hash, worker_id, layer_start, layer_end, suffix)
            keys.append(key)
        return keys

    def parse_key(self, key: str) -> dict:
        """Parse a cache key into its components.

        Args:
            key: LMCache-compatible cache key string.

        Returns:
            Dictionary with parsed key components.

        Raises:
            ValueError: If the key format is invalid.
        """
        parts = key.split("@")
        if len(parts) != 7:
            raise ValueError(f"Invalid key format: expected 7 components, got {len(parts)}")

        format_name, model, world_size, worker_id, layer_range, chunk_hash, suffix = parts

        # Parse layer range
        layer_parts = layer_range.split("-")
        if len(layer_parts) != 2:
            raise ValueError(f"Invalid layer range format: {layer_range}")

        return {
            "format": format_name,
            "model": model,
            "world_size": int(world_size),
            "worker_id": int(worker_id),
            "layer_start": int(layer_parts[0]),
            "layer_end": int(layer_parts[1]),
            "chunk_hash": chunk_hash,
            "suffix": suffix,
        }

    def make_metadata_key(self, chunk_hash: str, worker_id: int = 0) -> str:
        """Generate a metadata key for a chunk.

        Metadata keys store information about chunks like token count,
        creation time, etc.

        Args:
            chunk_hash: SHA-256 hash of the token chunk.
            worker_id: Worker ID (default: 0).

        Returns:
            Metadata cache key string.
        """
        return self.make_key(chunk_hash, worker_id, suffix="meta")
