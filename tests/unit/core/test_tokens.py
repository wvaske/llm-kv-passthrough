"""Unit tests for kvbench.core.tokens module."""

from __future__ import annotations

import hashlib

import pytest

from kvbench.core.tokens import (
    CacheKeyGenerator,
    TokenChunk,
    TokenProcessor,
)


class TestTokenChunk:
    """Tests for TokenChunk dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic TokenChunk creation."""
        chunk = TokenChunk(
            tokens=[1, 2, 3, 4, 5],
            chunk_index=0,
            chunk_hash="abc123",
            start_position=0,
        )
        assert chunk.tokens == [1, 2, 3, 4, 5]
        assert chunk.chunk_index == 0
        assert chunk.chunk_hash == "abc123"
        assert chunk.start_position == 0

    def test_size_property(self) -> None:
        """Test size property."""
        chunk = TokenChunk(
            tokens=[1, 2, 3],
            chunk_index=0,
            chunk_hash="hash",
            start_position=0,
        )
        assert chunk.size == 3

    def test_empty_chunk_size(self) -> None:
        """Test empty chunk size."""
        chunk = TokenChunk(
            tokens=[],
            chunk_index=0,
            chunk_hash="hash",
            start_position=0,
        )
        assert chunk.size == 0


class TestTokenProcessor:
    """Tests for TokenProcessor class."""

    def test_default_chunk_size(self) -> None:
        """Test default chunk size is 256."""
        processor = TokenProcessor()
        assert processor.chunk_size == 256

    def test_custom_chunk_size(self) -> None:
        """Test custom chunk size."""
        processor = TokenProcessor(chunk_size=128)
        assert processor.chunk_size == 128

    def test_invalid_chunk_size(self) -> None:
        """Test that chunk_size must be at least 1."""
        with pytest.raises(ValueError, match="chunk_size must be at least 1"):
            TokenProcessor(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be at least 1"):
            TokenProcessor(chunk_size=-1)


class TestSimulateTokenize:
    """Tests for TokenProcessor.simulate_tokenize method."""

    def test_empty_text(self) -> None:
        """Test tokenization of empty string."""
        processor = TokenProcessor()
        tokens = processor.simulate_tokenize("")
        assert tokens == []

    def test_produces_tokens(self) -> None:
        """Test that tokenization produces tokens."""
        processor = TokenProcessor()
        tokens = processor.simulate_tokenize("Hello, world!")
        assert len(tokens) > 0

    def test_deterministic(self) -> None:
        """Test that tokenization is deterministic."""
        processor = TokenProcessor()
        text = "Hello, world! This is a test."
        tokens1 = processor.simulate_tokenize(text)
        tokens2 = processor.simulate_tokenize(text)
        assert tokens1 == tokens2

    def test_different_texts_different_tokens(self) -> None:
        """Test that different texts produce different tokens."""
        processor = TokenProcessor()
        tokens1 = processor.simulate_tokenize("Hello")
        tokens2 = processor.simulate_tokenize("Goodbye")
        assert tokens1 != tokens2

    def test_token_count_scaling(self) -> None:
        """Test that token count scales with text length."""
        processor = TokenProcessor()
        short = processor.simulate_tokenize("Hi")
        medium = processor.simulate_tokenize("Hello, world! This is a test.")
        long = processor.simulate_tokenize("A" * 1000)
        assert len(short) < len(medium) < len(long)

    def test_chars_per_token_ratio(self) -> None:
        """Test that chars_per_token affects token count."""
        processor = TokenProcessor()
        text = "A" * 100  # 100 characters
        tokens_4 = processor.simulate_tokenize(text, chars_per_token=4.0)
        tokens_2 = processor.simulate_tokenize(text, chars_per_token=2.0)
        # 2 chars/token should produce ~2x as many tokens
        assert len(tokens_2) > len(tokens_4)

    def test_token_ids_in_vocab_range(self) -> None:
        """Test that token IDs are within vocabulary range."""
        processor = TokenProcessor()
        tokens = processor.simulate_tokenize("Hello, world! This is a test message.")
        for token in tokens:
            assert 0 <= token < 128256  # Default vocab size


class TestChunkTokens:
    """Tests for TokenProcessor.chunk_tokens method."""

    def test_empty_tokens(self) -> None:
        """Test chunking empty token list."""
        processor = TokenProcessor(chunk_size=10)
        chunks = processor.chunk_tokens([])
        assert chunks == []

    def test_single_chunk(self) -> None:
        """Test tokens that fit in single chunk."""
        processor = TokenProcessor(chunk_size=10)
        tokens = [1, 2, 3, 4, 5]
        chunks = processor.chunk_tokens(tokens)
        assert len(chunks) == 1
        assert chunks[0] == [1, 2, 3, 4, 5]

    def test_exact_chunks(self) -> None:
        """Test tokens that divide evenly into chunks."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(15))  # 0-14, 15 tokens
        chunks = processor.chunk_tokens(tokens)
        assert len(chunks) == 3
        assert chunks[0] == [0, 1, 2, 3, 4]
        assert chunks[1] == [5, 6, 7, 8, 9]
        assert chunks[2] == [10, 11, 12, 13, 14]

    def test_partial_final_chunk(self) -> None:
        """Test tokens with partial final chunk."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(12))  # 12 tokens
        chunks = processor.chunk_tokens(tokens)
        assert len(chunks) == 3
        assert chunks[0] == [0, 1, 2, 3, 4]
        assert chunks[1] == [5, 6, 7, 8, 9]
        assert chunks[2] == [10, 11]  # Partial chunk


class TestChunkTokensWithMetadata:
    """Tests for TokenProcessor.chunk_tokens_with_metadata method."""

    def test_empty_tokens(self) -> None:
        """Test chunking empty token list with metadata."""
        processor = TokenProcessor(chunk_size=10)
        chunks = processor.chunk_tokens_with_metadata([])
        assert chunks == []

    def test_returns_token_chunks(self) -> None:
        """Test that method returns TokenChunk objects."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(10))
        chunks = processor.chunk_tokens_with_metadata(tokens)
        assert len(chunks) == 2
        assert all(isinstance(c, TokenChunk) for c in chunks)

    def test_chunk_indices(self) -> None:
        """Test that chunk indices are correct."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(15))
        chunks = processor.chunk_tokens_with_metadata(tokens)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_start_positions(self) -> None:
        """Test that start positions are correct."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(15))
        chunks = processor.chunk_tokens_with_metadata(tokens)
        assert chunks[0].start_position == 0
        assert chunks[1].start_position == 5
        assert chunks[2].start_position == 10

    def test_hashes_generated(self) -> None:
        """Test that hashes are generated for each chunk."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(10))
        chunks = processor.chunk_tokens_with_metadata(tokens)
        for chunk in chunks:
            assert len(chunk.chunk_hash) == 64  # SHA-256 hex

    def test_different_chunks_different_hashes(self) -> None:
        """Test that different chunks have different hashes."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(10))
        chunks = processor.chunk_tokens_with_metadata(tokens)
        assert chunks[0].chunk_hash != chunks[1].chunk_hash


class TestComputeChunkHash:
    """Tests for TokenProcessor.compute_chunk_hash method."""

    def test_empty_tokens(self) -> None:
        """Test hash of empty token list."""
        processor = TokenProcessor()
        hash_val = processor.compute_chunk_hash([])
        # SHA-256 of empty bytes
        expected = hashlib.sha256(b"").hexdigest()
        assert hash_val == expected

    def test_hash_format(self) -> None:
        """Test that hash is SHA-256 hex string."""
        processor = TokenProcessor()
        hash_val = processor.compute_chunk_hash([1, 2, 3])
        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_deterministic(self) -> None:
        """Test that hashing is deterministic."""
        processor = TokenProcessor()
        tokens = [1, 2, 3, 4, 5]
        hash1 = processor.compute_chunk_hash(tokens)
        hash2 = processor.compute_chunk_hash(tokens)
        assert hash1 == hash2

    def test_different_tokens_different_hash(self) -> None:
        """Test that different tokens produce different hashes."""
        processor = TokenProcessor()
        hash1 = processor.compute_chunk_hash([1, 2, 3])
        hash2 = processor.compute_chunk_hash([4, 5, 6])
        assert hash1 != hash2

    def test_order_matters(self) -> None:
        """Test that token order affects hash."""
        processor = TokenProcessor()
        hash1 = processor.compute_chunk_hash([1, 2, 3])
        hash2 = processor.compute_chunk_hash([3, 2, 1])
        assert hash1 != hash2


class TestFindPrefixMatch:
    """Tests for TokenProcessor.find_prefix_match method."""

    def test_empty_tokens(self) -> None:
        """Test prefix match with empty tokens."""
        processor = TokenProcessor(chunk_size=5)
        cached = {"some_hash"}
        matched, hashes = processor.find_prefix_match([], cached)
        assert matched == 0
        assert hashes == []

    def test_empty_cache(self) -> None:
        """Test prefix match with empty cache."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(10))
        matched, hashes = processor.find_prefix_match(tokens, set())
        assert matched == 0
        assert hashes == []

    def test_full_prefix_match(self) -> None:
        """Test when all chunks are cached."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(10))

        # Pre-compute hashes
        chunks = processor.chunk_tokens_with_metadata(tokens)
        cached = {c.chunk_hash for c in chunks}

        matched, hashes = processor.find_prefix_match(tokens, cached)
        assert matched == 10
        assert len(hashes) == 2

    def test_partial_prefix_match(self) -> None:
        """Test when only first chunk is cached."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(15))

        # Only cache first chunk
        first_chunk_hash = processor.compute_chunk_hash(tokens[:5])
        cached = {first_chunk_hash}

        matched, hashes = processor.find_prefix_match(tokens, cached)
        assert matched == 5
        assert len(hashes) == 1

    def test_no_match(self) -> None:
        """Test when no chunks are cached."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(10))
        cached = {"some_other_hash"}

        matched, hashes = processor.find_prefix_match(tokens, cached)
        assert matched == 0
        assert hashes == []

    def test_stops_at_first_miss(self) -> None:
        """Test that matching stops at first miss (prefix caching)."""
        processor = TokenProcessor(chunk_size=5)
        tokens = list(range(15))

        # Cache first and third chunks, but not second
        chunks = processor.chunk_tokens_with_metadata(tokens)
        cached = {chunks[0].chunk_hash, chunks[2].chunk_hash}

        # Should only match first chunk (stops at miss)
        matched, hashes = processor.find_prefix_match(tokens, cached)
        assert matched == 5
        assert len(hashes) == 1


class TestCacheKeyGenerator:
    """Tests for CacheKeyGenerator class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        gen = CacheKeyGenerator()
        assert gen.config.model_name == "llama"
        assert gen.config.world_size == 1
        assert gen.config.format_version == "v1"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        gen = CacheKeyGenerator(
            model_name="qwen",
            world_size=4,
            format_version="v2",
        )
        assert gen.config.model_name == "qwen"
        assert gen.config.world_size == 4
        assert gen.config.format_version == "v2"


class TestMakeKey:
    """Tests for CacheKeyGenerator.make_key method."""

    def test_basic_key_format(self) -> None:
        """Test basic key format."""
        gen = CacheKeyGenerator(model_name="llama", world_size=1)
        key = gen.make_key("abc123hash", worker_id=0)
        parts = key.split("@")
        assert len(parts) == 7
        assert parts[0] == "lmcache"
        assert parts[1] == "llama"
        assert parts[2] == "1"
        assert parts[3] == "0"
        assert parts[5] == "abc123hash"
        assert parts[6] == "kv"

    def test_custom_suffix(self) -> None:
        """Test custom suffix."""
        gen = CacheKeyGenerator()
        key = gen.make_key("hash", suffix="meta")
        assert key.endswith("@meta")

    def test_layer_range(self) -> None:
        """Test layer range in key."""
        gen = CacheKeyGenerator()
        key = gen.make_key("hash", layer_start=0, layer_end=10)
        parts = key.split("@")
        assert parts[4] == "0-10"

    def test_layer_range_default(self) -> None:
        """Test default layer range (layer_end = layer_start)."""
        gen = CacheKeyGenerator()
        key = gen.make_key("hash", layer_start=5)
        parts = key.split("@")
        assert parts[4] == "5-5"

    def test_worker_id(self) -> None:
        """Test worker ID in key."""
        gen = CacheKeyGenerator()
        key = gen.make_key("hash", worker_id=3)
        parts = key.split("@")
        assert parts[3] == "3"


class TestMakeKeysForAllWorkers:
    """Tests for CacheKeyGenerator.make_keys_for_all_workers method."""

    def test_single_worker(self) -> None:
        """Test keys for single worker."""
        gen = CacheKeyGenerator(world_size=1)
        keys = gen.make_keys_for_all_workers("hash")
        assert len(keys) == 1

    def test_multiple_workers(self) -> None:
        """Test keys for multiple workers."""
        gen = CacheKeyGenerator(world_size=4)
        keys = gen.make_keys_for_all_workers("hash")
        assert len(keys) == 4

    def test_worker_ids_sequential(self) -> None:
        """Test that worker IDs are sequential."""
        gen = CacheKeyGenerator(world_size=4)
        keys = gen.make_keys_for_all_workers("hash")
        for i, key in enumerate(keys):
            parts = key.split("@")
            assert parts[3] == str(i)

    def test_same_hash_different_workers(self) -> None:
        """Test that all keys have same hash but different workers."""
        gen = CacheKeyGenerator(world_size=4)
        keys = gen.make_keys_for_all_workers("abc123")
        hashes = [k.split("@")[5] for k in keys]
        worker_ids = [k.split("@")[3] for k in keys]
        assert all(h == "abc123" for h in hashes)
        assert len(set(worker_ids)) == 4


class TestMakeKeysForLayers:
    """Tests for CacheKeyGenerator.make_keys_for_layers method."""

    def test_single_layer_shards(self) -> None:
        """Test layer-wise keys with single layer per shard."""
        gen = CacheKeyGenerator()
        keys = gen.make_keys_for_layers("hash", num_layers=32, layers_per_shard=1)
        assert len(keys) == 32

    def test_multi_layer_shards(self) -> None:
        """Test layer-wise keys with multiple layers per shard."""
        gen = CacheKeyGenerator()
        keys = gen.make_keys_for_layers("hash", num_layers=32, layers_per_shard=8)
        assert len(keys) == 4

    def test_layer_ranges(self) -> None:
        """Test that layer ranges are correct."""
        gen = CacheKeyGenerator()
        keys = gen.make_keys_for_layers("hash", num_layers=10, layers_per_shard=3)
        layer_ranges = [k.split("@")[4] for k in keys]
        assert layer_ranges == ["0-2", "3-5", "6-8", "9-9"]


class TestParseKey:
    """Tests for CacheKeyGenerator.parse_key method."""

    def test_parse_valid_key(self) -> None:
        """Test parsing a valid key."""
        gen = CacheKeyGenerator()
        key = "lmcache@llama@4@2@0-10@abc123@kv"
        parsed = gen.parse_key(key)
        assert parsed["format"] == "lmcache"
        assert parsed["model"] == "llama"
        assert parsed["world_size"] == 4
        assert parsed["worker_id"] == 2
        assert parsed["layer_start"] == 0
        assert parsed["layer_end"] == 10
        assert parsed["chunk_hash"] == "abc123"
        assert parsed["suffix"] == "kv"

    def test_parse_roundtrip(self) -> None:
        """Test that make_key and parse_key are inverse operations."""
        gen = CacheKeyGenerator(model_name="qwen", world_size=8)
        original_key = gen.make_key("myhash", worker_id=3, layer_start=5, layer_end=15)
        parsed = gen.parse_key(original_key)
        assert parsed["model"] == "qwen"
        assert parsed["world_size"] == 8
        assert parsed["worker_id"] == 3
        assert parsed["chunk_hash"] == "myhash"
        assert parsed["layer_start"] == 5
        assert parsed["layer_end"] == 15

    def test_parse_invalid_key_too_few_parts(self) -> None:
        """Test parsing key with too few parts."""
        gen = CacheKeyGenerator()
        with pytest.raises(ValueError, match="expected 7 components"):
            gen.parse_key("lmcache@llama@4@2@0-10@abc123")

    def test_parse_invalid_layer_range(self) -> None:
        """Test parsing key with invalid layer range."""
        gen = CacheKeyGenerator()
        with pytest.raises(ValueError, match="Invalid layer range"):
            gen.parse_key("lmcache@llama@4@2@invalid@abc123@kv")


class TestMakeMetadataKey:
    """Tests for CacheKeyGenerator.make_metadata_key method."""

    def test_metadata_key_suffix(self) -> None:
        """Test that metadata key has 'meta' suffix."""
        gen = CacheKeyGenerator()
        key = gen.make_metadata_key("hash")
        assert key.endswith("@meta")

    def test_metadata_key_structure(self) -> None:
        """Test metadata key structure."""
        gen = CacheKeyGenerator(model_name="llama", world_size=4)
        key = gen.make_metadata_key("abc123", worker_id=2)
        parsed = gen.parse_key(key)
        assert parsed["chunk_hash"] == "abc123"
        assert parsed["worker_id"] == 2
        assert parsed["suffix"] == "meta"
