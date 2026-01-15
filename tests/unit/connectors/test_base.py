"""Unit tests for kvbench.connectors.base module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from kvbench.connectors.base import ChunkMetadata, ConnectorStats


class TestConnectorStats:
    """Tests for ConnectorStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        stats = ConnectorStats()
        assert stats.stores == 0
        assert stats.store_bytes == 0
        assert stats.loads == 0
        assert stats.load_bytes == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.errors == 0
        assert isinstance(stats.created_at, datetime)

    def test_hit_rate_no_requests(self) -> None:
        """Test hit rate with no requests."""
        stats = ConnectorStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = ConnectorStats(hits=75, misses=25)
        assert stats.hit_rate == 75.0

    def test_total_operations(self) -> None:
        """Test total operations calculation."""
        stats = ConnectorStats(stores=10, loads=20)
        assert stats.total_operations == 30


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        meta = ChunkMetadata(
            chunk_hash="abc123",
            num_tokens=256,
            size_bytes=131072,
        )
        assert meta.chunk_hash == "abc123"
        assert meta.num_tokens == 256
        assert meta.size_bytes == 131072
        assert meta.model_name is None
        assert meta.layer_range is None
        assert meta.worker_id == 0
        assert isinstance(meta.created_at, datetime)

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        meta = ChunkMetadata(
            chunk_hash="xyz789",
            num_tokens=128,
            size_bytes=65536,
            model_name="llama-3.1-8b",
            layer_range=(0, 16),
            worker_id=2,
        )
        assert meta.model_name == "llama-3.1-8b"
        assert meta.layer_range == (0, 16)
        assert meta.worker_id == 2
