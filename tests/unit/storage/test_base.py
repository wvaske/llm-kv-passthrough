"""Unit tests for kvbench.storage.base module."""

from __future__ import annotations

from datetime import datetime, timedelta

from kvbench.storage.base import StorageItem, StorageStats


class TestStorageStats:
    """Tests for StorageStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        stats = StorageStats(total_bytes=1000)
        assert stats.total_bytes == 1000
        assert stats.used_bytes == 0
        assert stats.item_count == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert isinstance(stats.created_at, datetime)

    def test_available_bytes_property(self) -> None:
        """Test available_bytes calculation."""
        stats = StorageStats(total_bytes=1000, used_bytes=300)
        assert stats.available_bytes == 700

    def test_available_bytes_negative_handled(self) -> None:
        """Test available_bytes handles overflow gracefully."""
        stats = StorageStats(total_bytes=100, used_bytes=200)
        assert stats.available_bytes == 0

    def test_utilization_property(self) -> None:
        """Test utilization percentage calculation."""
        stats = StorageStats(total_bytes=1000, used_bytes=250)
        assert stats.utilization == 25.0

    def test_utilization_zero_total(self) -> None:
        """Test utilization with zero total bytes."""
        stats = StorageStats(total_bytes=0)
        assert stats.utilization == 0.0

    def test_hit_rate_property(self) -> None:
        """Test hit rate calculation."""
        stats = StorageStats(total_bytes=1000, hits=75, misses=25)
        assert stats.hit_rate == 75.0

    def test_hit_rate_no_requests(self) -> None:
        """Test hit rate with no requests."""
        stats = StorageStats(total_bytes=1000)
        assert stats.hit_rate == 0.0

    def test_full_utilization(self) -> None:
        """Test 100% utilization."""
        stats = StorageStats(total_bytes=1000, used_bytes=1000)
        assert stats.utilization == 100.0
        assert stats.available_bytes == 0


class TestStorageItem:
    """Tests for StorageItem dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        item = StorageItem(key="test", size_bytes=100)
        assert item.key == "test"
        assert item.size_bytes == 100
        assert isinstance(item.created_at, datetime)
        assert isinstance(item.accessed_at, datetime)
        assert item.ttl_seconds is None
        assert item.metadata is None

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        now = datetime.now()
        item = StorageItem(
            key="test",
            size_bytes=200,
            created_at=now,
            accessed_at=now,
            ttl_seconds=60,
            metadata={"custom": "data"},
        )
        assert item.ttl_seconds == 60
        assert item.metadata == {"custom": "data"}

    def test_is_expired_no_ttl(self) -> None:
        """Test is_expired returns False when no TTL is set."""
        item = StorageItem(key="test", size_bytes=100)
        assert item.is_expired is False

    def test_is_expired_not_expired(self) -> None:
        """Test is_expired returns False when TTL hasn't elapsed."""
        item = StorageItem(
            key="test",
            size_bytes=100,
            ttl_seconds=3600,  # 1 hour
        )
        assert item.is_expired is False

    def test_is_expired_expired(self) -> None:
        """Test is_expired returns True when TTL has elapsed."""
        past = datetime.now() - timedelta(seconds=120)
        item = StorageItem(
            key="test",
            size_bytes=100,
            created_at=past,
            ttl_seconds=60,  # Expired 60 seconds ago
        )
        assert item.is_expired is True
