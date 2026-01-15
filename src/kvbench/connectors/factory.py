"""
KV Connector Factory.

This module provides a factory function for creating KV connectors
based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kvbench.connectors.base import KVConnector
from kvbench.connectors.dynamo.connector import DynamoConnector
from kvbench.connectors.lmcache.connector import LMCacheConfig, LMCacheConnector
from kvbench.connectors.mooncake.connector import MooncakeConnector

if TYPE_CHECKING:
    from kvbench.core.config import ConnectorConfig
    from kvbench.storage.base import StorageBackend


def create_connector(
    config: ConnectorConfig,
    storage: StorageBackend,
    name: str | None = None,
) -> KVConnector:
    """Create a KV connector based on configuration.

    Args:
        config: Connector configuration specifying the connector type.
        storage: Storage backend for the connector to use.
        name: Optional name for the connector instance.

    Returns:
        An initialized KV connector instance.

    Raises:
        ValueError: If the connector type is unknown.
    """
    connector_name = name or config.connector_type

    if config.connector_type == "lmcache":
        lmcache_config = LMCacheConfig(
            chunk_size=config.lmcache_chunk_size,
        )
        return LMCacheConnector(
            storage=storage,
            config=lmcache_config,
            name=connector_name,
        )

    elif config.connector_type == "mooncake":
        return MooncakeConnector(
            storage=storage,
            name=connector_name,
        )

    elif config.connector_type == "dynamo":
        return DynamoConnector(
            storage=storage,
            name=connector_name,
        )

    elif config.connector_type == "mock":
        # Mock connector uses LMCache with default settings
        return LMCacheConnector(
            storage=storage,
            config=LMCacheConfig(),
            name=connector_name,
        )

    else:
        raise ValueError(f"Unknown connector type: {config.connector_type}")
