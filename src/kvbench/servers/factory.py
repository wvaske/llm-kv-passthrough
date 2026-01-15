"""
Server Factory.

This module provides a factory function for creating server instances
based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kvbench.servers.combined import CombinedServer
from kvbench.servers.decode import DecodeServer
from kvbench.servers.prefill import PrefillServer
from kvbench.servers.proxy import DisaggregatedProxy

if TYPE_CHECKING:
    from kvbench.connectors.base import KVConnector
    from kvbench.core.config import KVBenchConfig

ServerType = PrefillServer | DecodeServer | CombinedServer | DisaggregatedProxy


def create_server(
    config: KVBenchConfig,
    connector: KVConnector | None = None,
) -> ServerType:
    """Create a server based on configuration.

    Args:
        config: KV-Bench configuration specifying server type.
        connector: KV cache connector (required for prefill/decode/combined).

    Returns:
        An initialized server instance.

    Raises:
        ValueError: If the server type is unknown or connector is missing.
    """
    server_type = config.server.server_type

    if server_type == "prefill":
        if connector is None:
            raise ValueError("connector is required for prefill server")
        return PrefillServer(config=config, connector=connector)

    elif server_type == "decode":
        if connector is None:
            raise ValueError("connector is required for decode server")
        return DecodeServer(config=config, connector=connector)

    elif server_type == "combined":
        if connector is None:
            raise ValueError("connector is required for combined server")
        return CombinedServer(config=config, connector=connector)

    elif server_type == "proxy":
        return DisaggregatedProxy(config=config)

    else:
        raise ValueError(f"Unknown server type: {server_type}")
