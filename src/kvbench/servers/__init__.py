"""
KV-Bench Servers Module.

This module provides server implementations for mock LLM serving:
- PrefillServer: Handles the prefill phase of inference
- DecodeServer: Handles the decode phase with streaming
- CombinedServer: Single-node server with both phases
- DisaggregatedProxy: Routes requests to prefill/decode servers

All servers provide an OpenAI-compatible API.
"""

from kvbench.servers.combined import CombinedServer, CombinedStats
from kvbench.servers.decode import DecodeServer, DecodeStats
from kvbench.servers.factory import create_server
from kvbench.servers.openai_compat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ErrorResponse,
    HealthResponse,
    MessageRole,
    MetricsResponse,
    ModelInfo,
    ModelList,
    ResponseFormat,
    StreamChoice,
    Usage,
)
from kvbench.servers.prefill import PrefillServer, PrefillStats
from kvbench.servers.proxy import (
    DisaggregatedProxy,
    LoadBalanceStrategy,
    ProxyStats,
    ServerEndpoint,
)

__all__ = [
    # Servers
    "PrefillServer",
    "PrefillStats",
    "DecodeServer",
    "DecodeStats",
    "CombinedServer",
    "CombinedStats",
    "DisaggregatedProxy",
    "LoadBalanceStrategy",
    "ProxyStats",
    "ServerEndpoint",
    "create_server",
    # OpenAI Types
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "ChatMessage",
    "MessageRole",
    "Choice",
    "ChoiceMessage",
    "StreamChoice",
    "DeltaMessage",
    "Usage",
    "ResponseFormat",
    "ErrorResponse",
    "HealthResponse",
    "MetricsResponse",
    "ModelInfo",
    "ModelList",
]
