"""
FastAPI Application for KV-Bench Server.

This module provides a FastAPI application that exposes OpenAI-compatible
endpoints for the KV-Bench mock LLM server.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kvbench.connectors.factory import create_connector
from kvbench.core.config import KVBenchConfig
from kvbench.servers.combined import CombinedServer
from kvbench.servers.decode import DecodeServer
from kvbench.servers.factory import create_server
from kvbench.servers.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelList,
)
from kvbench.servers.prefill import PrefillServer
from kvbench.servers.proxy import DisaggregatedProxy
from kvbench.storage.factory import create_storage_backend

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class KVBenchApp:
    """KV-Bench FastAPI application wrapper.

    This class manages the lifecycle of the server and provides
    the FastAPI application instance.

    Attributes:
        config: KV-Bench configuration.
        app: FastAPI application instance.
    """

    def __init__(self, config: KVBenchConfig | None = None) -> None:
        """Initialize the KV-Bench application.

        Args:
            config: KV-Bench configuration. Uses defaults if not provided.
        """
        self.config = config or KVBenchConfig()
        self._server: PrefillServer | DecodeServer | CombinedServer | DisaggregatedProxy | None = None
        self._storage = None
        self._connector = None

        @asynccontextmanager
        async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
            """Manage application lifecycle."""
            await self._startup()
            yield
            await self._shutdown()

        self.app = FastAPI(
            title="KV-Bench",
            description="Mock LLM server for KV cache benchmarking",
            version="0.1.0",
            lifespan=lifespan,
        )
        self._setup_routes()

    async def _startup(self) -> None:
        """Initialize server components on startup."""
        logger.info(f"Starting KV-Bench server (type: {self.config.server.server_type})")

        # Create storage backend
        self._storage = create_storage_backend(
            self.config.storage,
            self.config.resources,
            name="kvbench-storage",
        )

        # Create connector (not needed for proxy)
        if self.config.server.server_type != "proxy":
            self._connector = create_connector(
                self.config.connector,
                self._storage,
                name="kvbench-connector",
            )

        # Create server
        self._server = create_server(self.config, self._connector)
        await self._server.start()

        logger.info(f"KV-Bench server started on {self.config.server.host}:{self.config.server.port}")

    async def _shutdown(self) -> None:
        """Cleanup server components on shutdown."""
        logger.info("Shutting down KV-Bench server...")

        if self._server:
            await self._server.stop()

        if self._connector and hasattr(self._connector, "close"):
            await self._connector.close()

        if self._storage and hasattr(self._storage, "close"):
            await self._storage.close()

        logger.info("KV-Bench server stopped")

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health_check() -> HealthResponse:
            """Health check endpoint."""
            if self._server is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            return await self._server.health_check()

        @self.app.get("/metrics")
        async def get_metrics() -> MetricsResponse:
            """Get server metrics."""
            if self._server is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            return await self._server.get_metrics()

        @self.app.get("/v1/models")
        async def list_models() -> ModelList:
            """List available models (OpenAI-compatible)."""
            if self._server is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            return await self._server.list_models()

        @self.app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse | StreamingResponse:
            """Chat completion endpoint (OpenAI-compatible)."""
            if self._server is None:
                raise HTTPException(status_code=503, detail="Server not initialized")

            # Handle streaming
            if request.stream:
                return StreamingResponse(
                    self._stream_response(request),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            # Non-streaming response
            if isinstance(self._server, (CombinedServer, DisaggregatedProxy, PrefillServer)):
                result = await self._server.chat_completions(request)
            elif isinstance(self._server, DecodeServer):
                # For decode-only server, assume context_length from prompt
                prompt = request.prompt_text or ""
                context_length = max(1, len(prompt) // 4)
                result = await self._server.chat_completions(request, context_length)
            else:
                raise HTTPException(status_code=500, detail="Unknown server type")

            if isinstance(result, ErrorResponse):
                return JSONResponse(
                    status_code=500,
                    content=result.model_dump(),
                )

            return result

        @self.app.exception_handler(Exception)
        async def global_exception_handler(
            _request: Request, exc: Exception
        ) -> JSONResponse:
            """Handle uncaught exceptions."""
            logger.exception(f"Unhandled exception: {exc}")
            error = ErrorResponse.create(str(exc), "server_error")
            return JSONResponse(status_code=500, content=error.model_dump())

    async def _stream_response(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        if self._server is None:
            yield ErrorResponse.create("Server not initialized", "server_error").model_dump_json()
            return

        if isinstance(self._server, (CombinedServer, DisaggregatedProxy)):
            async for chunk in self._server.chat_completions_stream(request):
                yield chunk
        elif isinstance(self._server, DecodeServer):
            prompt = request.prompt_text or ""
            context_length = max(1, len(prompt) // 4)
            async for chunk in self._server.chat_completions_stream(request, context_length):
                yield chunk
        else:
            # PrefillServer doesn't support streaming well
            error = ErrorResponse.create(
                "Prefill-only server doesn't support streaming",
                "invalid_request_error",
            )
            yield f"data: {error.model_dump_json()}\n\n"


def create_app(config: KVBenchConfig | None = None) -> FastAPI:
    """Create and return the FastAPI application.

    Args:
        config: KV-Bench configuration. Uses defaults if not provided.

    Returns:
        Configured FastAPI application.
    """
    kvbench_app = KVBenchApp(config)
    return kvbench_app.app


# Lazy app instance for uvicorn
# Created on first access to avoid import-time initialization
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get or create the default FastAPI application.

    This function lazily creates the app on first access,
    avoiding import-time initialization issues.
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app


# For uvicorn: kvbench.servers.app:app
# We need a callable or the app instance
app = get_app
