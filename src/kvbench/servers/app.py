"""Memory-bounded OpenAI-shaped API for the V1 storage workload engine."""

import json
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from kvbench.core.config import KVBenchConfig
from kvbench.kv.base import KVStack
from kvbench.kv.factory import create_kv_stack
from kvbench.metrics.prometheus import CONTENT_TYPE_LATEST, MetricsExporter
from kvbench.servers.factory import create_server
from kvbench.servers.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    MetricsResponse,
    ModelList,
)
from kvbench.servers.rate_server import RateServer
from kvbench.servers.warmup import WarmupController, WarmupRequest


class RequestLimitsMiddleware:
    """Bound chat body size and outstanding HTTP requests before JSON parsing."""

    def __init__(self, app: ASGIApp, max_bytes: int, max_requests: int) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.max_requests = max_requests
        self.inflight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/v1/chat/completions":
            return await self.app(scope, receive, send)
        if self.inflight >= self.max_requests:
            return await JSONResponse({"error": {"message": "Request queue is full"}}, status_code=503)(scope, receive, send)
        self.inflight += 1
        try:
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.max_bytes:
                    return await JSONResponse({"error": {"message": "Request body too large"}}, status_code=413)(scope, receive, send)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def limited_receive() -> Message:
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, limited_receive, send)
        finally:
            self.inflight -= 1


class KVBenchApp:
    def __init__(self, config: KVBenchConfig | None = None):
        self.config = config or KVBenchConfig()
        self._server: RateServer | None = None
        self._kv: KVStack | None = None
        self._warmup: WarmupController | None = None
        self._metrics = MetricsExporter(self)

        @asynccontextmanager
        async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
            try:
                await self._startup()
                yield
            finally:
                await self._shutdown()

        self.app = FastAPI(title="KV-Bench", version="1.0.0", lifespan=lifespan)
        s = self.config.scheduler
        self.app.add_middleware(RequestLimitsMiddleware, max_bytes=s.max_request_bytes,
                                max_requests=s.max_active_requests + s.max_pending_requests)
        self._setup_routes()

    async def _startup(self) -> None:
        self.config.validate_serving()
        self._kv = create_kv_stack(self.config)
        try:
            await self._kv.start()
            self._warmup = WarmupController(self._kv)
            self._server = create_server(self.config, self._kv)
            await self._server.start()
        except BaseException:
            await self._kv.close()
            raise

    async def _shutdown(self) -> None:
        try:
            if self._warmup is not None:
                await self._warmup.cancel()
            if self._server is not None:
                await self._server.stop()
        finally:
            if self._kv is not None:
                await self._kv.close()

    def _require_server(self) -> RateServer:
        if self._server is None:
            raise HTTPException(503, "Server not initialized")
        return self._server

    def _require_kv(self) -> KVStack:
        if self._kv is None:
            raise HTTPException(503, "KV manager not initialized")
        return self._kv

    def _require_warmup(self) -> WarmupController:
        if self._warmup is None:
            raise HTTPException(503, "Warmup controller not initialized")
        return self._warmup

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> JSONResponse:
            result = await self._require_server().health_check()
            return JSONResponse(result.model_dump(), status_code=200 if result.status == "healthy" else 503)

        @self.app.get("/metrics")
        async def metrics() -> Response:
            if not self.config.metrics.enabled:
                raise HTTPException(404, "Metrics disabled")
            return Response(self._metrics.render(), media_type=CONTENT_TYPE_LATEST)

        @self.app.get("/stats")
        @self.app.get("/kvbench/effective")
        async def stats() -> MetricsResponse:
            return await self._require_server().get_metrics()

        @self.app.get("/kvbench/manifest")
        async def manifest() -> dict[str, Any]:
            result = self.config.manifest()
            cfg = getattr(self._kv, "_lmcache_config", None)
            if cfg is not None:
                # Some LMCache defaults are infinite thresholds. Preserve them
                # as explicit strings rather than emitting invalid JSON floats.
                result["lmcache"] = json.loads(json.dumps(cfg.to_dict(), default=str), parse_constant=str)
            provenance = getattr(self._kv, "provenance", None)
            if provenance is not None:
                result["hicache"] = provenance()
            return result

        @self.app.get("/kvbench/state")
        async def state() -> dict[str, Any]:
            server = self._require_server()
            kv = self._require_kv()
            from dataclasses import asdict

            return {"capacity": kv.capacity_info(), "usage": kv.usage_info(),
                    "kv_stats": {**asdict(kv.stats), "token_hit_rate": kv.stats.token_hit_rate},
                    "scheduler": server.scheduler.info()}

        @self.app.post("/kvbench/flush")
        async def flush() -> dict[str, bool]:
            server = self._require_server()
            if server.scheduler.active or server.scheduler.pending or self._require_warmup().running:
                raise HTTPException(409, "Stop serving and warmup before draining storage")
            await self._require_kv().flush()
            return {"drained": True, "fsync": False}

        @self.app.get("/v1/models")
        async def models() -> ModelList:
            return await self._require_server().list_models()

        @self.app.post("/kvbench/warmup", status_code=202)
        async def warmup(request: WarmupRequest) -> dict[str, Any]:
            server = self._require_server()
            if server.scheduler.active or server.scheduler.pending:
                raise HTTPException(409, "Cannot warm up during serving traffic")
            try:
                return self._require_warmup().start(request).to_dict()
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc

        @self.app.get("/kvbench/warmup")
        async def warmup_status() -> dict[str, Any]:
            self._require_server()
            return self._require_warmup().status.to_dict()

        @self.app.delete("/kvbench/warmup")
        async def cancel_warmup() -> dict[str, Any]:
            self._require_server()
            warmup = self._require_warmup()
            await warmup.cancel()
            return warmup.status.to_dict()

        @self.app.post("/v1/chat/completions", response_model=None)
        async def chat(request: ChatCompletionRequest) -> Response | ChatCompletionResponse:
            server = self._require_server()
            if self._require_warmup().running:
                raise HTTPException(503, "Warmup is running; wait before measuring")
            try:
                server.validate_request(request)
            except ValueError as exc:
                return JSONResponse(ErrorResponse.create(str(exc)).model_dump(), status_code=400)
            if request.stream:
                return StreamingResponse(self._stream_response(request), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            started = time.monotonic()
            try:
                result = await server.chat_completions(request)
            finally:
                self._metrics.request_duration.observe(time.monotonic() - started)
            if isinstance(result, ErrorResponse):
                return JSONResponse(result.model_dump(), status_code=int(result.error.get("code", 500)))
            return result

        @self.app.exception_handler(Exception)
        async def failure(_request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(ErrorResponse.create(str(exc), "server_error").model_dump(), status_code=500)

    async def _stream_response(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        started = time.monotonic()
        first = True
        stream = self._require_server().chat_completions_stream(request)
        try:
            async for chunk in stream:
                if first and chunk.startswith("data: {"):
                    payload = json.loads(chunk[6:])
                    if any(c.get("delta", {}).get("content") for c in payload.get("choices", [])):
                        self._metrics.ttft.observe(time.monotonic() - started)
                        first = False
                yield chunk
        finally:
            await stream.aclose()
            self._metrics.request_duration.observe(time.monotonic() - started)


def create_app(config: KVBenchConfig | None = None) -> FastAPI:
    return KVBenchApp(config).app


def load_config() -> KVBenchConfig:
    path = os.environ.get("KVBENCH_CONFIG_FILE")
    return KVBenchConfig.from_yaml(path) if path else KVBenchConfig.from_env()


_app: FastAPI | None = None


def get_app() -> FastAPI:
    global _app
    if _app is None:
        _app = create_app(load_config())
    return _app


app = get_app
