"""
OpenAI-Compatible API Types.

This module defines Pydantic models for OpenAI-compatible chat completion
request and response types. These types enable compatibility with tools
like GenAI-Perf that expect the OpenAI API format.

Reference: https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A message in a chat conversation.

    Attributes:
        role: The role of the message author.
        content: The content of the message.
        name: Optional name of the author.
        function_call: Deprecated function call (for backward compatibility).
        tool_calls: Tool calls made by the assistant.
    """

    role: MessageRole
    content: str | list[Any] | None = None
    name: str | None = None
    function_call: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ResponseFormat(BaseModel):
    """Response format specification.

    Attributes:
        type: The type of response format.
    """

    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    Attributes:
        model: The model to use for completion.
        messages: The messages in the conversation.
        temperature: Sampling temperature (0-2).
        top_p: Nucleus sampling probability.
        n: Number of completions to generate.
        stream: Whether to stream partial results.
        stop: Stop sequences.
        max_tokens: Maximum tokens to generate.
        presence_penalty: Presence penalty (-2 to 2).
        frequency_penalty: Frequency penalty (-2 to 2).
        logit_bias: Token logit biases.
        user: Unique user identifier.
        response_format: Response format specification.
        seed: Random seed for deterministic sampling.
    """

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1, le=128)
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None

    @property
    def prompt_text(self) -> str:
        """Extract the full prompt text from messages.

        Tolerates OpenAI content-parts lists (``[{"type": "text",
        "text": ...}, ...]``) by concatenating their text segments.
        """
        parts = []
        for msg in self.messages:
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    seg.get("text", "") if isinstance(seg, dict) else str(seg)
                    for seg in content
                )
            if content:
                parts.append(f"{msg.role.value}: {content}")
        return "\n".join(parts)

    @property
    def last_user_message(self) -> str | None:
        """Get the last user message content."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER and msg.content:
                return msg.content
        return None


class Usage(BaseModel):
    """Token usage statistics.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total number of tokens used.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    """Message in a completion choice.

    Attributes:
        role: The role of the message author.
        content: The content of the message.
    """

    role: Literal["assistant"] = "assistant"
    content: str | None = None


class Choice(BaseModel):
    """A completion choice.

    Attributes:
        index: The index of this choice.
        message: The generated message.
        finish_reason: Why the completion finished.
    """

    index: int
    message: ChoiceMessage
    finish_reason: Literal["stop", "length", "content_filter", "function_call", "tool_calls"] | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response.

    Attributes:
        id: Unique identifier for the completion.
        object: Object type (always "chat.completion").
        created: Unix timestamp of creation.
        model: The model used.
        choices: The completion choices.
        usage: Token usage statistics.
        system_fingerprint: System fingerprint for reproducibility.
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None

    @classmethod
    def create(
        cls,
        model: str,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str = "stop",
    ) -> ChatCompletionResponse:
        """Create a simple completion response.

        Args:
            model: The model name.
            content: The generated content.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            finish_reason: Why the completion finished.

        Returns:
            A ChatCompletionResponse instance.
        """
        return cls(
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(content=content),
                    finish_reason=finish_reason,  # type: ignore[arg-type]
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class DeltaMessage(BaseModel):
    """Delta message for streaming responses.

    Attributes:
        role: The role of the message author (only in first chunk).
        content: The content delta.
    """

    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """A streaming choice.

    Attributes:
        index: The index of this choice.
        delta: The message delta.
        finish_reason: Why the completion finished (None until done).
    """

    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chat completion chunk.

    Attributes:
        id: Unique identifier for the completion.
        object: Object type (always "chat.completion.chunk").
        created: Unix timestamp of creation.
        model: The model used.
        choices: The streaming choices.
        system_fingerprint: System fingerprint for reproducibility.
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    system_fingerprint: str | None = None

    @classmethod
    def create_start(cls, model: str, completion_id: str | None = None) -> ChatCompletionChunk:
        """Create the first chunk with role.

        Args:
            model: The model name.
            completion_id: Optional ID to use.

        Returns:
            A ChatCompletionChunk for the start of streaming.
        """
        chunk = cls(
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                )
            ],
        )
        if completion_id:
            chunk.id = completion_id
        return chunk

    @classmethod
    def create_content(
        cls,
        model: str,
        content: str,
        completion_id: str | None = None,
    ) -> ChatCompletionChunk:
        """Create a content chunk.

        Args:
            model: The model name.
            content: The content delta.
            completion_id: Optional ID to use.

        Returns:
            A ChatCompletionChunk with content.
        """
        chunk = cls(
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=content),
                )
            ],
        )
        if completion_id:
            chunk.id = completion_id
        return chunk

    @classmethod
    def create_end(
        cls,
        model: str,
        completion_id: str | None = None,
        finish_reason: str = "stop",
    ) -> ChatCompletionChunk:
        """Create the final chunk.

        Args:
            model: The model name.
            completion_id: Optional ID to use.
            finish_reason: Why the completion finished.

        Returns:
            A ChatCompletionChunk for the end of streaming.
        """
        chunk = cls(
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason=finish_reason,  # type: ignore[arg-type]
                )
            ],
        )
        if completion_id:
            chunk.id = completion_id
        return chunk

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format.

        Returns:
            SSE-formatted string.
        """
        return f"data: {self.model_dump_json()}\n\n"


class ModelInfo(BaseModel):
    """Model information.

    Attributes:
        id: Model identifier.
        object: Object type (always "model").
        created: Unix timestamp of creation.
        owned_by: Organization that owns the model.
    """

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "kvbench"


class ModelList(BaseModel):
    """List of available models.

    Attributes:
        object: Object type (always "list").
        data: List of model information.
    """

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ErrorResponse(BaseModel):
    """Error response.

    Attributes:
        error: Error details.
    """

    error: dict[str, Any]

    @classmethod
    def create(
        cls,
        message: str,
        error_type: str = "invalid_request_error",
        code: str | None = None,
    ) -> ErrorResponse:
        """Create an error response.

        Args:
            message: Error message.
            error_type: Type of error.
            code: Optional error code.

        Returns:
            An ErrorResponse instance.
        """
        error: dict[str, Any] = {
            "message": message,
            "type": error_type,
        }
        if code:
            error["code"] = code
        return cls(error=error)


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Health status.
        instance_id: Instance identifier.
        server_type: Type of server.
        model: Model being served.
        uptime_seconds: Server uptime.
    """

    status: Literal["healthy", "degraded", "unhealthy"]
    instance_id: str
    server_type: str
    model: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Metrics response.

    Attributes:
        requests_total: Total number of requests.
        requests_success: Number of successful requests.
        requests_failed: Number of failed requests.
        tokens_processed: Total tokens processed.
        cache_hits: Number of cache hits.
        cache_misses: Number of cache misses.
        avg_latency_ms: Average latency in milliseconds.
    """

    requests_total: int
    requests_success: int
    requests_failed: int
    tokens_processed: int
    cache_hits: int
    cache_misses: int
    avg_latency_ms: float
