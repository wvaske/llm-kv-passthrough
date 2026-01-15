"""Unit tests for kvbench.servers.openai_compat module."""

from __future__ import annotations

import time

import pytest

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


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_user_message(self) -> None:
        """Test creating a user message."""
        msg = ChatMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_assistant_message(self) -> None:
        """Test creating an assistant message."""
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="Hi there")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there"

    def test_system_message(self) -> None:
        """Test creating a system message."""
        msg = ChatMessage(role=MessageRole.SYSTEM, content="You are helpful")
        assert msg.role == MessageRole.SYSTEM


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_basic_request(self) -> None:
        """Test creating a basic request."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        )
        assert request.model == "llama-3.1-8b"
        assert len(request.messages) == 1
        assert request.stream is False

    def test_streaming_request(self) -> None:
        """Test creating a streaming request."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            stream=True,
            max_tokens=100,
        )
        assert request.stream is True
        assert request.max_tokens == 100

    def test_prompt_text_property(self) -> None:
        """Test prompt_text property."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content="You are helpful"),
                ChatMessage(role=MessageRole.USER, content="Hello"),
            ],
        )
        prompt = request.prompt_text
        assert "system: You are helpful" in prompt
        assert "user: Hello" in prompt

    def test_last_user_message_property(self) -> None:
        """Test last_user_message property."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[
                ChatMessage(role=MessageRole.USER, content="First"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
                ChatMessage(role=MessageRole.USER, content="Second"),
            ],
        )
        assert request.last_user_message == "Second"

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            temperature=0.5,
        )
        assert request.temperature == 0.5

        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="llama-3.1-8b",
                messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
                temperature=3.0,  # Invalid: > 2.0
            )


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse model."""

    def test_create_helper(self) -> None:
        """Test create helper method."""
        response = ChatCompletionResponse.create(
            model="llama-3.1-8b",
            content="Hello there",
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert response.model == "llama-3.1-8b"
        assert response.choices[0].message.content == "Hello there"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    def test_auto_generated_id(self) -> None:
        """Test that ID is auto-generated."""
        response = ChatCompletionResponse.create(
            model="llama-3.1-8b",
            content="Test",
            prompt_tokens=1,
            completion_tokens=1,
        )
        assert response.id.startswith("chatcmpl-")

    def test_object_type(self) -> None:
        """Test that object type is correct."""
        response = ChatCompletionResponse.create(
            model="llama-3.1-8b",
            content="Test",
            prompt_tokens=1,
            completion_tokens=1,
        )
        assert response.object == "chat.completion"


class TestChatCompletionChunk:
    """Tests for ChatCompletionChunk model."""

    def test_create_start(self) -> None:
        """Test creating start chunk."""
        chunk = ChatCompletionChunk.create_start(model="llama-3.1-8b")
        assert chunk.model == "llama-3.1-8b"
        assert chunk.choices[0].delta.role == "assistant"
        assert chunk.object == "chat.completion.chunk"

    def test_create_content(self) -> None:
        """Test creating content chunk."""
        chunk = ChatCompletionChunk.create_content(
            model="llama-3.1-8b", content="Hello"
        )
        assert chunk.choices[0].delta.content == "Hello"

    def test_create_end(self) -> None:
        """Test creating end chunk."""
        chunk = ChatCompletionChunk.create_end(
            model="llama-3.1-8b", finish_reason="stop"
        )
        assert chunk.choices[0].finish_reason == "stop"

    def test_to_sse(self) -> None:
        """Test SSE formatting."""
        chunk = ChatCompletionChunk.create_content(
            model="llama-3.1-8b", content="Hi"
        )
        sse = chunk.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")

    def test_custom_completion_id(self) -> None:
        """Test custom completion ID."""
        chunk = ChatCompletionChunk.create_start(
            model="llama-3.1-8b", completion_id="custom-id"
        )
        assert chunk.id == "custom-id"


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_create_error(self) -> None:
        """Test creating an error response."""
        error = ErrorResponse.create(
            message="Something went wrong",
            error_type="server_error",
        )
        assert error.error["message"] == "Something went wrong"
        assert error.error["type"] == "server_error"

    def test_create_error_with_code(self) -> None:
        """Test creating an error with code."""
        error = ErrorResponse.create(
            message="Rate limited",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
        )
        assert error.error["code"] == "rate_limit_exceeded"


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response(self) -> None:
        """Test creating a health response."""
        health = HealthResponse(
            status="healthy",
            instance_id="kvbench-0",
            server_type="combined",
            model="llama-3.1-8b",
            uptime_seconds=3600.0,
        )
        assert health.status == "healthy"
        assert health.uptime_seconds == 3600.0


class TestMetricsResponse:
    """Tests for MetricsResponse model."""

    def test_metrics_response(self) -> None:
        """Test creating a metrics response."""
        metrics = MetricsResponse(
            requests_total=100,
            requests_success=95,
            requests_failed=5,
            tokens_processed=10000,
            cache_hits=80,
            cache_misses=20,
            avg_latency_ms=50.5,
        )
        assert metrics.requests_total == 100
        assert metrics.tokens_processed == 10000


class TestModelList:
    """Tests for ModelList model."""

    def test_model_list(self) -> None:
        """Test creating a model list."""
        models = ModelList(
            data=[
                ModelInfo(id="llama-3.1-8b"),
                ModelInfo(id="llama-3.1-70b"),
            ]
        )
        assert len(models.data) == 2
        assert models.object == "list"
