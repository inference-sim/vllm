# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration test for observability_config initialization in serving classes.

This is a regression test for the bug where OpenAIServing.__init__() did not
initialize self.observability_config, causing AttributeError when journey
tracing code accessed self.observability_config.journey_tracing_sample_rate.

The bug manifested as:
  AttributeError: 'OpenAIServingCompletion' object has no attribute 'observability_config'

This test ensures all serving endpoints can be instantiated with journey
tracing enabled and can handle requests without AttributeError.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


@pytest.fixture
def mock_engine_client_with_journey_tracing():
    """Create mock engine client with proper vllm_config structure for journey tracing."""
    mock_client = AsyncMock()

    # Create proper nested config structure
    mock_observability_config = MagicMock()
    mock_observability_config.journey_tracing_sample_rate = 1.0
    mock_observability_config.enable_journey_tracing = True
    mock_observability_config.otlp_traces_endpoint = "http://localhost:4317"
    mock_observability_config.enable_logging_iteration_details = False
    mock_observability_config.show_hidden_metrics = False
    mock_observability_config.kv_cache_metrics = False
    mock_observability_config.cudagraph_metrics = False
    mock_observability_config.enable_mfu_metrics = False

    mock_vllm_config = MagicMock()
    mock_vllm_config.observability_config = mock_observability_config

    mock_model_config = MagicMock()
    mock_model_config.max_model_len = 4096
    mock_model_config.logits_processors = []
    mock_model_config.logits_processor_pattern = None
    mock_model_config.generation_config = "auto"

    mock_client.vllm_config = mock_vllm_config
    mock_client.model_config = mock_model_config
    mock_client.is_tracing_enabled = AsyncMock(return_value=True)
    mock_client.errored = False
    mock_client.dead_error = None

    return mock_client


@pytest.fixture
def mock_models():
    """Create mock OpenAIServingModels with all required attributes."""
    mock_models = MagicMock()

    # Model config
    mock_model_config = MagicMock()
    mock_model_config.max_model_len = 4096
    mock_model_config.logits_processors = []
    mock_model_config.logits_processor_pattern = None
    mock_model_config.generation_config = "auto"
    mock_model_config.get_diff_sampling_param = MagicMock(return_value=None)
    mock_models.model_config = mock_model_config

    # Other required attributes
    mock_models.input_processor = MagicMock()
    mock_models.io_processor = MagicMock()
    mock_models.renderer = MagicMock()
    mock_models.renderer.tokenizer = MagicMock()
    mock_models.model_name = MagicMock(return_value="test-model")

    return mock_models


@pytest.mark.asyncio
async def test_completion_serving_initializes_observability_config(
    mock_engine_client_with_journey_tracing,
    mock_models
):
    """
    Test that OpenAIServingCompletion properly initializes observability_config.

    This is a regression test for the bug where observability_config was not
    initialized in OpenAIServing.__init__(), causing AttributeError when
    _create_api_span() tried to access journey_tracing_sample_rate.
    """
    # Instantiate the serving class
    serving = OpenAIServingCompletion(
        engine_client=mock_engine_client_with_journey_tracing,
        models=mock_models,
        request_logger=None,
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
    )

    # CRITICAL ASSERTION: Verify observability_config was initialized
    assert hasattr(serving, 'observability_config'), \
        "OpenAIServingCompletion must have observability_config attribute"

    assert serving.observability_config is not None, \
        "observability_config must not be None"

    assert hasattr(serving.observability_config, 'journey_tracing_sample_rate'), \
        "observability_config must have journey_tracing_sample_rate attribute"

    assert serving.observability_config.journey_tracing_sample_rate == 1.0, \
        "journey_tracing_sample_rate should match the configured value"


@pytest.mark.asyncio
async def test_chat_serving_initializes_observability_config(
    mock_engine_client_with_journey_tracing,
    mock_models
):
    """
    Test that OpenAIServingChat properly initializes observability_config.

    Ensures the bug fix propagates to all serving subclasses via inheritance.
    """
    # Instantiate the serving class
    serving = OpenAIServingChat(
        engine_client=mock_engine_client_with_journey_tracing,
        models=mock_models,
        response_role="assistant",
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
    )

    # CRITICAL ASSERTION: Verify observability_config was initialized
    assert hasattr(serving, 'observability_config'), \
        "OpenAIServingChat must have observability_config attribute"

    assert serving.observability_config is not None, \
        "observability_config must not be None"

    assert hasattr(serving.observability_config, 'journey_tracing_sample_rate'), \
        "observability_config must have journey_tracing_sample_rate attribute"


@pytest.mark.asyncio
async def test_create_api_span_does_not_crash(
    mock_engine_client_with_journey_tracing,
    mock_models
):
    """
    Integration test: Verify _create_api_span() can be called without AttributeError.

    This tests the actual code path that was failing in production with:
      AttributeError: 'OpenAIServingCompletion' object has no attribute 'observability_config'
    """
    serving = OpenAIServingCompletion(
        engine_client=mock_engine_client_with_journey_tracing,
        models=mock_models,
        request_logger=None,
    )

    # Pre-populate the tracing enabled cache to avoid async engine call
    serving._cached_is_tracing_enabled = True

    # Mock the OpenTelemetry tracer
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span

    with patch("opentelemetry.trace.get_tracer_provider") as mock_provider:
        mock_provider.return_value.get_tracer.return_value = mock_tracer

        with patch("vllm.tracing.SpanAttributes") as mock_span_attrs:
            mock_span_attrs.GEN_AI_REQUEST_ID = "gen_ai.request.id"
            mock_span_attrs.EVENT_TS_MONOTONIC = "event.ts.monotonic"

            with patch("vllm.tracing.extract_trace_context", return_value=None):
                # This should NOT raise AttributeError about observability_config
                try:
                    span = await serving._create_api_span(
                        request_id="test-req-123",
                        trace_headers=None,
                        endpoint="/v1/completions"
                    )

                    # Verify span was created (with sample_rate=1.0, should always succeed)
                    assert span is not None, "Span should be created when sampled"
                    assert mock_tracer.start_span.called, "Tracer should create span"

                except AttributeError as e:
                    if 'observability_config' in str(e):
                        pytest.fail(
                            f"AttributeError about observability_config indicates the bug is not fixed: {e}"
                        )
                    else:
                        # Other AttributeErrors are not related to our bug fix
                        raise


@pytest.mark.asyncio
async def test_completion_request_with_journey_tracing_does_not_crash(
    mock_engine_client_with_journey_tracing,
    mock_models
):
    """
    End-to-end integration test: Make a completion request with journey tracing enabled.

    This ensures the entire request path works without AttributeError about
    observability_config. We don't care if the request fully succeeds (it won't
    in this mock environment), we only care that it doesn't crash with the
    specific AttributeError that the bug caused.
    """
    serving = OpenAIServingCompletion(
        engine_client=mock_engine_client_with_journey_tracing,
        models=mock_models,
        request_logger=None,
    )

    # Create a minimal completion request
    request = CompletionRequest(
        model="test-model",
        prompt="Once upon a time",
        max_tokens=10
    )

    # Mock the renderer and other dependencies
    mock_renderer = MagicMock()
    mock_renderer.render_prompt_and_embeds = AsyncMock(return_value=[
        {"prompt": "Once upon a time", "prompt_token_ids": [1, 2, 3]}
    ])
    serving.renderer = mock_renderer

    with patch.object(serving, '_check_model', return_value=None):
        with patch.object(serving, '_get_completion_renderer', return_value=mock_renderer):
            with patch.object(serving, '_build_render_config', return_value=MagicMock()):
                # Try to render the request
                # This exercises the code path that accesses observability_config
                try:
                    result = await serving.render_completion_request(request)

                    # If we get here, no AttributeError was raised
                    # The result should be a list of engine prompts
                    assert result is not None

                except AttributeError as e:
                    if 'observability_config' in str(e):
                        pytest.fail(
                            f"Request rendering failed with AttributeError about "
                            f"observability_config, indicating bug is not fixed: {e}"
                        )
                    # Other AttributeErrors are fine - we only care about observability_config
                    raise

                except Exception as e:
                    # Other exceptions are expected in mock environment
                    # We only care that it's not AttributeError about observability_config
                    assert 'observability_config' not in str(e), \
                        f"Unexpected error mentioning observability_config: {e}"


@pytest.mark.asyncio
async def test_all_serving_classes_inherit_observability_config():
    """
    Meta-test: Verify all serving endpoint classes inherit observability_config.

    This ensures that fixing the base class fixes all endpoints.
    """
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.pooling.embed.serving import EmbeddingMixin
    from vllm.entrypoints.pooling.classify.serving import ClassificationMixin
    from vllm.entrypoints.pooling.score.serving import ServingScores
    from vllm.entrypoints.pooling.pooling.serving import OpenAIServingPooling
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    # Verify all classes inherit from OpenAIServing
    serving_classes = [
        OpenAIServingCompletion,
        OpenAIServingChat,
        EmbeddingMixin,
        ClassificationMixin,
        ServingScores,
        OpenAIServingPooling,
    ]

    for cls in serving_classes:
        assert issubclass(cls, OpenAIServing), \
            f"{cls.__name__} must inherit from OpenAIServing to get observability_config"
