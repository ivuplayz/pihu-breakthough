"""Abstract and concrete LLM provider interfaces for Pihu-BreakThough.

Stage 3 introduces production-ready provider abstractions, capability matrices,
stream chunk specifications, and intent/strategy taxonomies.
Actual network execution to external vendor APIs is reserved for Stage 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import math
from typing import Any, Dict, Generator, List, Optional

from pihu_core.config import Config

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Classified user intent determining optimal provider routing."""

    FAST_CHAT = "fast_chat"
    COMPLEX_REASONING = "complex_reasoning"
    CODING_SYSTEM = "coding_system"
    SUMMARIZATION = "summarization"
    LOCAL_PRIVATE = "local_private"
    FALLBACK_SAFE = "fallback_safe"


class RoutingStrategy(str, Enum):
    """Strategic routing behavior configured per request or globally."""

    AUTO = "auto"
    LOW_LATENCY = "low_latency"
    HIGH_QUALITY = "high_quality"
    SUMMARIZATION = "summarization"
    OFFLINE_ONLY = "offline_only"
    MANUAL = "manual"


class ProviderStatus(str, Enum):
    """Operational health status of an AI provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    TRIPPED = "tripped"
    UNAVAILABLE = "unavailable"


@dataclass
class ProviderCapabilities:
    """Declared capabilities and performance characteristics of a model provider."""

    supports_streaming: bool = True
    supports_tools: bool = False
    max_context_tokens: int = 8192
    typical_latency_ms: int = 500
    is_local: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability set to dictionary format."""
        return {
            "supports_streaming": self.supports_streaming,
            "supports_tools": self.supports_tools,
            "max_context_tokens": self.max_context_tokens,
            "typical_latency_ms": self.typical_latency_ms,
            "is_local": self.is_local,
        }


@dataclass
class ProviderResponse:
    """Standardized response from an LLM provider."""

    content: str
    provider: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "metadata": self.metadata,
        }


@dataclass
class StreamChunk:
    """Standardized streaming chunk emitted during token generation."""

    text: str
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "text": self.text,
            "is_final": self.is_final,
            "metadata": self.metadata,
        }


class BaseProvider(ABC):
    """Abstract base class for all LLM providers in Pihu-BreakThough."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Model capabilities and limits."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider credentials/endpoints are available in environment."""
        pass

    def health_check(self) -> bool:
        """Perform a quick active health probe. Default returns availability status."""
        return self.is_available()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string using standard ~4 chars/token heuristic."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    @abstractmethod
    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a complete synchronous response."""
        pass

    def stream_generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        """Generate response tokens as a stream.

        Default implementation falls back to emitting full generate() content as a single chunk.
        """
        resp = self.generate(
            message=message,
            history=history,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs,
        )
        yield StreamChunk(
            text=resp.content,
            is_final=True,
            metadata=resp.metadata,
        )


class Stage1DeterministicProvider(BaseProvider):
    """Deterministic local provider for core foundation and zero-failure safety net."""

    @property
    def name(self) -> str:
        return "stage1-deterministic"

    @property
    def model_name(self) -> str:
        return "core-deterministic-v1"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=False,
            max_context_tokens=16384,
            typical_latency_ms=5,
            is_local=True,
        )

    def is_available(self) -> bool:
        return True

    def health_check(self) -> bool:
        return True

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        history_len = len(history) if history else 0
        timestamp = datetime.now(timezone.utc).isoformat()

        reply = (
            f"Pihu-BreakThough Stage 1 Core is operational. "
            f"Received message ({len(message)} chars): \"{message.strip()}\". "
            f"History context contains {history_len} items."
        )

        return ProviderResponse(
            content=reply,
            provider=self.name,
            model=self.model_name,
            metadata={
                "received_chars": len(message),
                "history_length": history_len,
                "timestamp": timestamp,
                "stage": Config.STAGE,
            },
        )

    def stream_generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        resp = self.generate(
            message=message,
            history=history,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs,
        )
        words = resp.content.split(" ")
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            chunk_text = word if is_last else word + " "
            yield StreamChunk(
                text=chunk_text,
                is_final=is_last,
                metadata=resp.metadata if is_last else {},
            )


# Safe imports for AI vendor SDKs
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    GENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None  # type: ignore
    GROQ_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    InferenceClient = None  # type: ignore
    HF_AVAILABLE = False


class GeminiProvider(BaseProvider):
    """Google Gemini provider (Tier 1 Primary Frontier Engine) using official google-genai SDK."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            max_context_tokens=1048576,
            typical_latency_ms=450,
            is_local=False,
        )

    def is_available(self) -> bool:
        """Check presence of GEMINI_API_KEY and SDK availability."""
        return bool(Config.GEMINI_API_KEY and Config.GEMINI_API_KEY.strip() and GENAI_AVAILABLE)

    def _build_contents(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Map standard chat history to Gemini contents structure."""
        contents: List[Dict[str, Any]] = []
        if history:
            for item in history:
                # Gemini roles are 'user' and 'model'
                role = "model" if item.get("role") in ("assistant", "model") else "user"
                content_text = item.get("content", "")
                contents.append({"role": role, "parts": [{"text": content_text}]})

        contents.append({"role": "user", "parts": [{"text": message}]})
        return contents

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response via Google Gemini API using google-genai SDK."""
        if not self.is_available():
            raise RuntimeError("GEMINI_API_KEY is not configured or google-genai SDK is unavailable.")

        try:
            client = genai.Client(api_key=Config.GEMINI_API_KEY)
            contents = self._build_contents(message=message, history=history)
            timeout = kwargs.get("timeout", 30)

            active_tools = tools
            if active_tools is None and "tools" in kwargs:
                active_tools = kwargs["tools"]
            if active_tools is None:
                from pihu_core.tools import tool_registry
                active_tools = tool_registry.to_gemini_format()

            config = None
            config_kwargs: Dict[str, Any] = {}
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            if timeout and types:
                config_kwargs["http_options"] = types.HttpOptions(timeout=timeout)
            if active_tools and types:
                config_kwargs["tools"] = active_tools
            if config_kwargs and types:
                config = types.GenerateContentConfig(**config_kwargs)

            call_kwargs: Dict[str, Any] = {"model": self.model_name, "contents": contents}
            if config:
                call_kwargs["config"] = config

            response = client.models.generate_content(**call_kwargs)

            # Check if LLM requested tool calling
            function_calls = getattr(response, "function_calls", None)
            if isinstance(function_calls, (list, tuple)) and len(function_calls) > 0:
                from pihu_core.tools import tool_registry

                if hasattr(response, "candidates") and response.candidates and hasattr(response.candidates[0], "content") and response.candidates[0].content:
                    contents.append(response.candidates[0].content)

                for fc in function_calls:
                    fc_name = getattr(fc, "name", "")
                    fc_args = getattr(fc, "args", {}) or {}
                    if not isinstance(fc_args, dict):
                        fc_args = {}
                    tool_result = tool_registry.execute(fc_name, **fc_args)

                    if types:
                        contents.append(
                            types.Content(
                                role="tool",
                                parts=[
                                    types.Part.from_function_response(
                                        name=fc_name,
                                        response={"result": str(tool_result)},
                                    )
                                ],
                            )
                        )
                    else:
                        contents.append({
                            "role": "tool",
                            "parts": [{"function_response": {"name": fc_name, "response": {"result": str(tool_result)}}}],
                        })

                # Second request to LLM grounded in tool result
                response = client.models.generate_content(**call_kwargs)

            text_output = response.text if response and hasattr(response, "text") else ""
            timestamp = datetime.now(timezone.utc).isoformat()

            return ProviderResponse(
                content=text_output,
                provider=self.name,
                model=self.model_name,
                metadata={
                    "stage": Config.STAGE,
                    "timestamp": timestamp,
                    "provider": self.name,
                    "model": self.model_name,
                },
            )
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    def stream_generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        """Stream response tokens via Google Gemini API using google-genai SDK."""
        if not self.is_available():
            raise RuntimeError("GEMINI_API_KEY is not configured or google-genai SDK is unavailable.")

        try:
            client = genai.Client(api_key=Config.GEMINI_API_KEY)
            contents = self._build_contents(message=message, history=history)
            timeout = kwargs.get("timeout", 30)

            active_tools = tools
            if active_tools is None and "tools" in kwargs:
                active_tools = kwargs["tools"]
            if active_tools is None:
                from pihu_core.tools import tool_registry
                active_tools = tool_registry.to_gemini_format()

            config = None
            config_kwargs: Dict[str, Any] = {}
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            if timeout and types:
                config_kwargs["http_options"] = types.HttpOptions(timeout=timeout)
            if active_tools and types:
                config_kwargs["tools"] = active_tools
            if config_kwargs and types:
                config = types.GenerateContentConfig(**config_kwargs)

            call_kwargs: Dict[str, Any] = {"model": self.model_name, "contents": contents}
            if config:
                call_kwargs["config"] = config

            if active_tools:
                probe_resp = client.models.generate_content(**call_kwargs)
                function_calls = getattr(probe_resp, "function_calls", None)
                if isinstance(function_calls, (list, tuple)) and len(function_calls) > 0:
                    from pihu_core.tools import tool_registry

                    if hasattr(probe_resp, "candidates") and probe_resp.candidates and hasattr(probe_resp.candidates[0], "content") and probe_resp.candidates[0].content:
                        contents.append(probe_resp.candidates[0].content)

                    for fc in function_calls:
                        fc_name = getattr(fc, "name", "")
                        fc_args = getattr(fc, "args", {}) or {}
                        if not isinstance(fc_args, dict):
                            fc_args = {}
                        tool_result = tool_registry.execute(fc_name, **fc_args)

                        if types:
                            contents.append(
                                types.Content(
                                    role="tool",
                                    parts=[
                                        types.Part.from_function_response(
                                            name=fc_name,
                                            response={"result": str(tool_result)},
                                        )
                                    ],
                                )
                            )
                        else:
                            contents.append({
                                "role": "tool",
                                "parts": [{"function_response": {"name": fc_name, "response": {"result": str(tool_result)}}}],
                            })

                    response_stream = client.models.generate_content_stream(**call_kwargs)
                    for chunk in response_stream:
                        chunk_text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
                        if chunk_text:
                            yield StreamChunk(text=chunk_text, is_final=False)
                    yield StreamChunk(text="", is_final=True)
                    return

            response_stream = client.models.generate_content_stream(**call_kwargs)

            for chunk in response_stream:
                chunk_text = chunk.text if hasattr(chunk, "text") else ""
                if chunk_text:
                    yield StreamChunk(text=chunk_text, is_final=False)

            yield StreamChunk(text="", is_final=True)
        except Exception as exc:
            logger.error("Gemini stream failed: %s", exc)
            raise RuntimeError(f"Gemini streaming error: {exc}") from exc


class GroqProvider(BaseProvider):
    """Groq provider (Tier 2 Ultra-Low-Latency Inference Engine) using official groq SDK."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile") -> None:
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            max_context_tokens=32768,
            typical_latency_ms=180,
            is_local=False,
        )

    def is_available(self) -> bool:
        """Check presence of GROQ_API_KEY and SDK availability."""
        return bool(Config.GROQ_API_KEY and Config.GROQ_API_KEY.strip() and GROQ_AVAILABLE)

    def _build_messages(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Map chat history and system prompt to OpenAI/Groq messages format."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for item in history:
                role = item.get("role", "user")
                if role not in ("system", "user", "assistant"):
                    role = "user"
                messages.append({"role": role, "content": item.get("content", "")})

        messages.append({"role": "user", "content": message})
        return messages

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response via Groq API with tool calling support."""
        if not self.is_available():
            raise RuntimeError("GROQ_API_KEY is not configured or groq SDK is unavailable.")

        try:
            client = Groq(api_key=Config.GROQ_API_KEY, timeout=kwargs.get("timeout", 30))
            messages = self._build_messages(message=message, history=history, system_prompt=system_prompt)
            temperature = kwargs.get("temperature", 0.7)

            active_tools = tools
            if active_tools is None and "tools" in kwargs:
                active_tools = kwargs["tools"]
            if active_tools is None:
                from pihu_core.tools import tool_registry
                active_tools = tool_registry.to_openai_format()

            call_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if active_tools:
                call_kwargs["tools"] = active_tools

            completion = client.chat.completions.create(**call_kwargs)
            message_choice = completion.choices[0].message
            tool_calls = getattr(message_choice, "tool_calls", None)

            if isinstance(tool_calls, (list, tuple)) and len(tool_calls) > 0:
                from pihu_core.tools import tool_registry

                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": getattr(tc, "id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": getattr(getattr(tc, "function", None), "name", ""),
                                "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ],
                })

                for tc in tool_calls:
                    tc_id = getattr(tc, "id", "call_1")
                    tc_func = getattr(tc, "function", None)
                    tc_name = getattr(tc_func, "name", "") if tc_func else ""
                    raw_args = getattr(tc_func, "arguments", "{}") if tc_func else "{}"
                    if isinstance(raw_args, str):
                        try:
                            args_dict = json.loads(raw_args)
                        except Exception:
                            args_dict = {}
                    elif isinstance(raw_args, dict):
                        args_dict = raw_args
                    else:
                        args_dict = {}

                    tool_result = tool_registry.execute(tc_name, **args_dict)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": str(tool_result),
                    })

                call_kwargs["messages"] = messages
                completion = client.chat.completions.create(**call_kwargs)

            reply_text = completion.choices[0].message.content or ""
            timestamp = datetime.now(timezone.utc).isoformat()

            usage_meta: Dict[str, Any] = {}
            if hasattr(completion, "usage") and completion.usage:
                usage_meta = {
                    "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                    "total_tokens": getattr(completion.usage, "total_tokens", None),
                }

            return ProviderResponse(
                content=reply_text,
                provider=self.name,
                model=self.model_name,
                metadata={
                    "stage": Config.STAGE,
                    "timestamp": timestamp,
                    "usage": usage_meta,
                    "provider": self.name,
                    "model": self.model_name,
                },
            )
        except Exception as exc:
            logger.error("Groq API call failed: %s", exc)
            raise RuntimeError(f"Groq API error: {exc}") from exc

    def stream_generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        """Stream response tokens via Groq API with tool calling support."""
        if not self.is_available():
            raise RuntimeError("GROQ_API_KEY is not configured or groq SDK is unavailable.")

        try:
            client = Groq(api_key=Config.GROQ_API_KEY, timeout=kwargs.get("timeout", 30))
            messages = self._build_messages(message=message, history=history, system_prompt=system_prompt)
            temperature = kwargs.get("temperature", 0.7)

            active_tools = tools
            if active_tools is None and "tools" in kwargs:
                active_tools = kwargs["tools"]
            if active_tools is None:
                from pihu_core.tools import tool_registry
                active_tools = tool_registry.to_openai_format()

            if active_tools:
                probe_completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    tools=active_tools,
                )
                choices = getattr(probe_completion, "choices", None)
                if choices and len(choices) > 0:
                    msg_choice = choices[0].message
                    tool_calls = getattr(msg_choice, "tool_calls", None)

                    if isinstance(tool_calls, (list, tuple)) and len(tool_calls) > 0:
                        from pihu_core.tools import tool_registry

                        messages.append({
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": getattr(tc, "id", f"call_{i}"),
                                    "type": "function",
                                    "function": {
                                        "name": getattr(getattr(tc, "function", None), "name", ""),
                                        "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                                    },
                                }
                                for i, tc in enumerate(tool_calls)
                            ],
                        })

                        for tc in tool_calls:
                            tc_id = getattr(tc, "id", "call_1")
                            tc_func = getattr(tc, "function", None)
                            tc_name = getattr(tc_func, "name", "") if tc_func else ""
                            raw_args = getattr(tc_func, "arguments", "{}") if tc_func else "{}"
                            if isinstance(raw_args, str):
                                try:
                                    args_dict = json.loads(raw_args)
                                except Exception:
                                    args_dict = {}
                            elif isinstance(raw_args, dict):
                                args_dict = raw_args
                            else:
                                args_dict = {}

                            tool_result = tool_registry.execute(tc_name, **args_dict)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": str(tool_result),
                            })

                        stream = client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=temperature,
                            stream=True,
                        )
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content if chunk.choices and hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content") else ""
                            if delta:
                                yield StreamChunk(text=delta, is_final=False)

                        yield StreamChunk(text="", is_final=True)
                        return
                    else:
                        content = getattr(msg_choice, "content", "") or ""
                        if content:
                            yield StreamChunk(text=content, is_final=False)
                            yield StreamChunk(text="", is_final=True)
                            return
                elif hasattr(probe_completion, "__iter__") and not isinstance(probe_completion, (str, bytes, dict)):
                    for chunk in probe_completion:
                        delta = chunk.choices[0].delta.content if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content") else ""
                        if delta:
                            yield StreamChunk(text=delta, is_final=False)

                    yield StreamChunk(text="", is_final=True)
                    return

            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices and hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content") else ""
                if delta:
                    yield StreamChunk(text=delta, is_final=False)

            yield StreamChunk(text="", is_final=True)
        except Exception as exc:
            logger.error("Groq stream failed: %s", exc)
            raise RuntimeError(f"Groq streaming error: {exc}") from exc



class HuggingFaceProvider(BaseProvider):
    """Hugging Face Serverless Inference provider (Tier 3 Specialized Open-Weights Engine).

    Positioned as Tier 3 fallback and specialized for summarization tasks.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or Config.HF_MODEL or "meta-llama/Llama-3.2-3B-Instruct"

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=False,
            max_context_tokens=32768,
            typical_latency_ms=650,
            is_local=False,
        )

    def is_available(self) -> bool:
        """Check presence of HF_API_KEY and huggingface_hub SDK availability."""
        return bool(Config.HF_API_KEY and Config.HF_API_KEY.strip() and HF_AVAILABLE)

    def _build_messages(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Map chat history and system prompt to OpenAI/HF messages format."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for item in history:
                role = item.get("role", "user")
                if role not in ("system", "user", "assistant"):
                    role = "user"
                messages.append({"role": role, "content": item.get("content", "")})

        messages.append({"role": "user", "content": message})
        return messages

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response via Hugging Face Serverless Inference API."""
        if not self.is_available():
            raise RuntimeError("HF_API_KEY is not configured or huggingface_hub SDK is unavailable.")

        try:
            client = InferenceClient(token=Config.HF_API_KEY, timeout=kwargs.get("timeout", 30))
            messages = self._build_messages(message=message, history=history, system_prompt=system_prompt)
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            reply_text = completion.choices[0].message.content or ""
            timestamp = datetime.now(timezone.utc).isoformat()

            usage_meta: Dict[str, Any] = {}
            if hasattr(completion, "usage") and completion.usage:
                usage_meta = {
                    "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                    "total_tokens": getattr(completion.usage, "total_tokens", None),
                }

            return ProviderResponse(
                content=reply_text,
                provider=self.name,
                model=self.model_name,
                metadata={
                    "stage": Config.STAGE,
                    "timestamp": timestamp,
                    "usage": usage_meta,
                    "provider": self.name,
                    "model": self.model_name,
                },
            )
        except Exception as exc:
            sanitized_err = str(exc).replace(Config.HF_API_KEY or "", "***") if Config.HF_API_KEY else str(exc)
            logger.error("HuggingFace API call failed: %s", sanitized_err)
            raise RuntimeError(f"HuggingFace API error: {sanitized_err}") from exc

    def stream_generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        """Stream response tokens via Hugging Face Serverless Inference API."""
        if not self.is_available():
            raise RuntimeError("HF_API_KEY is not configured or huggingface_hub SDK is unavailable.")

        try:
            client = InferenceClient(token=Config.HF_API_KEY, timeout=kwargs.get("timeout", 30))
            messages = self._build_messages(message=message, history=history, system_prompt=system_prompt)
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)

            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                delta = ""
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content") and choice.delta.content:
                        delta = choice.delta.content
                if delta:
                    yield StreamChunk(text=delta, is_final=False)

            yield StreamChunk(text="", is_final=True)
        except Exception as exc:
            sanitized_err = str(exc).replace(Config.HF_API_KEY or "", "***") if Config.HF_API_KEY else str(exc)
            logger.error("HuggingFace stream failed: %s", sanitized_err)
            raise RuntimeError(f"HuggingFace streaming error: {sanitized_err}") from exc


class OllamaProvider(BaseProvider):
    """Local Ollama provider (Tier 4 Private & Offline Local Engine)."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return "llama3"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=False,
            max_context_tokens=8192,
            typical_latency_ms=600,
            is_local=True,
        )

    def is_available(self) -> bool:
        return bool(Config.OLLAMA_BASE_URL and Config.OLLAMA_BASE_URL.strip())

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        if not self.is_available():
            raise RuntimeError("OLLAMA_BASE_URL is not configured.")

        raise NotImplementedError(
            "Ollama local API client execution is scheduled for Stage 4."
        )
