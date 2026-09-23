"""Chat endpoint for Pihu-BreakThough API tying Brain and Memory together.

Stage 4 connects the ProviderRouter with persistent conversation history
backed by Neon PostgreSQL via ConversationRepository.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional
import urllib.request
import uuid

from flask import Blueprint, jsonify, request, Response, stream_with_context
from werkzeug.exceptions import HTTPException

from pihu_core.audio import audio_transcriber, AudioTranscriptionError
from pihu_core.config import Config
from pihu_core.repository import conversation_repo
from pihu_core.router import router

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)

MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_ITEMS = 50
VALID_ROLES = {"user", "assistant", "system"}


def _error_response(code: str, message: str, status_code: int = 400) -> tuple[Any, int]:
    """Helper to return consistent, structured error JSON without stack traces."""
    return (
        jsonify(
            {
                "ok": False,
                "error": {
                    "code": code,
                    "message": message,
                },
            }
        ),
        status_code,
    )


@chat_bp.route("/chat", methods=["POST"])
def post_chat() -> tuple[Any, int]:
    """Process chat interactions with persistent conversation memory and multi-tier routing.

    Accepts:
        {
            "message": "User query string",
            "conversation_id": "optional-uuid-string",
            "history": [{"role": "user", "content": "..."}, ...],   (optional client history)
            "provider": "gemini"                                   (optional provider hint)
            "strategy": "auto"                                     (optional routing strategy)
        }

    Returns:
        JSON response with AI response, conversation_id, active provider, model, and metadata.
    """
    # 1. Content-Type and JSON validity verification
    if not request.is_json:
        return _error_response(
            code="INVALID_CONTENT_TYPE",
            message="Content-Type must be 'application/json'.",
            status_code=400,
        )

    try:
        data = request.get_json(silent=False)
    except HTTPException as http_err:
        if http_err.code == 413 or (http_err.code and http_err.code >= 500):
            raise
        return _error_response(
            code="INVALID_JSON",
            message="Request body could not be parsed as valid JSON.",
            status_code=400,
        )
    except Exception:
        return _error_response(
            code="INVALID_JSON",
            message="Request body could not be parsed as valid JSON.",
            status_code=400,
        )

    if data is None or not isinstance(data, dict):
        return _error_response(
            code="INVALID_JSON",
            message="Request body must be a JSON object.",
            status_code=400,
        )

    # 2. Optional 'audio' speech-to-text processing (Groq Whisper)
    has_audio = False
    transcribed_text: Optional[str] = None
    if "audio" in data and data["audio"] is not None:
        raw_audio = data["audio"]
        if not isinstance(raw_audio, dict):
            return _error_response(
                code="INVALID_AUDIO_TYPE",
                message="The 'audio' field must be an object with 'data' and optional 'mime_type'.",
                status_code=400,
            )

        if "data" not in raw_audio:
            return _error_response(
                code="MISSING_AUDIO_DATA",
                message="The 'audio' object must contain a 'data' field.",
                status_code=400,
            )

        audio_b64 = raw_audio["data"]
        if not isinstance(audio_b64, str):
            return _error_response(
                code="INVALID_AUDIO_DATA",
                message="The 'audio.data' field must be a base64-encoded string.",
                status_code=400,
            )

        audio_mime = raw_audio.get("mime_type", "audio/webm")
        if not isinstance(audio_mime, str):
            return _error_response(
                code="INVALID_AUDIO_MIME_TYPE",
                message="The 'audio.mime_type' field must be a string.",
                status_code=400,
            )

        try:
            transcribed_text = audio_transcriber.transcribe(
                audio_data=audio_b64,
                mime_type=audio_mime,
            )
            has_audio = True
        except AudioTranscriptionError as a_err:
            logger.warning("Audio transcription failed: %s", a_err)
            return _error_response(
                code="AUDIO_TRANSCRIPTION_FAILED",
                message=f"Voice transcription failed: {a_err}",
                status_code=400,
            )
        except Exception as exc:
            logger.error("Unexpected audio transcription error: %s", exc)
            return _error_response(
                code="AUDIO_TRANSCRIPTION_FAILED",
                message=f"Voice transcription failed unexpectedly: {exc}",
                status_code=400,
            )

    # 3. 'message' field resolution & validation
    raw_message = data.get("message")
    if raw_message is not None and not isinstance(raw_message, str):
        return _error_response(
            code="INVALID_MESSAGE_TYPE",
            message="The 'message' field must be a string.",
            status_code=400,
        )

    clean_message = raw_message.strip() if isinstance(raw_message, str) else ""

    # Inject or combine transcribed audio text into message
    if has_audio and transcribed_text:
        if clean_message:
            clean_message = f"{clean_message} {transcribed_text}".strip()
        else:
            clean_message = transcribed_text

    if not clean_message:
        if "message" not in data and not has_audio:
            return _error_response(
                code="MISSING_MESSAGE",
                message="The 'message' field is required.",
                status_code=400,
            )
        return _error_response(
            code="EMPTY_MESSAGE",
            message="The 'message' field cannot be empty or contain only whitespace.",
            status_code=400,
        )

    if len(clean_message) > MAX_MESSAGE_LENGTH:
        return _error_response(
            code="MESSAGE_TOO_LONG",
            message=f"Message exceeds maximum allowed length of {MAX_MESSAGE_LENGTH} characters.",
            status_code=400,
        )

    # 4. 'conversation_id' validation (optional)
    conversation_id = data.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        return _error_response(
            code="INVALID_CONVERSATION_ID",
            message="The 'conversation_id' field must be a string.",
            status_code=400,
        )

    # 4. 'history' validation (optional client-side history)
    client_history: Optional[List[Dict[str, str]]] = None
    if "history" in data and data["history"] is not None:
        raw_history = data["history"]
        if not isinstance(raw_history, list):
            return _error_response(
                code="INVALID_HISTORY",
                message="The 'history' field must be an array of message objects.",
                status_code=400,
            )

        if len(raw_history) > MAX_HISTORY_ITEMS:
            return _error_response(
                code="HISTORY_TOO_LONG",
                message=f"The 'history' field exceeds maximum allowed length of {MAX_HISTORY_ITEMS} items.",
                status_code=400,
            )

        sanitized_history: List[Dict[str, str]] = []
        for index, item in enumerate(raw_history):
            if not isinstance(item, dict):
                return _error_response(
                    code="INVALID_HISTORY_ITEM",
                    message=f"History item at index {index} must be an object with 'role' and 'content'.",
                    status_code=400,
                )

            role = item.get("role")
            content = item.get("content")

            if not isinstance(role, str) or role.lower() not in VALID_ROLES:
                return _error_response(
                    code="INVALID_HISTORY_ROLE",
                    message=f"History item at index {index} has invalid 'role'. Allowed values: {sorted(list(VALID_ROLES))}.",
                    status_code=400,
                )

            if not isinstance(content, str):
                return _error_response(
                    code="INVALID_HISTORY_CONTENT",
                    message=f"History item at index {index} must have a string 'content' field.",
                    status_code=400,
                )

            sanitized_history.append({"role": role.lower(), "content": content})

        client_history = sanitized_history

    # 5. 'images' validation (optional multimodal vision payloads)
    validated_images: Optional[List[Dict[str, Any]]] = None
    if "images" in data and data["images"] is not None:
        raw_images = data["images"]
        if not isinstance(raw_images, list):
            return _error_response(
                code="INVALID_IMAGES_TYPE",
                message="The 'images' field must be an array of image objects.",
                status_code=400,
            )

        if len(raw_images) > 3:
            return _error_response(
                code="TOO_MANY_IMAGES",
                message="Maximum of 3 images allowed per request.",
                status_code=400,
            )

        SUPPORTED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB per image

        processed_images: List[Dict[str, Any]] = []

        for idx, img_item in enumerate(raw_images):
            if not isinstance(img_item, dict):
                return _error_response(
                    code="INVALID_IMAGE_ITEM",
                    message=f"Image item at index {idx} must be an object.",
                    status_code=400,
                )

            b64_str: Optional[str] = None
            mime_type: Optional[str] = img_item.get("mime_type")

            # Check for URL format (data URI or remote URL)
            if "url" in img_item and img_item["url"] is not None:
                url_val = img_item["url"]
                if not isinstance(url_val, str):
                    return _error_response(
                        code="INVALID_IMAGE_URL",
                        message=f"Image 'url' at index {idx} must be a string.",
                        status_code=400,
                    )
                url_clean = url_val.strip()
                if url_clean.startswith("data:"):
                    try:
                        header, b64_part = url_clean.split(",", 1)
                        b64_str = b64_part
                        if ";base64" in header:
                            extracted_mime = header.split(":", 1)[1].split(";", 1)[0]
                            if not mime_type:
                                mime_type = extracted_mime
                    except Exception:
                        return _error_response(
                            code="INVALID_DATA_URI",
                            message=f"Image at index {idx} contains a malformed data URI.",
                            status_code=400,
                        )
                elif url_clean.startswith("http://") or url_clean.startswith("https://"):
                    try:
                        req = urllib.request.Request(
                            url_clean,
                            headers={"User-Agent": "Pihu-BreakThough/1.0"},
                        )
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            fetched_bytes = resp.read(MAX_IMAGE_BYTES + 1)
                            if len(fetched_bytes) > MAX_IMAGE_BYTES:
                                return _error_response(
                                    code="IMAGE_TOO_LARGE",
                                    message=f"Image fetched from URL at index {idx} exceeds 5MB limit.",
                                    status_code=400,
                                )
                            b64_str = base64.b64encode(fetched_bytes).decode("utf-8")
                            content_type = resp.headers.get_content_type()
                            if not mime_type and content_type:
                                mime_type = content_type
                    except Exception as url_err:
                        return _error_response(
                            code="IMAGE_FETCH_FAILED",
                            message=f"Failed to fetch image from URL at index {idx}: {url_err}",
                            status_code=400,
                        )
                else:
                    return _error_response(
                        code="INVALID_IMAGE_URL",
                        message=f"Image URL at index {idx} must be a valid http(s) URL or data URI.",
                        status_code=400,
                    )
            elif "data" in img_item:
                raw_data = img_item["data"]
                if not isinstance(raw_data, str):
                    return _error_response(
                        code="INVALID_IMAGE_DATA",
                        message=f"Image 'data' at index {idx} must be a base64-encoded string.",
                        status_code=400,
                    )
                clean_data = raw_data.strip()
                if clean_data.startswith("data:"):
                    try:
                        header, b64_part = clean_data.split(",", 1)
                        b64_str = b64_part
                        if ";base64" in header:
                            extracted_mime = header.split(":", 1)[1].split(";", 1)[0]
                            if not mime_type:
                                mime_type = extracted_mime
                    except Exception:
                        return _error_response(
                            code="INVALID_DATA_URI",
                            message=f"Image at index {idx} contains a malformed data URI.",
                            status_code=400,
                        )
                else:
                    b64_str = clean_data
            else:
                return _error_response(
                    code="MISSING_IMAGE_PAYLOAD",
                    message=f"Image item at index {idx} must provide 'data' or 'url'.",
                    status_code=400,
                )

            if not b64_str:
                return _error_response(
                    code="EMPTY_IMAGE_DATA",
                    message=f"Image data at index {idx} cannot be empty.",
                    status_code=400,
                )

            if not mime_type or not isinstance(mime_type, str):
                return _error_response(
                    code="MISSING_IMAGE_MIME_TYPE",
                    message=f"Image item at index {idx} must have a valid 'mime_type'.",
                    status_code=400,
                )

            clean_mime = mime_type.strip().lower()
            if clean_mime not in SUPPORTED_IMAGE_MIMES:
                return _error_response(
                    code="UNSUPPORTED_IMAGE_MIME_TYPE",
                    message=f"Unsupported image MIME type '{clean_mime}'. Supported formats: {sorted(list(SUPPORTED_IMAGE_MIMES))}.",
                    status_code=400,
                )

            # Validate base64 encoding and size limit
            try:
                decoded_bytes = base64.b64decode(b64_str, validate=True)
            except Exception:
                return _error_response(
                    code="INVALID_BASE64_IMAGE",
                    message=f"Image data at index {idx} is not valid base64.",
                    status_code=400,
                )

            if len(decoded_bytes) == 0:
                return _error_response(
                    code="EMPTY_IMAGE_DATA",
                    message=f"Decoded image data at index {idx} is empty.",
                    status_code=400,
                )

            if len(decoded_bytes) > MAX_IMAGE_BYTES:
                return _error_response(
                    code="IMAGE_TOO_LARGE",
                    message=f"Image at index {idx} exceeds maximum allowed size of 5MB.",
                    status_code=400,
                )

            processed_images.append({
                "data": b64_str,
                "mime_type": clean_mime,
            })

        if processed_images:
            validated_images = processed_images

    # 6. Optional parameters (provider hint, strategy)
    provider_name: Optional[str] = None
    if "provider" in data and data["provider"] is not None:
        if not isinstance(data["provider"], str):
            return _error_response(
                code="INVALID_PROVIDER_TYPE",
                message="The 'provider' field must be a string.",
                status_code=400,
            )
        provider_name = data["provider"].strip()

    strategy: str = "auto"
    if "strategy" in data and data["strategy"] is not None:
        if not isinstance(data["strategy"], str):
            return _error_response(
                code="INVALID_STRATEGY_TYPE",
                message="The 'strategy' field must be a string.",
                status_code=400,
            )
        strategy = data["strategy"].strip()

    stream_requested = False
    if "stream" in data and data["stream"] is not None:
        if not isinstance(data["stream"], bool):
            return _error_response(
                code="INVALID_STREAM_TYPE",
                message="The 'stream' field must be a boolean.",
                status_code=400,
            )
        stream_requested = data["stream"]

    # 6. Conversation & Memory Resolution
    effective_history: List[Dict[str, str]] = []
    user_id: Optional[str] = None
    has_db = conversation_repo.is_available()
    system_prompt: Optional[str] = None

    if has_db:
        try:
            # Resolve or create conversation
            if conversation_id:
                conv = conversation_repo.get_conversation(conversation_id)
                if not conv:
                    conv = conversation_repo.create_conversation(title=clean_message[:50])
                    conversation_id = conv["id"]
                    user_id = conv.get("user_id")
                else:
                    user_id = conv.get("user_id")
            else:
                conv = conversation_repo.create_conversation(title=clean_message[:50])
                conversation_id = conv["id"]
                user_id = conv.get("user_id")

            # Load persistent history from database
            db_messages = conversation_repo.get_messages(conversation_id, limit=MAX_HISTORY_ITEMS)
            for m in db_messages:
                effective_history.append({"role": m["role"], "content": m["content"]})

            # Fetch existing long-term user memories
            user_memories = conversation_repo.get_memories(user_id=user_id)
            if user_memories:
                memory_lines = [
                    f"- [{m.get('category', 'general').capitalize()}] {m.get('content')}"
                    for m in user_memories
                    if m.get("content")
                ]
                if memory_lines:
                    memories_text = "\n".join(memory_lines)
                    system_prompt = (
                        "You are Pihu, an intelligent, helpful, and empathetic AI assistant.\n\n"
                        "User Context/Memories:\n"
                        f"{memories_text}\n\n"
                        "Instructions:\n"
                        "- Use the above user context to provide personalized, relevant, and context-aware responses.\n"
                        "- If the user shares new important facts, personal preferences, or project details, use your 'save_core_memory' tool to persist them."
                    )

            # Save incoming User message to database
            user_db_content = clean_message
            if has_audio:
                user_db_content = f"[🎤 Voice Message] {user_db_content}"
            if validated_images:
                num_imgs = len(validated_images)
                tag = f"[Attached {num_imgs} Image{'s' if num_imgs > 1 else ''}]"
                user_db_content = f"{user_db_content} {tag}"

            conversation_repo.save_message(
                conversation_id=conversation_id,
                role="user",
                content=user_db_content,
            )
        except Exception as exc:
            logger.warning("Database conversation resolution failed: %s. Continuing with client memory.", exc)
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            effective_history = client_history or []
    else:
        # Fallback when database is unconfigured (preserves zero-dependency standalone execution)
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        effective_history = client_history or []

    if not system_prompt:
        system_prompt = (
            "You are Pihu, an intelligent, helpful, and empathetic AI assistant. "
            "If the user shares personal facts, preferences, or important project details, "
            "use your 'save_core_memory' tool to remember them for future conversations."
        )

    # 7. Execute Real-Time Streaming if requested
    if stream_requested:
        @stream_with_context
        def generate_sse():
            accumulated_chunks: List[str] = []
            final_provider = provider_name or "gemini"

            try:
                for chunk_event in router.route_stream(
                    message=clean_message,
                    history=effective_history,
                    strategy=strategy,
                    provider_name=provider_name,
                    system_prompt=system_prompt,
                    images=validated_images,
                ):
                    chunk_text = chunk_event.get("text", "")
                    chunk_prov = chunk_event.get("provider", final_provider)
                    final_provider = chunk_prov
                    if chunk_text:
                        accumulated_chunks.append(chunk_text)

                    event_payload = {
                        "chunk": chunk_text,
                        "provider": chunk_prov,
                    }
                    yield f"data: {json.dumps(event_payload)}\n\n"

            except Exception as exc:
                logger.exception("Error during SSE streaming: %s", exc)
                error_payload = {
                    "chunk": f"\n[Streaming error: {exc}]",
                    "provider": final_provider,
                    "error": True,
                }
                yield f"data: {json.dumps(error_payload)}\n\n"

            finally:
                full_text = "".join(accumulated_chunks)
                if has_db and conversation_id and full_text:
                    try:
                        conversation_repo.save_message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=full_text,
                            provider_used=final_provider,
                        )
                        logger.info(
                            "Persisted full streamed assistant message (%d chars) for conversation '%s'.",
                            len(full_text),
                            conversation_id,
                        )
                    except Exception as save_err:
                        logger.warning("Failed to persist streamed assistant response: %s", save_err)

        return Response(generate_sse(), mimetype="text/event-stream")

    # 8. Execute Brain Routing (Synchronous Atomic Flow)
    try:
        result = router.route_chat(
            message=clean_message,
            history=effective_history,
            strategy=strategy,
            provider_name=provider_name,
            system_prompt=system_prompt,
            images=validated_images,
        )

        ai_response = result["response"]
        active_provider = result["provider"]

        if has_audio:
            result["metadata"]["audio_transcribed"] = True
            result["metadata"]["transcription"] = transcribed_text

        # Save AI Response to database if available
        if has_db and conversation_id:
            try:
                conversation_repo.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=ai_response,
                    provider_used=active_provider,
                )
            except Exception as exc:
                logger.warning("Failed to persist assistant response to database: %s", exc)

        response_payload = {
            "ok": True,
            "app": Config.APP_NAME,
            "conversation_id": conversation_id,
            "answer": ai_response,
            "response": ai_response,
            "provider": active_provider,
            "model": result["model"],
            "metadata": result["metadata"],
        }
        return jsonify(response_payload), 200

    except Exception as exc:
        logger.exception("Error processing chat interaction: %s", exc)
        return _error_response(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred while processing the chat request.",
            status_code=500,
        )
