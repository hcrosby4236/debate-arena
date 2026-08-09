"""
Backboard API Python client (async only)
"""

import asyncio
import json
import mimetypes
import os
import uuid
from typing import Optional, List, Dict, Any, Union, AsyncIterator
from pathlib import Path

import httpx

# Canonical set of accepted file extensions (mirrors the backend).
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf",
    ".doc", ".docx",
    ".ppt", ".pptx",
    ".xls", ".xlsx",
    # Text / Data
    ".txt", ".csv",
    ".md", ".markdown",
    ".json", ".jsonl",
    ".xml",
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css",
    ".cpp", ".c", ".h",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sql",
    # Images
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif",
    # Audio
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac",
    # Video
    ".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".webm",
}


def _guess_content_type(file_path: Union[str, Path]) -> str:
    """Return a reasonable MIME type for *file_path*, falling back to
    ``application/octet-stream`` when the stdlib cannot detect one."""
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"


def _validate_extension(file_path: Union[str, Path]) -> None:
    """Raise ``ValueError`` if *file_path* has an unsupported extension."""
    ext = os.path.splitext(str(file_path))[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"File type '{ext}' is not supported. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

from ._version import __version__
from .models import (
    Assistant, AssistantCloneResponse, Thread, Document, Message, MessageResponse, ChatMessagesResponse, ContextUsage,
    ToolOutputsResponse, ToolDefinition, ToolOutput,
    Memory, MemoryCreate, MemoryUpdate, MemoriesListResponse, MemoryStats,
    Model, ModelsListResponse,
    EmbeddingModel, EmbeddingModelsListResponse, ProvidersListResponse,
    MemoryOperationStatus,
)
from .exceptions import (
    BackboardAPIError, BackboardValidationError, BackboardNotFoundError,
    BackboardRateLimitError, BackboardServerError
)


class BackboardClient:
    """
    Async Backboard API client for building conversational AI applications
    with persistent memory and intelligent document processing.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://app.backboard.io/api",
        timeout: int = 30,
    ):
        """
        Initialize the Backboard client (async only)

        Args:
            api_key: Your Backboard API key
            base_url: API base URL (default: https://app.backboard.io/api)
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

        # Create async HTTP client
        self._client = httpx.AsyncClient(
            headers={
                "X-API-Key": self.api_key,
                "User-Agent": f"backboard-python-sdk/{__version__} (async)",
            },
            timeout=self.timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BackboardClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make HTTP request with error handling (async)."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        headers = {"X-API-Key": self.api_key}
        # Let httpx set Content-Type automatically based on payload

        try:
            response = await self._client.request(
                method=method,
                url=url,
                json=json_data,
                data=data,
                files=files,
                params=params,
                headers=headers,
            )
            if response.status_code >= 400:
                await self._handle_error_response(response)
            return response
        except httpx.TimeoutException:
            raise BackboardAPIError("Request timed out")
        except httpx.NetworkError:
            raise BackboardAPIError("Connection error")
        except httpx.HTTPError as e:
            raise BackboardAPIError(f"Request failed: {str(e)}")

    async def _stream_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = {"X-API-Key": self.api_key}
        if headers:
            request_headers.update(headers)
        try:
            async with self._client.stream(
                method=method,
                url=url,
                json=json_data,
                data=data,
                files=files,
                params=params,
                headers=request_headers,
            ) as response:
                if response.status_code >= 400:
                    await self._handle_error_response(response)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        try:
                            payload = json.loads(line[6:])
                            # If the backend emits an error event over SSE, raise an SDK exception
                            if isinstance(payload, dict) and payload.get('type') == 'error':
                                error_message = (
                                    payload.get('error')
                                    or payload.get('message')
                                    or "Streaming error"
                                )
                                raise BackboardAPIError(error_message)
                            # Raise for explicit run failure events
                            if isinstance(payload, dict) and payload.get('type') == 'run_failed':
                                error_message = (
                                    payload.get('error')
                                    or payload.get('message')
                                    or "Run failed"
                                )
                                raise BackboardAPIError(error_message)
                            # If a run ends with a non-completed status, raise
                            if (
                                isinstance(payload, dict)
                                and payload.get('type') == 'run_ended'
                                and payload.get('status') not in (None, 'completed')
                            ):
                                status = payload.get('status')
                                raise BackboardAPIError(f"Run ended with status: {status}")
                            yield payload
                        except json.JSONDecodeError:
                            continue
        except httpx.TimeoutException:
            raise BackboardAPIError("Request timed out")
        except httpx.NetworkError:
            raise BackboardAPIError("Connection error")
        except httpx.HTTPError as e:
            raise BackboardAPIError(f"Request failed: {str(e)}")

    async def _handle_error_response(self, response: httpx.Response):
        """Handle error responses from the API (async)."""
        try:
            error_data = response.json()
            error_message = error_data.get('detail', f"HTTP {response.status_code}")
        except Exception:
            try:
                text = response.text
            except Exception:
                text = "<no body>"
            error_message = f"HTTP {response.status_code}: {text}"

        if response.status_code == 400:
            raise BackboardValidationError(error_message, response.status_code, response)
        elif response.status_code == 404:
            raise BackboardNotFoundError(error_message, response.status_code, response)
        elif response.status_code == 429:
            raise BackboardRateLimitError(error_message, response.status_code, response)
        elif response.status_code >= 500:
            raise BackboardServerError(error_message, response.status_code, response)
        else:
            raise BackboardAPIError(error_message, response.status_code, response)

    async def create_assistant(
        self,
        name: str,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Union[ToolDefinition, Dict[str, Any]]]] = None,
        tok_k: Optional[int] = None,
        custom_fact_extraction_prompt: Optional[str] = None,
        custom_update_memory_prompt: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        embedding_dims: Optional[int] = None,
    ) -> Assistant:
        data = {"name": name}
        if description is not None:
            data["description"] = description
        if system_prompt is not None:
            data["system_prompt"] = system_prompt
        if tools:
            data["tools"] = [self._tool_to_dict(tool) for tool in tools]
        if tok_k is not None:
            data["tok_k"] = tok_k
        if custom_fact_extraction_prompt is not None:
            data["custom_fact_extraction_prompt"] = custom_fact_extraction_prompt
        if custom_update_memory_prompt is not None:
            data["custom_update_memory_prompt"] = custom_update_memory_prompt
        if embedding_provider:
            data["embedding_provider"] = embedding_provider
        if embedding_model_name:
            data["embedding_model_name"] = embedding_model_name
        if embedding_dims:
            data["embedding_dims"] = embedding_dims

        response = await self._make_request("POST", "/assistants", json_data=data)
        return Assistant.model_validate(response.json())

    async def list_assistants(self, skip: int = 0, limit: int = 100) -> List[Assistant]:
        """List all assistants for the authenticated user.

        Args:
            skip: Number of records to skip (0–10 000, default 0).
            limit: Maximum number of records to return (1–200, default 100).
        """
        limit = max(1, min(limit, 200))
        skip = max(0, min(skip, 10_000))
        params = {"skip": skip, "limit": limit}
        response = await self._make_request("GET", "/assistants", params=params)
        return [Assistant.model_validate(data) for data in response.json()]

    async def get_assistant(self, assistant_id: Union[str, uuid.UUID]) -> Assistant:
        response = await self._make_request("GET", f"/assistants/{assistant_id}")
        return Assistant.model_validate(response.json())

    async def update_assistant(
        self,
        assistant_id: Union[str, uuid.UUID],
        name: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Union[ToolDefinition, Dict[str, Any]]]] = None,
        tok_k: Optional[int] = None,
        custom_fact_extraction_prompt: Optional[str] = None,
        custom_update_memory_prompt: Optional[str] = None,
    ) -> Assistant:
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if system_prompt is not None:
            data["system_prompt"] = system_prompt
        if tools is not None:
            data["tools"] = [self._tool_to_dict(tool) for tool in tools]
        if tok_k is not None:
            data["tok_k"] = tok_k
        if custom_fact_extraction_prompt is not None:
            data["custom_fact_extraction_prompt"] = custom_fact_extraction_prompt
        if custom_update_memory_prompt is not None:
            data["custom_update_memory_prompt"] = custom_update_memory_prompt

        response = await self._make_request("PUT", f"/assistants/{assistant_id}", json_data=data)
        return Assistant.model_validate(response.json())

    async def delete_assistant(self, assistant_id: Union[str, uuid.UUID]) -> Dict[str, Any]:
        response = await self._make_request("DELETE", f"/assistants/{assistant_id}")
        return response.json()

    async def clone_assistant(
        self,
        assistant_id: Union[str, uuid.UUID],
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        copy_documents: bool = True,
        copy_memories: bool = True,
    ) -> "AssistantCloneResponse":
        """Clone an assistant's configuration, assistant-level documents, and active memories.

        Args:
            assistant_id: External UUID of the source assistant to clone.
            name: Optional name for the cloned assistant.
                  Defaults to ``"{source name} Copy"`` server-side.
            system_prompt: Optional system prompt override for the clone.
                  Defaults to the source assistant's prompt.
            copy_documents: When True (default), clones indexed assistant-level
                  documents and their vector embeddings into the new assistant.
            copy_memories: When True (default), clones active memories attached
                  to the source assistant. The cloner becomes the owner of every
                  cloned memory.

        Returns:
            ``AssistantCloneResponse`` containing the new assistant plus counts
            of documents and memories that were cloned.
        """
        data: Dict[str, Any] = {
            "copy_documents": copy_documents,
            "copy_memories": copy_memories,
        }
        if name is not None:
            data["name"] = name
        if system_prompt is not None:
            data["system_prompt"] = system_prompt

        response = await self._make_request(
            "POST", f"/assistants/{assistant_id}/clone", json_data=data
        )
        return AssistantCloneResponse.model_validate(response.json())

    async def create_thread(self, assistant_id: Union[str, uuid.UUID]) -> Thread:
        response = await self._make_request("POST", f"/assistants/{assistant_id}/threads", json_data={})
        return Thread.model_validate(response.json())

    async def list_threads_for_assistant(
        self,
        assistant_id: Union[str, uuid.UUID],
        skip: int = 0,
        limit: int = 100,
    ) -> List[Thread]:
        """List threads for a specific assistant.

        Args:
            assistant_id: The assistant's UUID.
            skip: Number of records to skip (0–10 000, default 0).
            limit: Maximum number of records to return (1–200, default 100).
        """
        limit = max(1, min(limit, 200))
        skip = max(0, min(skip, 10_000))
        params = {"skip": skip, "limit": limit}
        response = await self._make_request("GET", f"/assistants/{assistant_id}/threads", params=params)
        return [Thread.model_validate(data) for data in response.json()]

    async def list_threads(self, skip: int = 0, limit: int = 100) -> List[Thread]:
        """List all threads for the authenticated user.

        Args:
            skip: Number of records to skip (0–10 000, default 0).
            limit: Maximum number of records to return (1–200, default 100).
        """
        limit = max(1, min(limit, 200))
        skip = max(0, min(skip, 10_000))
        params = {"skip": skip, "limit": limit}
        response = await self._make_request("GET", "/threads", params=params)
        return [Thread.model_validate(data) for data in response.json()]

    async def get_thread(self, thread_id: Union[str, uuid.UUID]) -> Thread:
        response = await self._make_request("GET", f"/threads/{thread_id}")
        return Thread.model_validate(response.json())

    async def delete_thread(self, thread_id: Union[str, uuid.UUID]) -> Dict[str, Any]:
        response = await self._make_request("DELETE", f"/threads/{thread_id}")
        return response.json()

    @staticmethod
    def _parse_add_message_response(data: dict) -> "ChatMessagesResponse":
        """Handle both ChatMessagesResponse and legacy MessageResponse formats."""
        if "messages" in data:
            return ChatMessagesResponse.model_validate(data)
        ctx_raw = data.pop("context_usage", None)
        ctx = ContextUsage.model_validate(ctx_raw) if ctx_raw else None
        return ChatMessagesResponse(messages=[data], context_usage=ctx)

    async def add_message(
        self,
        thread_id: Union[str, uuid.UUID],
        content: Optional[str] = None,
        files: Optional[List[Union[str, Path]]] = None,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        stream: bool = False,
        memory: Optional[str] = None,
        memory_pro: Optional[str] = None,
        memory_response_citation: Optional[bool] = None,
        memory_citation: Optional[bool] = None,
        web_search: Optional[str] = None,
        image_generation: Optional[str] = None,
        image_model_provider: Optional[str] = None,
        image_model_name: Optional[str] = None,
        send_to_llm: Optional[str] = None,
        json_output: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        voice: Optional[Dict[str, Any]] = None,
        audio_file: Optional[Union[str, Path]] = None,
    ) -> Union[ChatMessagesResponse, AsyncIterator[Dict[str, Any]]]:
        """Add a message to a thread.

        Args:
            thinking: Reasoning controls (e.g. ``{"effort": "high"}``).
            voice: Optional voice config dict with ``stt`` and/or ``tts``
                sub-keys to enable speech-to-text / text-to-speech.
            audio_file: Path to a single audio file for STT. Required when
                ``voice`` contains an ``stt`` key.
        """
        form_data: Dict[str, Any] = {
            "stream": "true" if stream else "false",
            "thread_id": str(thread_id),
        }
        if content:
            form_data["content"] = content
        if llm_provider:
            form_data["llm_provider"] = llm_provider
        if model_name:
            form_data["model_name"] = model_name
        if memory:
            form_data["memory"] = memory
        if memory_pro:
            form_data["memory_pro"] = memory_pro
        _cite = memory_response_citation if memory_response_citation is not None else memory_citation
        if _cite is not None:
            form_data["memory_response_citation"] = "true" if _cite else "false"
        if web_search:
            form_data["web_search"] = web_search
        if image_generation:
            form_data["image_generation"] = image_generation
        if image_model_provider:
            form_data["image_model_provider"] = image_model_provider
        if image_model_name:
            form_data["image_model_name"] = image_model_name
        if send_to_llm is not None:
            form_data["send_to_llm"] = send_to_llm
        if json_output is not None:
            form_data["json_output"] = "true" if json_output else "false"
        if metadata:
            form_data["metadata"] = json.dumps(metadata)
        if voice:
            form_data["voice"] = json.dumps(voice)

        has_uploads = bool(files) or bool(audio_file)

        if has_uploads:
            if thinking is not None:
                form_data["thinking"] = json.dumps(thinking)
            files_data = []
            file_handles = []
            try:
                if files:
                    for file_path in files:
                        path = Path(file_path)
                        if not path.exists():
                            raise FileNotFoundError(f"File not found: {file_path}")
                        _validate_extension(path)
                        file_handle = open(path, "rb")
                        file_handles.append(file_handle)
                        files_data.append(("files", (path.name, file_handle, _guess_content_type(path))))

                if audio_file:
                    audio_path = Path(audio_file)
                    if not audio_path.exists():
                        raise FileNotFoundError(f"Audio file not found: {audio_file}")
                    audio_handle = open(audio_path, "rb")
                    file_handles.append(audio_handle)
                    files_data.append(("audio_file", (audio_path.name, audio_handle, _guess_content_type(audio_path))))

                if stream:
                    return self._parse_streaming_response_with_file_cleanup(
                        method="POST",
                        endpoint="/threads/messages",
                        data=form_data,
                        files=files_data,
                        file_handles=file_handles,
                    )
                response = await self._make_request(
                    "POST",
                    "/threads/messages",
                    data=form_data,
                    files=files_data,
                )
                return self._parse_add_message_response(response.json())
            finally:
                if not stream:
                    for fh in file_handles:
                        fh.close()
        elif thinking is not None:
            json_body: Dict[str, Any] = {"stream": stream, "thread_id": str(thread_id)}
            if content:
                json_body["content"] = content
            if llm_provider:
                json_body["llm_provider"] = llm_provider
            if model_name:
                json_body["model_name"] = model_name
            if memory:
                json_body["memory"] = memory
            if memory_pro:
                json_body["memory_pro"] = memory_pro
            if web_search:
                json_body["web_search"] = web_search
            if image_generation:
                json_body["image_generation"] = image_generation
            if image_model_provider:
                json_body["image_model_provider"] = image_model_provider
            if image_model_name:
                json_body["image_model_name"] = image_model_name
            if send_to_llm is not None:
                json_body["send_to_llm"] = send_to_llm
            if metadata:
                json_body["metadata"] = metadata
            json_body["thinking"] = thinking
            if stream:
                return self._parse_streaming_response_iter(
                    method="POST",
                    endpoint="/threads/messages",
                    json_data=json_body,
                )
            response = await self._make_request(
                "POST",
                "/threads/messages",
                json_data=json_body,
            )
            return self._parse_add_message_response(response.json())
        else:
            if stream:
                return self._parse_streaming_response_iter(
                    method="POST",
                    endpoint="/threads/messages",
                    data=form_data,
                )
            response = await self._make_request(
                "POST",
                "/threads/messages",
                data=form_data,
            )
            return self._parse_add_message_response(response.json())

    async def send_message(
        self,
        content: str,
        *,
        thread_id: Optional[Union[str, uuid.UUID]] = None,
        assistant_id: Optional[Union[str, uuid.UUID]] = None,
        system_prompt: Optional[str] = None,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        stream: bool = False,
        memory: Optional[str] = None,
        memory_pro: Optional[str] = None,
        memory_response_citation: Optional[bool] = None,
        web_search: Optional[str] = None,
        image_generation: Optional[str] = None,
        image_model_provider: Optional[str] = None,
        image_model_name: Optional[str] = None,
        send_to_llm: Optional[str] = None,
        json_output: Optional[bool] = None,
        tools: Optional[List[Union[ToolDefinition, Dict[str, Any]]]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[ChatMessagesResponse, AsyncIterator[Dict[str, Any]]]:
        """Send a message without pre-creating an assistant or thread.

        This is the **recommended** entry-point for new integrations.
        Everything you pass is used for **this turn only** — nothing
        is persisted to the assistant.  To configure an assistant
        permanently, use the dedicated assistant endpoints.

        * Omit ``thread_id`` → a new thread (and default assistant) are
          created automatically.  The response contains ``thread_id`` and
          ``assistant_id`` so you can continue the conversation.
        * Pass ``thread_id`` → message is appended to the existing thread
          (stateful continuation).

        Args:
            content: Text content of the message.
            thread_id: Existing thread UUID.  Omit to auto-create.
            assistant_id: Pin the new thread to this assistant.
            system_prompt: Instructions for this turn.  Must be re-passed
                on every call; not persisted.
            llm_provider: LLM provider (``openai``, ``anthropic``, …).
            model_name: Model name (``gpt-4o``, ``claude-…``, …).
            stream: Stream the response as SSE chunks.
            memory: Memory Lite mode (``Auto``, ``Readonly``, ``off``).
            memory_pro: Memory Pro mode (``Auto``, ``Readonly``).
            memory_response_citation: Cite retrieved memories in the reply.
            web_search: ``Auto`` to enable, ``off`` to disable.
            image_generation: ``"auto"`` to enable the built-in image tool.
            image_model_provider: Provider for the image tool model.
            image_model_name: Model name for the image tool.
            send_to_llm: ``"false"`` to save without triggering the LLM.
            json_output: Request JSON object output from the model.
            tools: Tool definitions for this turn (OpenAI-style).  Must
                be re-passed on every call; not persisted.
            thinking: Reasoning controls (e.g. ``{"effort": "high"}``).
            metadata: Arbitrary metadata dict.

        Returns:
            ``ChatMessagesResponse`` (non-streaming) or
            ``AsyncIterator[dict]`` (streaming).
        """
        body: Dict[str, Any] = {"content": content, "stream": stream}
        if thread_id is not None:
            body["thread_id"] = str(thread_id)
        if assistant_id is not None:
            body["assistant_id"] = str(assistant_id)
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if llm_provider:
            body["llm_provider"] = llm_provider
        if model_name:
            body["model_name"] = model_name
        if memory:
            body["memory"] = memory
        if memory_pro:
            body["memory_pro"] = memory_pro
        if memory_response_citation is not None:
            body["memory_response_citation"] = memory_response_citation
        if web_search:
            body["web_search"] = web_search
        if image_generation:
            body["image_generation"] = image_generation
        if image_model_provider:
            body["image_model_provider"] = image_model_provider
        if image_model_name:
            body["image_model_name"] = image_model_name
        if send_to_llm is not None:
            body["send_to_llm"] = send_to_llm
        if json_output is not None:
            body["json_output"] = json_output
        if tools is not None:
            body["tools"] = [self._tool_to_dict(t) for t in tools]
        if thinking is not None:
            body["thinking"] = thinking
        if metadata is not None:
            body["metadata"] = metadata

        if stream:
            return self._parse_streaming_response_iter(
                method="POST",
                endpoint="/threads/messages",
                json_data=body,
            )
        response = await self._make_request(
            "POST", "/threads/messages", json_data=body,
        )
        return self._parse_add_message_response(response.json())

    async def submit_tool_outputs_simple(
        self,
        thread_id: Union[str, uuid.UUID],
        tool_outputs: List[Union[ToolOutput, Dict[str, str]]],
        stream: bool = False,
    ) -> Union[ChatMessagesResponse, AsyncIterator[Dict[str, Any]]]:
        """Submit tool outputs via the simplified endpoint.

        This is the **recommended** way to return tool results.  The
        ``run_id`` is resolved server-side from the latest
        ``REQUIRES_ACTION`` message on the thread — you never need to
        track it yourself.

        Args:
            thread_id: Thread UUID the outputs belong to.
            tool_outputs: List of ``{tool_call_id, output}`` dicts.
            stream: Stream the response as SSE chunks.
        """
        formatted: List[Dict[str, str]] = []
        for output in tool_outputs:
            if isinstance(output, dict):
                formatted.append(output)
            else:
                formatted.append({
                    "tool_call_id": output.tool_call_id,
                    "output": output.output,
                })
        body = {
            "thread_id": str(thread_id),
            "tool_outputs": formatted,
            "stream": stream,
        }

        if stream:
            return self._parse_streaming_response_iter(
                method="POST",
                endpoint="/threads/tool-outputs",
                json_data=body,
            )
        response = await self._make_request(
            "POST", "/threads/tool-outputs", json_data=body,
        )
        return self._parse_add_message_response(response.json())

    async def submit_tool_outputs(
        self,
        thread_id: Union[str, uuid.UUID],
        run_id: str,
        tool_outputs: List[Union[ToolOutput, Dict[str, str]]],
        stream: bool = False
    ) -> Union[ToolOutputsResponse, AsyncIterator[Dict[str, Any]]]:
        formatted_outputs: List[Dict[str, str]] = []
        for output in tool_outputs:
            if isinstance(output, dict):
                formatted_outputs.append(output)
            else:
                formatted_outputs.append({
                    "tool_call_id": output.tool_call_id,
                    "output": output.output,
                })
        data = {"tool_outputs": formatted_outputs}
        params = {"stream": "true" if stream else "false"}

        if stream:
            return self._parse_streaming_response_iter(
                method="POST",
                endpoint=f"/threads/{thread_id}/runs/{run_id}/submit-tool-outputs",
                json_data=data,
                params=params,
            )
        response = await self._make_request(
            "POST",
            f"/threads/{thread_id}/runs/{run_id}/submit-tool-outputs",
            json_data=data,
            params=params,
        )
        return ToolOutputsResponse.model_validate(response.json())

    async def cancel_run(
        self,
        thread_id: Union[str, uuid.UUID],
        run_id: str,
    ) -> Dict[str, Any]:
        """Cancel a run on a thread.

        Idempotent — cancelling an already-finished or already-cancelled
        run returns 200.

        Args:
            thread_id: UUID of the thread.
            run_id: The run identifier to cancel.

        Returns:
            Dict with ``thread_id``, ``run_id``, and ``cancelled`` keys.
        """
        response = await self._make_request(
            "POST", f"/threads/{thread_id}/runs/{run_id}/cancel"
        )
        return response.json()

    async def upload_document_to_assistant(
        self,
        assistant_id: Union[str, uuid.UUID],
        file_path: Union[str, Path]
    ) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        _validate_extension(path)
        content_type = _guess_content_type(path)
        with open(path, "rb") as file_handle:
            files = {"file": (path.name, file_handle, content_type)}
            response = await self._make_request(
                "POST",
                f"/assistants/{assistant_id}/documents",
                files=files,
            )
            return Document.model_validate(response.json())

    async def upload_document_to_thread(
        self,
        thread_id: Union[str, uuid.UUID],
        file_path: Union[str, Path]
    ) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        _validate_extension(path)
        content_type = _guess_content_type(path)
        with open(path, "rb") as file_handle:
            files = {"file": (path.name, file_handle, content_type)}
            response = await self._make_request(
                "POST",
                f"/threads/{thread_id}/documents",
                files=files,
            )
            return Document.model_validate(response.json())

    async def list_assistant_documents(self, assistant_id: Union[str, uuid.UUID]) -> List[Document]:
        response = await self._make_request("GET", f"/assistants/{assistant_id}/documents")
        return [Document.model_validate(data) for data in response.json()]

    async def list_thread_documents(self, thread_id: Union[str, uuid.UUID]) -> List[Document]:
        response = await self._make_request("GET", f"/threads/{thread_id}/documents")
        return [Document.model_validate(data) for data in response.json()]

    async def get_document_status(self, document_id: Union[str, uuid.UUID]) -> Document:
        response = await self._make_request("GET", f"/documents/{document_id}/status")
        return Document.model_validate(response.json())

    async def delete_document(self, document_id: Union[str, uuid.UUID]) -> Dict[str, Any]:
        response = await self._make_request("DELETE", f"/documents/{document_id}")
        return response.json()

    async def get_memories(
        self,
        assistant_id: Union[str, uuid.UUID],
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> MemoriesListResponse:
        """
        Get memories for an assistant with optional pagination.

        Args:
            assistant_id: UUID of the assistant
            page: Page number (1-indexed). Omit to fetch all.
            page_size: Items per page (1-100, default 25).

        Returns:
            MemoriesListResponse with list of memories and total_count
        """
        params: Dict[str, Any] = {}
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["page_size"] = page_size
        response = await self._make_request(
            "GET", f"/assistants/{assistant_id}/memories", params=params or None
        )
        return MemoriesListResponse.model_validate(response.json())

    async def add_memory(
        self,
        assistant_id: Union[str, uuid.UUID],
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new memory to an assistant.
        
        Args:
            assistant_id: UUID of the assistant
            content: Memory content text
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with success status and memory_id
        """
        data = {"content": content}
        if metadata:
            data["metadata"] = metadata
            
        response = await self._make_request("POST", f"/assistants/{assistant_id}/memories", json_data=data)
        return response.json()

    async def get_memory(
        self,
        assistant_id: Union[str, uuid.UUID],
        memory_id: str
    ) -> Memory:
        """
        Get a specific memory by ID.
        
        Args:
            assistant_id: UUID of the assistant
            memory_id: ID of the memory
            
        Returns:
            Memory object
        """
        response = await self._make_request("GET", f"/assistants/{assistant_id}/memories/{memory_id}")
        return Memory.model_validate(response.json())

    async def update_memory(
        self,
        assistant_id: Union[str, uuid.UUID],
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Update an existing memory.
        
        Args:
            assistant_id: UUID of the assistant
            memory_id: ID of the memory to update
            content: New memory content
            metadata: Optional new metadata
            
        Returns:
            Updated Memory object
        """
        data = {"content": content}
        if metadata:
            data["metadata"] = metadata
            
        response = await self._make_request("PUT", f"/assistants/{assistant_id}/memories/{memory_id}", json_data=data)
        return Memory.model_validate(response.json())

    async def delete_memory(
        self,
        assistant_id: Union[str, uuid.UUID],
        memory_id: str
    ) -> Dict[str, Any]:
        """
        Delete a memory.
        
        Args:
            assistant_id: UUID of the assistant
            memory_id: ID of the memory to delete
            
        Returns:
            Dictionary with success status and message
        """
        response = await self._make_request("DELETE", f"/assistants/{assistant_id}/memories/{memory_id}")
        return response.json()

    async def get_memory_stats(self, assistant_id: Union[str, uuid.UUID]) -> MemoryStats:
        """
        Get memory statistics for an assistant.
        
        Args:
            assistant_id: UUID of the assistant
            
        Returns:
            MemoryStats object with usage information
        """
        response = await self._make_request("GET", f"/assistants/{assistant_id}/memories/stats")
        return MemoryStats.model_validate(response.json())

    async def reset_memories(
        self,
        assistant_id: Union[str, uuid.UUID],
    ) -> Dict[str, Any]:
        """Delete all memories for an assistant (database + vector store).

        Args:
            assistant_id: UUID of the assistant

        Returns:
            Dict with ``success`` bool and ``message``
        """
        response = await self._make_request(
            "DELETE", f"/assistants/{assistant_id}/memories"
        )
        return response.json()

    async def search_memories(
        self,
        assistant_id: Union[str, uuid.UUID],
        query: str,
        limit: int = 5,
    ) -> dict:
        """
        Search memories for an assistant using a query string.

        Args:
            assistant_id: UUID of the assistant
            query: Search query text
            limit: Maximum number of results (1-50, default 5)

        Returns:
            Dict with 'memories' list and 'total_count'
        """
        data = {"query": query, "limit": limit}
        response = await self._make_request(
            "POST", f"/assistants/{assistant_id}/memories/search", json_data=data
        )
        return response.json()

    # Model methods

    async def list_models(
        self,
        model_type: Optional[str] = None,
        provider: Optional[str] = None,
        supports_tools: Optional[bool] = None,
        min_context: Optional[int] = None,
        max_context: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> ModelsListResponse:
        """
        List all available models with optional filtering.

        Args:
            model_type: Filter by model type ('llm' or 'embedding')
            provider: Filter by provider name
            supports_tools: Filter by tool/function calling support
            min_context: Minimum context window size
            max_context: Maximum context window size
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            ModelsListResponse with list of models and total count
        """
        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if model_type is not None:
            params["model_type"] = model_type
        if provider is not None:
            params["provider"] = provider
        if supports_tools is not None:
            params["supports_tools"] = supports_tools
        if min_context is not None:
            params["min_context"] = min_context
        if max_context is not None:
            params["max_context"] = max_context

        response = await self._make_request("GET", "/models", params=params)
        return ModelsListResponse.model_validate(response.json())

    async def get_model(self, model_name: str) -> Model:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model (case-insensitive)

        Returns:
            Model object with full details including pricing
        """
        response = await self._make_request("GET", f"/models/{model_name}")
        return Model.model_validate(response.json())

    async def list_models_by_provider(
        self,
        provider_name: str,
        skip: int = 0,
        limit: int = 100,
    ) -> ModelsListResponse:
        """
        List all models from a specific provider.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            ModelsListResponse with list of models and total count
        """
        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        response = await self._make_request("GET", f"/models/provider/{provider_name}", params=params)
        return ModelsListResponse.model_validate(response.json())

    async def list_providers(self) -> ProvidersListResponse:
        """
        List all unique model providers.

        Returns:
            ProvidersListResponse with list of provider names and total count
        """
        response = await self._make_request("GET", "/models/providers")
        return ProvidersListResponse.model_validate(response.json())

    async def list_embedding_models(
        self,
        provider: Optional[str] = None,
        min_dimensions: Optional[int] = None,
        max_dimensions: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> EmbeddingModelsListResponse:
        """
        List all available embedding models with optional filtering.

        Args:
            provider: Filter by provider name
            min_dimensions: Minimum embedding dimensions
            max_dimensions: Maximum embedding dimensions
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            EmbeddingModelsListResponse with list of embedding models and total count
        """
        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if provider is not None:
            params["provider"] = provider
        if min_dimensions is not None:
            params["min_dimensions"] = min_dimensions
        if max_dimensions is not None:
            params["max_dimensions"] = max_dimensions

        response = await self._make_request("GET", "/models/embedding/all", params=params)
        return EmbeddingModelsListResponse.model_validate(response.json())

    async def list_embedding_providers(self) -> ProvidersListResponse:
        """
        List all unique embedding model providers.

        Returns:
            ProvidersListResponse with list of provider names and total count
        """
        response = await self._make_request("GET", "/models/embedding/providers")
        return ProvidersListResponse.model_validate(response.json())

    async def get_embedding_model(self, model_name: str) -> EmbeddingModel:
        """
        Get detailed information about a specific embedding model.

        Args:
            model_name: Name of the embedding model (case-insensitive)

        Returns:
            EmbeddingModel object with full details
        """
        response = await self._make_request("GET", f"/models/embedding/{model_name}")
        return EmbeddingModel.model_validate(response.json())

    # Memory operation methods

    async def get_memory_operation_status(self, operation_id: str) -> MemoryOperationStatus:
        """
        Get the status of a memory add/update/delete operation.

        Args:
            operation_id: UUID of the memory operation

        Returns:
            MemoryOperationStatus with operation details
        """
        response = await self._make_request("GET", f"/assistants/memories/operations/{operation_id}")
        return MemoryOperationStatus.model_validate(response.json())

    @staticmethod
    def _guess_mime_type(file_path: Union[str, Path]) -> str:
        """Guess the MIME type of a file based on its extension."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"

    def _tool_to_dict(self, tool: Union[ToolDefinition, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(tool, dict):
            return tool
        return {
            "type": tool.type,
            "function": {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": {
                    "type": tool.function.parameters.type,
                    "properties": {
                        k: {
                            "type": v.type,
                            "description": v.description,
                            "enum": v.enum,
                            "properties": v.properties,
                            "items": v.items,
                        }
                        for k, v in (tool.function.parameters.properties or {}).items()
                    },
                    "required": tool.function.parameters.required,
                },
            },
        }

    def _parse_streaming_response_iter(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        async def generator() -> AsyncIterator[Dict[str, Any]]:
            async for event in self._stream_request(
                method=method,
                endpoint=endpoint,
                json_data=json_data,
                data=data,
                files=files,
                params=params,
            ):
                yield event
        return generator()

    def _parse_streaming_response_with_file_cleanup(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[List[Any]] = None,
        file_handles: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Parse streaming response and ensure file handles are closed when done.
        This is needed because when we return a generator, file handles must stay
        open until the stream is consumed.
        """
        async def generator() -> AsyncIterator[Dict[str, Any]]:
            try:
                async for event in self._stream_request(
                    method=method,
                    endpoint=endpoint,
                    data=data,
                    files=files,
                    params=params,
                ):
                    yield event
            finally:
                # Close file handles after stream completes or on error
                if file_handles:
                    for fh in file_handles:
                        try:
                            fh.close()
                        except Exception:
                            pass  # Ignore errors closing files
        return generator()

    # ── WebSocket voice streaming ────────────────────────────────────────────

    async def stream_voice(
        self,
        thread_id: str,
        config: Dict[str, Any],
        timeout: float = 15.0,
    ) -> "VoiceSession":
        """Open a WebSocket voice streaming session.

        Args:
            thread_id: The thread to stream voice to.
            config: Session config dict, e.g.
                     ``{"voice": {"stt": {...}, "tts": {...}}, "send_to_llm": True}``
            timeout: Seconds to wait for the ``session.begin`` event.

        Returns:
            A :class:`VoiceSession` that wraps the WebSocket connection.

        Raises:
            BackboardAPIError: If the server responds with an error.
            ImportError: If the ``websockets`` package is not installed.
        """
        try:
            import websockets  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'websockets' package is required for voice streaming. "
                "Install it with:  pip install websockets"
            )

        ws_base = (
            self.base_url
            .replace("https://", "wss://")
            .replace("http://", "ws://")
        )
        url = f"{ws_base}/threads/messages?api_key={self.api_key}"

        ws = await websockets.connect(url, ping_interval=None)
        try:
            payload = {**config, "thread_id": thread_id}
            await ws.send(json.dumps(payload))
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            begin = json.loads(raw)
        except Exception:
            await ws.close()
            raise

        if begin.get("type") == "error":
            await ws.close()
            raise BackboardAPIError(
                begin.get("message", "Voice session failed"),
                status_code=None,
            )

        if begin.get("type") != "session.begin":
            await ws.close()
            raise BackboardAPIError(
                f"Expected session.begin, got {begin.get('type')}",
                status_code=None,
            )

        return VoiceSession(ws, begin)


class VoiceSession:
    """WebSocket voice streaming session returned by
    :meth:`BackboardClient.stream_voice`.

    Typical usage::

        session = await client.stream_voice(thread_id, config)
        async with session:
            await session.send_audio(pcm_bytes)
            async for event in session.events():
                print(event["type"])
    """

    def __init__(self, ws, begin_event: Dict[str, Any]):
        self._ws = ws
        self.begin_event = begin_event
        self.session_id: str = begin_event.get("session_id", "")
        self.thread_id: str = begin_event.get("thread_id", "")
        self.assistant_id: str = begin_event.get("assistant_id", "")
        self.pipeline: Dict[str, Any] = begin_event.get("pipeline", {})

    async def send_audio(self, data: bytes) -> None:
        """Send a binary audio frame (PCM 16-bit, 16 kHz mono)."""
        await self._ws.send(data)

    async def commit(self) -> None:
        """Send a manual commit message (for ``commit_strategy='manual'``)."""
        await self._ws.send(json.dumps({"type": "commit"}))

    async def send_json(self, payload: Dict[str, Any]) -> None:
        """Send an arbitrary JSON control message."""
        await self._ws.send(json.dumps(payload))

    async def receive(self, timeout: float = 15.0) -> Dict[str, Any]:
        """Receive a single JSON event from the server."""
        raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        return json.loads(raw)

    async def events(self, timeout: float = 15.0):
        """Async generator yielding events until the connection closes or
        no event arrives within *timeout* seconds."""
        while True:
            try:
                yield await self.receive(timeout=timeout)
            except asyncio.TimeoutError:
                return
            except Exception as exc:
                try:
                    import websockets as _ws_lib
                    if isinstance(exc, _ws_lib.exceptions.ConnectionClosed):
                        return
                except ImportError:
                    pass
                raise

    async def close(self) -> None:
        """Close the underlying WebSocket."""
        await self._ws.close()

    async def __aenter__(self) -> "VoiceSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
