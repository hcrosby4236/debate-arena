"""
Backboard API data models using Pydantic
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import uuid
import json

from pydantic import BaseModel, Field, field_validator, computed_field


class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class MessageRole(str, Enum):
    """Message role types"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCallFunction(BaseModel):
    """Tool call function definition"""
    name: str
    arguments: str  # JSON string of arguments
    
    @computed_field
    @property
    def parsed_arguments(self) -> Dict[str, Any]:
        """Parse arguments JSON string into a dictionary"""
        try:
            return json.loads(self.arguments)
        except (json.JSONDecodeError, TypeError):
            return {}


class ToolCall(BaseModel):
    """Tool call from assistant response"""
    id: str
    type: str
    function: ToolCallFunction


class ToolParameterProperties(BaseModel):
    """Tool parameter property definition"""
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None
    items: Optional[Dict[str, Any]] = None


class ToolParameters(BaseModel):
    """Tool parameters definition"""
    type: str = "object"
    properties: Dict[str, ToolParameterProperties] = Field(default_factory=dict)
    required: Optional[List[str]] = None


class FunctionDefinition(BaseModel):
    """Function definition for tools"""
    name: str
    description: Optional[str] = None
    parameters: ToolParameters


class ToolDefinition(BaseModel):
    """Tool definition"""
    type: str = "function"
    function: Optional[FunctionDefinition] = None


class Assistant(BaseModel):
    """Assistant model"""
    assistant_id: uuid.UUID
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None
    tok_k: Optional[int] = None
    embedding_provider: Optional[str] = None
    embedding_model_name: Optional[str] = None
    embedding_dims: Optional[int] = None
    custom_fact_extraction_prompt: Optional[str] = None
    custom_update_memory_prompt: Optional[str] = None
    created_at: datetime

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class AssistantCloneResponse(BaseModel):
    """Response returned by POST /assistants/{assistant_id}/clone"""
    assistant: Assistant
    documents_cloned: int = 0
    memories_cloned: int = 0


class AttachmentInfo(BaseModel):
    """Message attachment information"""
    document_id: uuid.UUID
    filename: str
    status: str
    file_size_bytes: Optional[int] = None
    summary: Optional[str] = None


class Message(BaseModel):
    """Message model"""
    message_id: uuid.UUID
    role: MessageRole
    content: Optional[str] = None
    created_at: datetime
    status: Optional[str] = None
    # API uses `metadata_` (not `metadata`); keep field name `metadata_` for parity.
    metadata_: Optional[Dict[str, Any]] = Field(default=None, alias='metadata_')
    attachments: Optional[List[AttachmentInfo]] = None

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    class Config:
        populate_by_name = True


class LatestMessageInfo(BaseModel):
    """Reduced latest message payload (unique fields only)"""
    metadata_: Optional[Dict[str, Any]] = Field(default=None, alias="metadata_")
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    created_at: Optional[datetime] = None

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    class Config:
        populate_by_name = True


class Thread(BaseModel):
    """Thread model"""
    thread_id: uuid.UUID
    created_at: datetime
    messages: List[Message] = Field(default_factory=list)
    # API uses `metadata_` (not `metadata`); keep field name `metadata_` for parity.
    metadata_: Optional[Dict[str, Any]] = Field(default=None, alias='metadata_')

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    class Config:
        populate_by_name = True


class Document(BaseModel):
    """Document model"""
    document_id: uuid.UUID
    filename: str
    status: DocumentStatus
    created_at: datetime
    status_message: Optional[str] = None
    summary: Optional[str] = None
    updated_at: Optional[datetime] = None
    file_size_bytes: Optional[int] = None
    total_tokens: Optional[int] = None
    chunk_count: Optional[int] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    document_type: Optional[str] = None
    # API uses `metadata_` (not `metadata`); keep field name `metadata_` for parity.
    metadata_: Optional[Dict[str, Any]] = Field(default=None, alias='metadata_')

    @field_validator('created_at', 'updated_at', 'processing_started_at', 'processing_completed_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    class Config:
        populate_by_name = True


class ContextUsage(BaseModel):
    """Token context-window usage returned alongside chat messages."""
    percent: float = 0
    used_tokens: int = 0
    context_limit: int = 0
    summary_tokens: int = 0
    model: str = "unknown"


class ChatMessagesResponse(BaseModel):
    """Response from adding a message (multi-message format).

    Convenience properties proxy to the **last** message in the ``messages``
    list so callers can write ``response.status`` instead of
    ``response.messages[-1].get("status")``.

    ``thread_id`` and ``assistant_id`` are always populated by
    ``POST /threads/messages`` so callers can continue the conversation::

        r = await client.send_message("Hi!")
        r2 = await client.send_message("Again", thread_id=r.thread_id)
    """
    messages: List[Dict[str, Any]]
    context_usage: Optional[ContextUsage] = None

    @property
    def status(self) -> Optional[str]:
        if self.messages:
            return self.messages[-1].get("status")
        return None

    @property
    def tool_calls(self) -> Optional[List[ToolCall]]:
        if self.messages:
            raw = self.messages[-1].get("tool_calls")
            if raw:
                return [ToolCall.model_validate(tc) for tc in raw]
        return None

    @property
    def run_id(self) -> Optional[str]:
        if self.messages:
            return self.messages[-1].get("run_id")
        return None

    @property
    def reasoning(self) -> Optional[str]:
        if self.messages:
            return self.messages[-1].get("reasoning")
        return None

    @property
    def content(self) -> Optional[str]:
        if self.messages:
            return self.messages[-1].get("content")
        return None

    @property
    def thread_id(self) -> Optional[str]:
        if self.messages:
            return self.messages[-1].get("thread_id")
        return None

    @property
    def assistant_id(self) -> Optional[str]:
        if self.messages:
            return self.messages[-1].get("assistant_id")
        return None


class MessageResponse(BaseModel):
    """Response from adding a message to a thread"""
    message: str
    thread_id: uuid.UUID
    content: Optional[str] = None
    # Backend response schema allows these to be null in some cases
    message_id: Optional[uuid.UUID] = None
    role: Optional[MessageRole] = None
    status: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    run_id: Optional[str] = None
    memory_operation_id: Optional[str] = None
    retrieved_memories: Optional[List[Dict[str, Any]]] = None
    retrieved_files: Optional[List[str]] = None
    reasoning: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    created_at: Optional[datetime] = None
    attachments: Optional[List[AttachmentInfo]] = None
    # Nested STT/TTS: transcript, audio URLs, billing dimensions (matches API `voice_records`)
    voice_records: Optional[Dict[str, Any]] = None
    timestamp: datetime

    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v
    
    def __str__(self):
        """Return a clean string representation with readable values"""
        return (
            f"MessageResponse(\n"
            f"  message='{self.message}',\n"
            f"  thread_id='{self.thread_id}',\n"
            f"  content='{self.content}',\n"
            f"  message_id='{self.message_id}',\n"
            f"  role='{self.role.value if self.role else None}',\n"
            f"  status='{self.status}',\n"
            f"  tool_calls={self.tool_calls},\n"
            f"  run_id='{self.run_id}',\n"
            f"  memory_operation_id='{self.memory_operation_id}',\n"
            f"  retrieved_files={self.retrieved_files},\n"
            f"  reasoning='{self.reasoning}',\n"
            f"  model_provider='{self.model_provider}',\n"
            f"  model_name='{self.model_name}',\n"
            f"  input_tokens={self.input_tokens},\n"
            f"  output_tokens={self.output_tokens},\n"
            f"  total_tokens={self.total_tokens},\n"
            f"  created_at='{self.created_at.isoformat() if self.created_at else None}',\n"
            f"  attachments={self.attachments},\n"
            f"  timestamp='{self.timestamp.isoformat()}'\n"
            f")"
        )


class ToolOutput(BaseModel):
    """Tool output for submitting tool results"""
    tool_call_id: str
    output: str


class SubmitToolOutputsRequest(BaseModel):
    """Request for submitting tool outputs"""
    tool_outputs: List[ToolOutput]


class ToolOutputsResponse(BaseModel):
    """Response from submitting tool outputs"""
    message: str
    thread_id: uuid.UUID
    run_id: str
    content: Optional[str] = None
    # Backend response schema allows these to be null in some cases
    message_id: Optional[uuid.UUID] = None
    role: Optional[MessageRole] = None
    status: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    memory_operation_id: Optional[str] = None
    retrieved_memories: Optional[List[Dict[str, Any]]] = None
    retrieved_files: Optional[List[str]] = None
    reasoning: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    created_at: Optional[datetime] = None
    timestamp: datetime
    context_usage: Optional[ContextUsage] = None

    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v
    
    def __str__(self):
        """Return a clean string representation with readable values"""
        return (
            f"ToolOutputsResponse(\n"
            f"  message='{self.message}',\n"
            f"  thread_id='{self.thread_id}',\n"
            f"  run_id='{self.run_id}',\n"
            f"  content='{self.content}',\n"
            f"  message_id='{self.message_id}',\n"
            f"  role='{self.role.value if self.role else None}',\n"
            f"  status='{self.status}',\n"
            f"  tool_calls={self.tool_calls},\n"
            f"  memory_operation_id='{self.memory_operation_id}',\n"
            f"  retrieved_files={self.retrieved_files},\n"
            f"  reasoning='{self.reasoning}',\n"
            f"  model_provider='{self.model_provider}',\n"
            f"  model_name='{self.model_name}',\n"
            f"  input_tokens={self.input_tokens},\n"
            f"  output_tokens={self.output_tokens},\n"
            f"  total_tokens={self.total_tokens},\n"
            f"  created_at='{self.created_at.isoformat() if self.created_at else None}',\n"
            f"  timestamp='{self.timestamp.isoformat()}'\n"
            f")"
        )


# Memory Models

class Memory(BaseModel):
    """Memory model"""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MemoryCreate(BaseModel):
    """Schema for creating a memory"""
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MemoryUpdate(BaseModel):
    """Schema for updating a memory"""
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MemoriesListResponse(BaseModel):
    """Response for listing memories"""
    memories: List[Memory]
    total_count: int
    page: Optional[int] = None
    page_size: Optional[int] = None
    total_pages: Optional[int] = None


class MemoryStats(BaseModel):
    """Memory statistics"""
    total_memories: int = 0
    last_updated: Optional[str] = None
    limits: Optional[Dict[str, Any]] = None


# Model Catalog Models

class Model(BaseModel):
    """Model information including pricing and modality capabilities"""
    name: str
    provider: str
    model_type: str
    context_limit: int
    max_output_tokens: Optional[int] = None
    supports_tools: Optional[bool] = None
    supports_vision: Optional[bool] = None
    supports_audio_input: Optional[bool] = None
    supports_video_input: Optional[bool] = None
    supports_image_output: Optional[bool] = None
    supports_audio_output: Optional[bool] = None
    supports_video_output: Optional[bool] = None
    api_mode: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    input_cost_per_1m_tokens: Optional[float] = None
    cached_input_cost_per_1m_tokens: Optional[float] = None
    image_input_cost_per_1m_tokens: Optional[float] = None
    output_cost_per_1m_tokens: Optional[float] = None
    image_output_cost_per_1m_tokens: Optional[float] = None
    cost_per_image: Optional[float] = None
    last_updated: Optional[datetime] = None

    @field_validator('last_updated', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class ModelsListResponse(BaseModel):
    """Response for listing models"""
    models: List[Model]
    total: int


class ImageModel(BaseModel):
    """Image generation model information and pricing"""
    name: str
    provider: str
    model_type: str
    context_limit: int
    supports_vision: Optional[bool] = None
    input_cost_per_1m_tokens: Optional[float] = None
    image_input_cost_per_1m_tokens: Optional[float] = None
    output_cost_per_1m_tokens: Optional[float] = None
    image_output_cost_per_1m_tokens: Optional[float] = None
    cost_per_image: Optional[float] = None
    last_updated: Optional[datetime] = None

    @field_validator('last_updated', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class ImageModelsListResponse(BaseModel):
    """Response for listing image generation models"""
    models: List[ImageModel]
    total: int


class EmbeddingModel(BaseModel):
    """Embedding model information"""
    name: str
    provider: str
    embedding_dimensions: int
    context_limit: int
    last_updated: Optional[datetime] = None

    @field_validator('last_updated', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class EmbeddingModelsListResponse(BaseModel):
    """Response for listing embedding models"""
    models: List[EmbeddingModel]
    total: int


class VoiceModel(BaseModel):
    """Voice model (TTS/STT) information including pricing"""
    name: str
    provider: str
    model_type: str
    supports_streaming: Optional[bool] = None
    input_cost_per_1m_tokens: Optional[float] = None
    output_cost_per_1m_tokens: Optional[float] = None
    audio_input_cost_per_1m_tokens: Optional[float] = None
    cached_audio_input_cost_per_1m_tokens: Optional[float] = None
    audio_output_cost_per_1m_tokens: Optional[float] = None
    cost_per_1m_characters: Optional[float] = None
    cost_per_minute: Optional[float] = None
    supported_output_formats: Optional[str] = None
    supported_input_formats: Optional[str] = None
    last_updated: Optional[datetime] = None

    @field_validator('last_updated', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class VoiceModelsListResponse(BaseModel):
    """Response for listing voice (TTS/STT) models"""
    models: List[VoiceModel]
    total: int


class ProvidersListResponse(BaseModel):
    """Response for listing providers"""
    providers: List[str]
    total: int


class MemoryOperationStatus(BaseModel):
    """Status of a memory add/update/delete operation"""
    operation_id: str
    status: str
    memory_ids: Optional[List[Any]] = None
    result_count: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


# Billing Models


class UsageEvent(BaseModel):
    """Single usage event from billing activity"""
    id: int
    client_id: int
    ledger_id: Optional[int] = None
    operation_id: Optional[str] = None
    message_id: Optional[int] = None
    model_name: Optional[str] = None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    audio_cached_tokens: int = 0
    characters: int = 0
    duration_seconds: float = 0.0
    token_cost_usd: float
    vector_reads: int
    vector_cost_usd: float
    memory_write_cost_usd: float = 0.0
    total_cost_usd: float
    subscription_credits_used: float = 0.0
    regular_credits_used: float = 0.0
    created_at: datetime

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class UsageEventSummary(BaseModel):
    """Slim usage event returned by GET /billing/usage/recent"""
    date: datetime
    model: Optional[str] = None
    input_tokens: int
    output_tokens: int
    vector_reads: int
    vector_cost_usd: float
    memory_write_cost_usd: float
    credit_type: str
    amount_deducted_usd: float

    @field_validator('date', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class PaginationMeta(BaseModel):
    """Pagination metadata"""
    page: int
    page_size: int
    total: int
    total_pages: int


class PaginatedUsageResponse(BaseModel):
    """Paginated list of usage events (full)"""
    data: List[UsageEvent]
    pagination: PaginationMeta


class PaginatedUsageSummaryResponse(BaseModel):
    """Paginated list of slim usage event summaries"""
    data: List[UsageEventSummary]
    pagination: PaginationMeta