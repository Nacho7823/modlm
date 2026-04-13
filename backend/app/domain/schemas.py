from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MessageSchema(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.now)
    tool_calls: Optional[list["ToolCallSchema"]] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError("Role must be user, assistant, or system")
        return v

    class Config:
        from_attributes = True


class ToolCallSchema(BaseModel):
    tool_name: str
    arguments: dict
    result: Optional[str] = None

    class Config:
        from_attributes = True


class ToolSchema(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    function: str = Field(..., min_length=1)

    class Config:
        from_attributes = True


class ChatSchema(BaseModel):
    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    messages: list[MessageSchema] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


class ChatHistorySchema(BaseModel):
    chats: list[ChatSchema] = Field(default_factory=list)
    active_chat_id: Optional[str] = None

    class Config:
        from_attributes = True


class LLMProviderConfigSchema(BaseModel):
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    api_key: str = Field(default="")
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)

    class Config:
        from_attributes = True


class LLMConfigSchema(BaseModel):
    providers: dict[str, LLMProviderConfigSchema] = Field(default_factory=dict)
    active_provider: str = Field(default="openai")

    class Config:
        from_attributes = True


class ChatRequestSchema(BaseModel):
    message: str = Field(..., min_length=1)
    chat_id: Optional[str] = None

    class Config:
        from_attributes = True


class ChatResponseSchema(BaseModel):
    chat_id: str
    message: MessageSchema
    response: str

    class Config:
        from_attributes = True


class ConfigUpdateSchema(BaseModel):
    provider: str = Field(..., min_length=1)
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)

    class Config:
        from_attributes = True
