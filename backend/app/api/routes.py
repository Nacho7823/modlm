import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from ..core.exceptions import (
    LLMException,
    NotFoundException,
)
from ..domain.schemas import (
    ChatHistorySchema,
    ChatRequestSchema,
    ChatResponseSchema,
    ChatSchema,
    ConfigUpdateSchema,
    LLMConfigSchema,
    LLMProviderConfigSchema,
    MessageSchema,
)
from ..services import chat_storage, llm_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponseSchema,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(request: ChatRequestSchema) -> ChatResponseSchema:
    try:
        chat_id = request.chat_id
        user_message = MessageSchema(
            role="user",
            content=request.message,
        )

        if not chat_id:
            chat = chat_storage.create_chat()
            chat_id = chat.id
        else:
            chat = chat_storage.get_chat(chat_id)

        chat_storage.add_message(chat_id, user_message)

        messages_for_llm = [
            {"role": msg.role, "content": msg.content} for msg in chat.messages
        ]
        messages_for_llm.append({"role": "user", "content": request.message})

        response_text = llm_service.generate(messages_for_llm)

        assistant_message = MessageSchema(
            role="assistant",
            content=response_text,
        )
        chat_storage.add_message(chat_id, assistant_message)

        return ChatResponseSchema(
            chat_id=chat_id,
            message=user_message,
            response=response_text,
        )

    except NotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)
    except LLMException as e:
        reason = e.details.get("reason", e.message)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=reason)
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/chat/history", response_model=ChatHistorySchema)
async def get_chat_history() -> ChatHistorySchema:
    return chat_storage.get_history()


@router.get("/chat/{chat_id}", response_model=ChatSchema)
async def get_chat(chat_id: str) -> ChatSchema:
    try:
        return chat_storage.get_chat(chat_id)
    except NotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.delete("/chat/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: str) -> None:
    try:
        chat_storage.delete_chat(chat_id)
    except NotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.delete("/chat", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history() -> None:
    chat_storage.clear_history()


@router.get("/config", response_model=LLMConfigSchema)
async def get_config() -> LLMConfigSchema:
    active = llm_service.get_active_provider() or "mock"
    providers = {
        name: LLMProviderConfigSchema(
            provider=name,
            model=llm_service.get_provider(name).get_model_name()
            if llm_service.get_provider(name)
            else "",
            api_key="***",
            base_url=None,
            temperature=0.7,
            max_tokens=None,
        )
        for name in llm_service.list_providers()
    }
    return LLMConfigSchema(
        providers=providers,
        active_provider=active,
    )


@router.put("/config", response_model=LLMConfigSchema)
async def update_config(config: ConfigUpdateSchema) -> LLMConfigSchema:
    try:
        provider_config = LLMProviderConfigSchema(
            provider=config.provider,
            model=config.model or "gpt-3.5-turbo",
            api_key=config.api_key or "",
            base_url=config.base_url,
            temperature=config.temperature or 0.7,
            max_tokens=config.max_tokens,
        )
        llm_service.register_provider(config.provider, provider_config)
        llm_service.set_active_provider(config.provider)

        return await get_config()

    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/chat/new", response_model=ChatSchema)
async def create_new_chat(title: str | None = None) -> ChatSchema:
    return chat_storage.create_chat(title)
