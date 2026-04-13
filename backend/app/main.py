import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .core.config import settings
from .domain.schemas import LLMProviderConfigSchema
from .services import config_storage, llm_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_saved_config() -> None:
    try:
        saved = config_storage.load()
        providers = saved.get("providers", {})

        for name, cfg in providers.items():
            provider_config = LLMProviderConfigSchema(
                provider=name,
                model=cfg.get("model", "gpt-3.5-turbo"),
                api_key=cfg.get("api_key", ""),
                base_url=cfg.get("base_url"),
                temperature=cfg.get("temperature", 0.7),
                max_tokens=cfg.get("max_tokens"),
                system_prompt=cfg.get("system_prompt"),
            )
            llm_service.register_provider(name, provider_config)

        active = saved.get("active_provider", "openai")
        if active in providers:
            llm_service.set_active_provider(active)

        logger.info(f"Loaded config: {len(providers)} providers, active={active}")
    except Exception as e:
        logger.warning(f"Could not load saved config: {e}")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=settings.api_prefix)

    @app.on_event("startup")
    async def startup_event():
        load_saved_config()

    return app


app = create_app()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.app_version}
