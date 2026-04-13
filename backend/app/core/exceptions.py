from typing import Any


class AppException(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class NotFoundException(AppException):
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            message=f"{resource} not found",
            details={"resource": resource, "resource_id": resource_id},
        )


class ValidationException(AppException):
    def __init__(self, field: str, reason: str):
        super().__init__(
            message=f"Validation error: {field}",
            details={"field": field, "reason": reason},
        )


class LLMException(AppException):
    def __init__(self, provider: str, reason: str):
        super().__init__(
            message=f"LLM error from {provider}",
            details={"provider": provider, "reason": reason},
        )


class ConfigurationException(AppException):
    def __init__(self, setting: str, reason: str):
        super().__init__(
            message=f"Configuration error: {setting}",
            details={"setting": setting, "reason": reason},
        )


class StorageException(AppException):
    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Storage error during {operation}",
            details={"operation": operation, "reason": reason},
        )
