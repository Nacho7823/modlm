from ..core.exceptions import AppException


class LLMServiceError(AppException):
    pass


class StorageError(AppException):
    pass


class ConfigError(AppException):
    pass
