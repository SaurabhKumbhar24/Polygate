from . import batch
from .batch import BatchRequest, BatchResult
from .client import chat
from .concurrent import Completion, map_chat, map_prompts
from .exceptions import (
    PolygateError,
    MissingAPIKeyError,
    MissingEndpointError,
    ProviderAPIError,
    UnsupportedProviderError,
)
from .providers import CANONICAL_PROVIDERS, SUPPORTED_PROVIDERS
from .retry import KeyPool, Retry
from .types import ChatResponse, Usage

__version__ = "0.3.0"

__all__ = [
    "chat",
    # Many at once: concurrently now, or as a cheaper offline job.
    "map_chat",
    "map_prompts",
    "Completion",
    "batch",
    "BatchRequest",
    "BatchResult",
    "ChatResponse",
    "Usage",
    "Retry",
    "KeyPool",
    "PolygateError",
    "UnsupportedProviderError",
    "MissingAPIKeyError",
    "MissingEndpointError",
    "ProviderAPIError",
    "SUPPORTED_PROVIDERS",
    "CANONICAL_PROVIDERS",
]
