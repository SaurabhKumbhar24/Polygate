from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Usage:
    """Normalized token usage, regardless of provider."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """Normalized response returned by chat(), regardless of provider."""
    content: str
    role: str
    model: str
    provider: str
    usage: Usage
    raw: Dict[str, Any] = field(default_factory=dict)
