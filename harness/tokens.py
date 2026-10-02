"""Counting tokens the way OpenAI models do, with an estimate when tiktoken can't load."""

import logging
from collections.abc import Callable
from functools import cache

logger = logging.getLogger(__name__)

# English averages a little under four characters per token.
_CHARS_PER_TOKEN = 4


@cache
def _counter() -> Callable[[str], int]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # an offline machine: estimate instead of failing
        logger.warning("tiktoken unavailable; estimating token counts from length")
        return lambda text: -(-len(text) // _CHARS_PER_TOKEN)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def count_tokens(text: str) -> int:
    return _counter()(text) if text else 0
