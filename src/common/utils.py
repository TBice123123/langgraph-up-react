"""Utility & helper functions."""

import os
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_qwq import ChatQwen
from langchain_dev_utils import (
    batch_register_model_provider,
    load_chat_model as load_chat_model_utils,
)

from langchain_siliconflow import ChatSiliconFlow


batch_register_model_provider(
    [
        {
            "provider": "dashscope",
            "chat_model": ChatQwen,
        },
        {
            "provider": "siliconflow",
            "chat_model": ChatSiliconFlow,
        },
    ]
)


def normalize_region(region: str) -> Optional[str]:
    """Normalize region aliases to standard values.

    Args:
        region: Region string to normalize

    Returns:
        Normalized region ('prc' or 'international') or None if invalid
    """
    if not region:
        return None

    region_lower = region.lower()
    if region_lower in ("prc", "cn"):
        return "prc"
    elif region_lower in ("international", "en"):
        return "international"
    return None


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(
    fully_specified_name: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider:model'.
    """
    region = os.getenv("REGION")
    base_url = None
    if "dashscope" in fully_specified_name:
        base_url = os.getenv("DASHSCOPE_API_BASE")
        if base_url is None and region:
            normalized_region = normalize_region(region)
            if normalized_region == "prc":
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            elif normalized_region == "international":
                base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    if "siliconflow" in fully_specified_name:
        base_url = os.getenv("SILICONFLOW_API_BASE")
        if base_url is None and region:
            normalized_region = normalize_region(region)
            if normalized_region == "prc":
                base_url = "https://api.siliconflow.cn/v1"
            elif normalized_region == "international":
                base_url = "https://api.siliconflow.com/v1"
    if base_url:
        kwargs["base_url"] = base_url

    return load_chat_model_utils(fully_specified_name, **kwargs)
