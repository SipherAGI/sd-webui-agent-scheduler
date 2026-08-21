"""pydantic 版本相容工具 / pydantic version compatibility utilities"""

import pydantic

from typing import Any

"""是否為 pydantic v2 / Whether pydantic v2 is in use

A1111 classic 使用 pydantic v1，Forge Neo 使用 pydantic v2，
兩者的 model config 定義方式不同。
A1111 classic uses pydantic v1 while Forge Neo uses pydantic v2,
which differ in how model config is defined.
"""
PYDANTIC_V2 = int(pydantic.VERSION.split(".")[0]) >= 2


def api_task_schema_extra(schema: dict[str, Any], model) -> None:
    """從 JSON Schema 中移除 API 內部使用的欄位 / Remove internal API fields from the JSON schema

    send_images 與 save_images 僅供 sd-webui 內部 API 使用，
    對 queue API 的使用者並無意義，故於文件中隱藏。
    send_images and save_images are only used by sd-webui's internal API,
    meaningless to queue API consumers, so they are hidden from the docs.

    @param schema - 產生的 JSON schema / Generated JSON schema
    @param model - 對應的 pydantic 模型 / Corresponding pydantic model
    """
    props = schema.get("properties", {})
    props.pop("send_images", None)
    props.pop("save_images", None)