"""infotext 解析相容工具 / infotext parsing compatibility utilities"""

from typing import Any


def parse_generation_parameters(
    infotext: str, skip_fields: list[str] | None = None
) -> dict[str, Any]:
    """
    解析生成參數的 infotext 字串為字典 / Parse generation parameter infotext string into a dictionary

    此函式作為相容性封裝層，統一處理不同 sd-webui 版本間
    解析函式所在模組位置不同的差異，對呼叫端隱藏版本差異。
    This function acts as a compatibility wrapper that normalizes the
    module location differences of the parsing function across sd-webui
    versions, hiding version differences from callers.

    Neo 版的解析函式接受額外的 skip_fields 參數，
    舊版 A1111 則不接受，因此僅在呼叫端明確提供時才轉傳。
    The Neo version's parser accepts an extra skip_fields argument,
    while legacy A1111 does not, so it is only forwarded when provided.

    @param infotext - 包含生成參數的資訊文字 / Infotext containing generation parameters
    @param skip_fields - 要跳過的欄位名稱列表（僅 Neo 支援）/ Field names to skip (Neo only)
    @returns 解析後的生成參數字典 / Parsed generation parameter dictionary
    """
    try:
        from modules.infotext_utils import parse_generation_parameters as parse
    except ImportError:
        from modules.generation_parameters_copypaste import (
            parse_generation_parameters as parse,
        )

    if skip_fields is not None:
        return parse(infotext, skip_fields)
    return parse(infotext)