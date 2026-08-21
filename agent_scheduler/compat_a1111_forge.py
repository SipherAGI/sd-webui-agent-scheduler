"""sd-webui A1111 版與 Forge Neo 版的相容性函式庫 / Compatibility library for sd-webui A1111 and Forge Neo versions."""

from typing import Any


def parse_generation_parameters(infotext: str) -> dict[str, Any]:
    """
    解析生成參數的 infotext 字串為字典 / Parse generation parameter infotext string into a dictionary

    此函式作為相容性封裝層，統一處理不同 sd-webui 版本間
    解析函式所在模組位置不同的差異，對呼叫端隱藏版本差異。
    This function acts as a compatibility wrapper that normalizes the
    module location differences of the parsing function across sd-webui
    versions, hiding version differences from callers.

    @param infotext - 包含生成參數的資訊文字 / Infotext containing generation parameters
    @returns 解析後的生成參數字典 / Parsed generation parameter dictionary
    """
    """
    優先嘗試新版 (Forge Neo) 的模組位置
    Try the newer (Forge Neo) module location first

    新版 sd-webui 將解析函式移至 modules.infotext_utils；
    若為舊版 A1111 則此匯入會失敗，改為使用舊版模組位置。
    Newer sd-webui moved the parsing function to modules.infotext_utils;
    on legacy A1111 this import fails, so fall back to the legacy module location.
    """
    try:
        from modules.infotext_utils import parse_generation_parameters as parse
    except ImportError:
        from modules.generation_parameters_copypaste import (
            parse_generation_parameters as parse,
        )

    return parse(infotext)