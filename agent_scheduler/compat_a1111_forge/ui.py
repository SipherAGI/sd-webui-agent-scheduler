"""gradio UI 相依結構的版本相容工具 / Version-compatible utilities for gradio UI dependency structures"""

import json

from typing import Any, Dict, List

"""
統一取得 gradio Blocks 的 click 相依列表 / Uniformly get the click dependencies of a gradio Blocks

舊版 gradio (A1111 classic) 以 root.dependencies 儲存，其元素為 dict，
包含 trigger / targets / inputs / outputs 等以元件 id 表示的欄位。
新版 gradio (Forge Neo, 4.x) 改以 root.fns 儲存 BlockFunction 物件，
targets 為 (id, event_name) 元組、inputs / outputs 為元件物件列表。
此函式將兩種結構正規化為舊版 dict 格式，供既有程式碼直接使用。
Legacy gradio (A1111 classic) stores dependencies in root.dependencies as dicts
with trigger / targets / inputs / outputs fields expressed as component ids.
Newer gradio (Forge Neo, 4.x) stores BlockFunction objects in root.fns instead,
with targets as (id, event_name) tuples and inputs / outputs as component lists.
This function normalizes both into the legacy dict format for reuse by existing code.
"""


def get_ui_dependencies(root: Any) -> List[Dict[str, Any]]:
    """取得正規化的 UI 相依列表 / Get the normalized UI dependency list

    @param root - gradio Blocks 實例 / gradio Blocks instance
    @returns 正規化相依列表，元素格式為 / Normalized dependency list, each element shaped as
        {"trigger": str, "targets": [int], "inputs": [int], "outputs": [int]}
    """
    if hasattr(root, "dependencies"):
        return root.dependencies

    dependencies: List[Dict[str, Any]] = []
    for fn in getattr(root, "fns", {}).values():
        trigger = fn.targets[0][1] if fn.targets else None
        dependencies.append(
            {
                "trigger": trigger,
                "targets": [block_id for block_id, _ in fn.targets],
                "inputs": [c._id for c in fn.inputs],
                "outputs": [c._id for c in fn.outputs],
            }
        )
    return dependencies


def get_ui_fns(root: Any) -> List[Any]:
    """取得 BlockFunction 物件列表 / Get the list of BlockFunction objects

    新版 gradio 的 root.fns 為 dict，迭代時會得到 id 鍵而非函式物件；
    此 helper 統一回傳可迭代的 BlockFunction 列表。
    Newer gradio's root.fns is a dict, so iterating yields id keys instead of
    function objects; this helper uniformly returns an iterable list of BlockFunction.

    @param root - gradio Blocks 實例 / gradio Blocks instance
    @returns BlockFunction 物件列表 / List of BlockFunction objects
    """
    fns = getattr(root, "fns", None)
    if isinstance(fns, dict):
        return list(fns.values())
    return list(fns) if fns else []


def get_ui_task_geninfo(result: Any) -> Any:
    """從 txt2img / img2img 的 UI 回傳元組中取出 generation info JSON 字串 / Extract the generation info JSON string from a txt2img / img2img UI result tuple

    A1111 回傳 (images, geninfo, html_info, comments)，geninfo 位於索引 1。
    Forge 回傳 (gallery, video_arg, geninfo, html_info, comments)，geninfo 位於索引 2。
    以「字串且可解析為 JSON」定位 geninfo，因此兩種結構皆相容。
    A1111 returns (images, geninfo, html_info, comments) with geninfo at index 1.
    Forge returns (gallery, video_arg, geninfo, html_info, comments) with geninfo at index 2.
    The geninfo is located by being a string parseable as JSON, so both structures work.

    @param result - wrap_gradio_call 的回傳元組 / result tuple from wrap_gradio_call
    @returns geninfo JSON 字串；找不到時回傳 None / geninfo JSON string, or None if not found
    """
    # 從第 2 個元素開始掃描：geninfo 是「可解析為 JSON 的字串」。
    # 用 json.loads 探測而非寫死索引，因此 A1111 (idx 1) 與 Forge (idx 2) 皆相容。
    # Scan from the 2nd element onward: geninfo is "a string parseable as JSON".
    # Probing with json.loads (instead of hard-coding indices) keeps A1111 (idx 1)
    # and Forge (idx 2) both compatible.
    for item in result[1:]:
        if isinstance(item, str):
            try:
                json.loads(item)
                return item
            except (TypeError, ValueError):
                continue
    return None


def get_ui_task_error_text(result: Any) -> str:
    """從 UI 回傳元組中取出錯誤/資訊文字（排除 geninfo）/ Get the error / info text from the UI result tuple (excluding geninfo)

    A1111 將錯誤資訊放在索引 2（html_info），Forge 則依例外或正常回傳路徑
    分別位於索引 1（exception fallback 的 ""）或索引 3（html_info）。
    此函式收集除 geninfo 以外的所有字串元素，讓呼叫端可針對
    "CUDA out of memory" 等訊息做檢查，而不需理會各版本的索引差異。
    A1111 places error info at index 2 (html_info); Forge places it at index 1
    (the "" in the exception fallback) or index 3 (html_info) depending on path.
    This collects every string element except geninfo so callers can scan for
    messages such as "CUDA out of memory" without caring about per-version indexes.

    @param result - wrap_gradio_call 的回傳元組 / result tuple from wrap_gradio_call
    @returns 合併後的錯誤/資訊文字 / the concatenated error / info text
    """
    geninfo = get_ui_task_geninfo(result)
    # 收集合併除 geninfo 以外的所有字串，讓呼叫端能以單一字串掃描 OOM 等錯誤訊息
    # Collect and join every string except geninfo so callers can scan one string for
    # errors such as "CUDA out of memory".
    return "\n".join(item for item in result[1:] if isinstance(item, str) and item != geninfo)