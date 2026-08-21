"""gradio UI 相依結構的版本相容工具 / Version-compatible utilities for gradio UI dependency structures"""

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