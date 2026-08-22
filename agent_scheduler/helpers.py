"""
helpers.py — Agent Scheduler 的通用輔助工具 / General-purpose helper utilities for the Agent Scheduler.

本模組提供日誌設定、單例元類、gradio 元件搜尋、巢狀字典存取、帶重試的請求，
以及程序安全結束等共用功能，供擴充其他模組重複使用。
This module provides logging setup, a singleton metaclass, gradio component lookup, nested dict
access, retrying requests, and safe process exit used across the extension.
"""

import os
import sys
import abc
import atexit
import time
import logging
import platform
import requests
import traceback
from typing import Callable, List, NoReturn

import gradio as gr
from gradio.blocks import Block, BlockContext

from agent_scheduler.compat_a1111_forge.ui import get_ui_dependencies

# 記錄目前作業系統，用於跨平台行為判斷 / Record the current OS for cross-platform branching.
is_windows = platform.system() == "Windows"
is_macos = platform.system() == "Darwin"

# 若根 logger 已有處理器（通常由 sd-webui 啟動流程設定），直接沿用其 "sd" logger / Reuse the existing "sd" logger if handlers are already configured.
if logging.getLogger().hasHandlers():
    log = logging.getLogger("sd")
else:
    import copy
    class ColoredFormatter(logging.Formatter):
        """
        為不同等級的日誌加上終端機顏色 / Add terminal colors to log records by level.

        為什麼需要：在 sd-webui 尚未配置日誌處理器時，提供具可讀性的彩色輸出。
        Why: when sd-webui has not configured logging, provide readable color-coded console output.
        """
        COLORS = {
            "DEBUG": "\033[0;36m",  # CYAN
            "INFO": "\033[0;32m",  # GREEN
            "WARNING": "\033[0;33m",  # YELLOW
            "ERROR": "\033[0;31m",  # RED
            "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
            "RESET": "\033[0m",  # RESET COLOR
        }

        def format(self, record):
            colored_record = copy.copy(record)
            levelname = colored_record.levelname
            seq = self.COLORS.get(levelname, self.COLORS["RESET"])
            colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
            return super().format(colored_record)

    # Create a new logger
    # 建立專屬的 AgentScheduler logger，不向上層傳播以免重複輸出 / Create a dedicated logger that does not propagate to avoid duplicate output.
    logger = logging.getLogger("AgentScheduler")
    logger.propagate = False

    # Add handler if we don't have one.
    # 若尚未有處理器，則加入帶彩色格式的 stdout 處理器 / Attach a color-formatted stdout handler if none exists yet.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)

    # Configure logger
    # 預設日誌等級設為 INFO / Default logging level is INFO.
    loglevel = logging.INFO
    logger.setLevel(loglevel)

    log = logger


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    確保類別在全域只會產生單一實例的元類 / Metaclass guaranteeing a single global instance per class.

    為什麼需要：排程器中的管理器（如任務佇列）應全域唯一，避免狀態分散。
    Why: schedulers (e.g. task queues) must be globally unique to keep shared state consistent.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        # 若該類別尚未建立實例，則建立並快取；之後皆回傳同一實例 / Lazily create and cache the instance, always returning the same one.
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def compare_components_with_ids(components: List[Block], ids: List[int]):
    """
    比對元件清單與其 id 是否一一對應 / Check that components match the given ids one-to-one.

    為什麼需要：在尋找 UI 相依性時，需確認找到的元件順序與預期 id 完全一致。
    Why: when resolving UI dependencies we must confirm the found components exactly match the expected ids in order.
    """
    # 數量需相同且每個元件的 id 都相符 / Counts must match and every component id must be equal.
    return len(components) == len(ids) and all(
        component._id == _id for component, _id in zip(components, ids)
    )


def get_component_by_elem_id(root: Block, elem_id: str):
    """
    依 elem_id 在 gradio 元件樹中遞迴尋找元件 / Recursively find a gradio component by its elem_id.

    為什麼需要：排程器需要根據 elem_id 定位特定 UI 元件以讀取或綁定其值。
    Why: the scheduler must locate specific UI components by elem_id to read or bind their values.
    """
    # 根節點即為目標則直接回傳 / Return the root itself if it matches.
    if root.elem_id == elem_id:
        return root

    elem = None
    # 若為容器（BlockContext），則遞迴搜尋其子元件，找到即停止 / If a container, recurse into children and stop at the first match.
    if isinstance(root, BlockContext):
        for block in root.children:
            elem = get_component_by_elem_id(block, elem_id)
            if elem is not None:
                break

    return elem


def get_components_by_ids(root: Block, ids: List[int]):
    """
    依 id 清單在 gradio 元件樹中蒐集對應元件 / Collect gradio components matching the given id list.

    為什麼需要：UI 相依性中記錄的是元件 id，需轉回實際元件物件才能操作。
    Why: UI dependencies are recorded as component ids, which must be resolved back to objects to use them.
    """
    components: List[Block] = []

    # 若根節點的 id 在清單中，則加入並從待找清單移除 / If root matches, add it and remove from the pending ids.
    if root._id in ids:
        components.append(root)
        ids = [_id for _id in ids if _id != root._id]

    # 若為容器則遞迴蒐集其餘子元件 / If a container, recurse to collect remaining child components.
    if isinstance(root, BlockContext):
        for block in root.children:
            components.extend(get_components_by_ids(block, ids))

    return components


def detect_control_net(root: gr.Blocks, submit: gr.Button):
    """
    偵測 submit 按鈕所連動的 ControlNet 單元型別 / Detect the ControlNet unit type wired to the submit button.

    為什麼需要：ControlNet 單元型別在執行期才存在，需從 UI 依賴關係推導其具體類別。
    Why: the ControlNet unit type only exists at runtime and must be inferred from UI dependency wiring.
    """
    UiControlNetUnit = None

    # 取出所有「點擊 submit 即觸發」的 UI 依賴 / Collect every dependency triggered by clicking the submit button.
    dependencies: List[dict] = [
        x
        for x in get_ui_dependencies(root)
        if x["trigger"] == "click" and submit._id in x["targets"]
    ]
    for d in dependencies:
        # 僅處理單一輸出的依賴（ControlNet 單元通常對應一個 State 輸出）/ Only single-output deps (a ControlNet unit maps to one State output).
        if len(d["outputs"]) == 1:
            outputs = get_components_by_ids(root, d["outputs"])
            output = outputs[0]
            # 若該輸出是一個值為 ControlNet 單元的 State，則記錄其型別 / If the output is a State holding a ControlNet unit, record its type.
            if (
                isinstance(output, gr.State)
                and type(output.value).__name__ == "UiControlNetUnit"
            ):
                UiControlNetUnit = type(output.value)

    return UiControlNetUnit


def get_dict_attribute(dict_inst: dict, name_string: str, default=None):
    """
    以點號路徑讀取巢狀字典中的值 / Read a value from a nested dict using dot-path notation.

    為什麼需要：任務參數常有巢狀結構（如 alwayson_scripts.controlnet.args），以字串路徑存取較方便。
    Why: task params are often nested (e.g. alwayson_scripts.controlnet.args), so a string path is convenient.
    """
    # 將點號路徑拆解為鍵串 / Split the dot-path into individual keys.
    nested_keys = name_string.split(".")
    value = dict_inst

    # 逐層取值，任一層缺失即回傳預設值 / Walk each level; return default as soon as any level is missing.
    for key in nested_keys:
        value = value.get(key, None)

        if value is None:
            return default

    return value


def set_dict_attribute(dict_inst: dict, name_string: str, value):
    """
    Set an attribute to a dictionary using dot notation.
    If the attribute does not already exist, it will create a nested dictionary.

    Parameters:
        - dict_inst: the dictionary instance to set the attribute
        - name_string: the attribute name in dot notation (ex: 'attribute.name')
        - value: the value to set for the attribute

    Returns:
        None
    """
    # Split the attribute names by dot
    # 將點號路徑拆解為鍵清單 / Split the dot-path into a list of keys.
    name_list = name_string.split(".")

    # Traverse the dictionary and create a nested dictionary if necessary
    # 逐層走訪，遇到不存在的鍵就自動建立巢狀字典 / Walk each level, auto-creating nested dicts when a key is missing.
    current_dict = dict_inst
    for name in name_list[:-1]:
        # 若不存在則先建立空字典，確保路徑暢通 / Create an empty dict if the key is absent so the path exists.
        if name not in current_dict:
            current_dict[name] = {}
        current_dict = current_dict[name]

    # Set the final attribute to its value
    # 在最後一層設定實際值 / Set the actual value at the final level.
    current_dict[name_list[-1]] = value


def request_with_retry(
    make_request: Callable[[], requests.Response],
    max_try: int = 3,
    retries: int = 0,
):
    """
    執行請求並在連線失敗時重試 / Execute a request and retry on connection failure.

    為什麼需要：上傳結果到外部服務時網路可能暫時不穩，重試可提高成功率且避免任務因瞬斷而失敗。
    Why: uploading results to external services can hit transient network errors, so retrying improves robustness.
    """
    try:
        res = make_request()
        # 視為失敗的狀態碼（>400）直接拋錯進入重試/錯誤處理 / Treat status >400 as failure and raise into the error path.
        if res.status_code > 400:
            raise Exception(res.text)

        return True
    # 連線錯誤時依剩餘次數重試，超過上限則放棄 / On connection error retry up to the limit, then give up.
    except requests.exceptions.ConnectionError:
        log.error("[ArtVenture] Connection error while uploading result")
        # 已達最大重試次數則結束並回傳失敗 / Stop and report failure once retries are exhausted.
        if retries >= max_try - 1:
            return False

        # 等待後遞迴重試，次數加一 / Wait then recurse with an incremented retry count.
        time.sleep(1)
        log.info(f"[ArtVenture] Retrying {retries + 1}...")
        return request_with_retry(
            make_request,
            max_try=max_try,
            retries=retries + 1,
        )
    # 其他例外一律記錄並回傳失敗，不中斷排程 / Any other exception is logged and reported as failure without crashing the scheduler.
    except Exception as e:
        log.error("[ArtVenture] Error while uploading result")
        log.error(e)
        log.debug(traceback.format_exc())
        return False


def _exit(status: int) -> NoReturn:
    """
    強制結束程序並確保緩衝區已清空 / Forcefully exit the process after flushing buffers.

    為什麼需要：排程器在嚴重錯誤時需立即終止，但仍應執行 atexit 清理並清空輸出。
    Why: on fatal errors the scheduler must terminate immediately while still running atexit cleanup and flushing output.
    """
    # 嘗試執行已註冊的結束函式；任何異常都忽略以免阻擋結束 / Run registered atexit funcs, ignoring any error so exit is never blocked.
    try:
        atexit._run_exitfuncs()
    except:
        pass
    # 確保標準輸出/錯誤緩衝區都已寫出 / Flush stdout and stderr so nothing is lost.
    sys.stdout.flush()
    sys.stderr.flush()
    # 立即終止程序 / Terminate the process immediately.
    os._exit(status)
