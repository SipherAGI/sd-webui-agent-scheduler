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

is_windows = platform.system() == "Windows"
is_macos = platform.system() == "Darwin"

if logging.getLogger().hasHandlers():
    log = logging.getLogger("sd")
else:
    import copy
    class ColoredFormatter(logging.Formatter):
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
    logger = logging.getLogger("AgentScheduler")
    logger.propagate = False

    # Add handler if we don't have one.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)

    # Configure logger
    loglevel = logging.INFO
    logger.setLevel(loglevel)

    log = logger


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def compare_components_with_ids(components: List[Block], ids: List[int]):
    return len(components) == len(ids) and all(
        component._id == _id for component, _id in zip(components, ids)
    )


def get_component_by_elem_id(root: Block, elem_id: str):
    if root.elem_id == elem_id:
        return root

    elem = None
    if isinstance(root, BlockContext):
        for block in root.children:
            elem = get_component_by_elem_id(block, elem_id)
            if elem is not None:
                break

    return elem


def get_components_by_ids(root: Block, ids: List[int]):
    components: List[Block] = []

    if root._id in ids:
        components.append(root)
        ids = [_id for _id in ids if _id != root._id]

    if isinstance(root, BlockContext):
        for block in root.children:
            components.extend(get_components_by_ids(block, ids))

    return components


def detect_control_net(root: gr.Blocks, submit: gr.Button):
    UiControlNetUnit = None

    dependencies: List[dict] = [
        x
        for x in get_ui_dependencies(root)
        if x["trigger"] == "click" and submit._id in x["targets"]
    ]
    for d in dependencies:
        if len(d["outputs"]) == 1:
            outputs = get_components_by_ids(root, d["outputs"])
            output = outputs[0]
            if (
                isinstance(output, gr.State)
                and type(output.value).__name__ == "UiControlNetUnit"
            ):
                UiControlNetUnit = type(output.value)

    return UiControlNetUnit


def get_dict_attribute(dict_inst: dict, name_string: str, default=None):
    nested_keys = name_string.split(".")
    value = dict_inst

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
    name_list = name_string.split(".")

    # Traverse the dictionary and create a nested dictionary if necessary
    current_dict = dict_inst
    for name in name_list[:-1]:
        if name not in current_dict:
            current_dict[name] = {}
        current_dict = current_dict[name]

    # Set the final attribute to its value
    current_dict[name_list[-1]] = value


def request_with_retry(
    make_request: Callable[[], requests.Response],
    max_try: int = 3,
    retries: int = 0,
):
    try:
        res = make_request()
        if res.status_code > 400:
            raise Exception(res.text)

        return True
    except requests.exceptions.ConnectionError:
        log.error("[ArtVenture] Connection error while uploading result")
        if retries >= max_try - 1:
            return False

        time.sleep(1)
        log.info(f"[ArtVenture] Retrying {retries + 1}...")
        return request_with_retry(
            make_request,
            max_try=max_try,
            retries=retries + 1,
        )
    except Exception as e:
        log.error("[ArtVenture] Error while uploading result")
        log.error(e)
        log.debug(traceback.format_exc())
        return False


def _exit(status: int) -> NoReturn:
    try:
        atexit._run_exitfuncs()
    except:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
