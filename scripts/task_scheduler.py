"""任務排程器腳本 / Task scheduler script.

將 Agent Scheduler 的佇列功能掛載到 sd-webui 的 txt2img / img2img UI，
並負責註冊排程相關的 UI 元件、設定項目與 app 啟動邏輯。
Mounts the Agent Scheduler queue into the sd-webui txt2img/img2img UI and
registers scheduler-related UI components, settings, and app-started hooks.
"""

import os
import json
import gradio as gr
from PIL import Image
from uuid import uuid4
from typing import List
from collections import defaultdict
from datetime import datetime, timedelta

from modules import call_queue, shared, script_callbacks, scripts, ui_components
from modules.shared import list_checkpoint_tiles, refresh_checkpoints
from modules.cmd_args import parser
from modules.ui import create_refresh_button
from modules.ui_common import save_files
from modules.sd_models import model_path
from agent_scheduler.compat_a1111_forge.infotext import parse_generation_parameters
from agent_scheduler.compat_a1111_forge.paste_params import (
    ParamBinding,
    connect_paste_params_buttons,
    register_paste_params_button,
    registered_param_bindings,
)
from agent_scheduler.compat_a1111_forge.ui import get_ui_dependencies, get_ui_fns

from agent_scheduler.task_runner import TaskRunner, get_instance
from agent_scheduler.helpers import log, compare_components_with_ids, get_components_by_ids, is_macos
from agent_scheduler.db import init as init_db, task_manager, TaskStatus
from agent_scheduler.api import regsiter_apis

# 判斷是否為 SD.Next 環境，以利切換不同的 UI 元件型別 / Detect SD.Next to pick a compatible UI component type
is_sdnext = parser.description == "SD.Next"
ToolButton = gr.Button if is_sdnext else ui_components.ToolButton

# 全域任務執行器實例，於 on_app_started 時初始化 / Global task runner instance, initialized in on_app_started
task_runner: TaskRunner = None

checkpoint_current = "Current Checkpoint"
checkpoint_runtime = "Runtime Checkpoint"
queue_with_every_checkpoints = "$$_queue_with_all_checkpoints_$$"

ui_placement_as_tab = "As a tab"
ui_placement_append_to_main = "Append to main UI"

placement_under_generate = "Near Generate button"
placement_between_prompt_and_generate = "Between Prompt and Generate button"

completion_action_choices = ["Do nothing", "Shut down", "Restart", "Sleep", "Hibernate", "Stop webui", "Restart webui"]

task_filter_choices = ["All", "Bookmarked", "Done", "Failed", "Interrupted"]

enqueue_key_modifiers = [
    "Command" if is_macos else "Ctrl",
    "Control" if is_macos else "Alt",
    "Shift",
]
enqueue_default_hotkey = enqueue_key_modifiers[0] + "+KeyE"
enqueue_key_codes = {}
enqueue_key_codes.update({chr(i): "Key" + chr(i) for i in range(ord("A"), ord("Z") + 1)})
enqueue_key_codes.update({chr(i): "Digit" + chr(i) for i in range(ord("0"), ord("9") + 1)})
enqueue_key_codes.update({"`": "Backquote", "Enter": "Enter"})

task_history_retenion_map = {
    "1 day": 1,
    "3 days": 3,
    "7 days": 7,
    "14 days": 14,
    "30 days": 30,
    "90 days": 90,
    "Keep forever": 0,
}

init_db()


class Script(scripts.Script):
    """sd-webui 擴充主腳本 / Main sd-webui script extension.

    負責在 txt2img / img2img 介面上注入 Enqueue 按鈕與 checkpoint 下拉選單，
    並將其點擊事件綁定到排程器的任務註冊邏輯。
    Injects the Enqueue button and checkpoint dropdown into the txt2img/img2img UI
    and binds their click events to the scheduler's task-registration logic.
    """

    def __init__(self):
        super().__init__()
        # 註冊 app 啟動回呼，等待 UI 元件就緒後綁定事件 / Register app-started hook to bind events once UI components exist
        script_callbacks.on_app_started(lambda block, _: self.on_app_started(block))
        self.checkpoint_override = checkpoint_current
        self.generate_button = None
        self.enqueue_row = None
        self.checkpoint_dropdown = None
        self.submit_button = None

    def title(self):
        # 擴充在 UI 上顯示的名稱 / Display name of the extension in the UI
        return "Agent Scheduler"

    def show(self, is_img2img):
        # 此擴充在兩種模式皆常駐顯示 / Always show this extension in both modes
        return scripts.AlwaysVisible

    def on_checkpoint_changed(self, checkpoint):
        # 記錄使用者於下拉選單選取的 checkpoint，供後續任務使用 / Store the checkpoint chosen in the dropdown for queued tasks
        self.checkpoint_override = checkpoint

    def after_component(self, component, **_kwargs):
        """在每個 UI 元件建立後掛載 Enqueue 按鈕 / Hook to mount the Enqueue button after each UI component is built.

        sd-webui 會在建構每個元件時呼叫此函式，我們藉此定位 generate 按鈕所在的容器，
        並依使用者設定的擺放位置把 Enqueue 列插入正確的位置。
        sd-webui calls this per component; we locate the generate button's container and insert the
        Enqueue row at the user-configured placement.
        """
        # 依模式選擇對應的 elem_id 前綴 / Pick the elem_id prefix based on the current mode
        id_part = "img2img" if self.is_img2img else "txt2img"

        enqueue_wrapper = f"{id_part}_enqueue_wrapper"
        generate_id = f"{id_part}_generate"

        # 讀取影響按鈕擺放的使用者設定 / Read the settings that affect button placement
        compact_prompt_box = getattr(shared.opts, "compact_prompt_box", False)
        queue_button_placement = getattr(shared.opts, "queue_button_placement", placement_under_generate)

        component_elem_id = _kwargs.get('elem_id')

        # 先捕捉 generate 按鈕以備後續綁定 / Capture the generate button for later event binding
        if component_elem_id == generate_id:
            self.generate_button = component

        # 若 Enqueue 列已建立，或遇到不需處理的容器則提早結束 / Stop early if the row already exists or this container is irrelevant
        if self.enqueue_row is not None:
            return
        elif component_elem_id is None or component_elem_id == enqueue_wrapper:
            return

        generate_box = f"{id_part}_generate_box"
        actions_column_id = f"{id_part}_actions_column"
        results_id = f"{id_part}_results"
        neg_id = f"{id_part}_neg_prompt"
        toprow_id = f"{id_part}_toprow"

        # 是否選擇放在提示詞與生成按鈕之間 / Whether the button should sit between prompt and generate
        bool_placement_between_prompt_and_generate = queue_button_placement == placement_between_prompt_and_generate

        def add_enqueue_row(elem_id):
            # 找到目標容器後插入 Enqueue 列；若元件已被放在錯誶的父層，則搬移到正確位置 / Find the target container then insert the row, reparenting if misplaced
            parent = component if elem_id is None or component.elem_id == elem_id else component.parent
            while parent is not None:
                if parent.elem_id is None or elem_id is None or parent.elem_id == elem_id:
                    self.add_enqueue_button()
                    if component.parent != parent:
                        component.parent.children.pop()
                        parent.add(self.enqueue_row)
                    break
                parent = parent.parent

        # 依設定與介面佈局，選擇合適的掛載點插入 Enqueue 列 / Choose the mount point based on settings and layout
        if component_elem_id == generate_id:
            if not compact_prompt_box:
                if not bool_placement_between_prompt_and_generate:
                    add_enqueue_row(actions_column_id)
        elif component_elem_id == neg_id:
            if not compact_prompt_box:
                if bool_placement_between_prompt_and_generate:
                    add_enqueue_row(toprow_id)
        elif component_elem_id == results_id:
            if compact_prompt_box and not bool_placement_between_prompt_and_generate:
                add_enqueue_row(results_id)

    def on_app_started(self, block):
        # UI 元件就緒後才綁定 Enqueue 按鈕事件 / Bind the Enqueue button only after UI components exist
        if self.generate_button is not None:
            self.bind_enqueue_button(block)

    def add_enqueue_button(self):
        # 建立 Enqueue 列：checkpoint 下拉選單與 Enqueue 按鈕 / Build the Enqueue row: checkpoint dropdown plus the Enqueue button
        id_part = "img2img" if self.is_img2img else "txt2img"
        with gr.Row(elem_id=f"{id_part}_enqueue_wrapper") as row:
            self.enqueue_row = row
            hide_checkpoint = getattr(shared.opts, "queue_button_hide_checkpoint", True)
            self.checkpoint_dropdown = gr.Dropdown(
                choices=get_checkpoint_choices(),
                value=checkpoint_current,
                show_label=False,
                interactive=True,
                visible=not hide_checkpoint,
            )
            # 當顯示 checkpoint 下拉時，提供重新整理按鈕以同步模型清單 / When the dropdown is visible, offer a refresh button to sync the model list
            if not hide_checkpoint:
                create_refresh_button(
                    self.checkpoint_dropdown,
                    refresh_checkpoints,
                    lambda: {"choices": get_checkpoint_choices()},
                    f"refresh_{id_part}_checkpoint",
                )
            # 實際的 Enqueue 按鈕 / The actual Enqueue submit button
            self.submit_button = gr.Button("Enqueue", elem_id=f"{id_part}_enqueue", variant="primary")

    def bind_enqueue_button(self, root: gr.Blocks):
        # 找出 generate 按鈕背後的 gradio 依賴，用來複製其輸入元件並掛載 Enqueue 行為 / Locate the generate button's dependencies to clone its inputs and attach Enqueue behavior
        generate = self.generate_button
        is_img2img = self.is_img2img
        dependencies: List[dict] = [
            x for x in get_ui_dependencies(root) if x["trigger"] == "click" and generate._id in x["targets"]
        ]

        dependency: dict = None
        cnet_dependency: dict = None
        UiControlNetUnit = None
        # 從依賴中識別主生成函式與 ControlNet 單元，分別綁定 / Identify the main generate fn and the ControlNet unit among dependencies
        for d in dependencies:
            if len(d["outputs"]) == 1:
                outputs = get_components_by_ids(root, d["outputs"])
                output = outputs[0]
                if isinstance(output, gr.State) and type(output.value).__name__ == "UiControlNetUnit":
                    cnet_dependency = d
                    UiControlNetUnit = type(output.value)

            elif len(d["outputs"]) >= 4:
                dependency = d

        with root:
            # 同步 checkpoint 下拉選單的變更 / Sync changes from the checkpoint dropdown
            if self.checkpoint_dropdown is not None:
                self.checkpoint_dropdown.change(fn=self.on_checkpoint_changed, inputs=[self.checkpoint_dropdown])

            # 找到與主生成函式相同輸入的 BlockFunction，作為 Enqueue 的輸入來源 / Find the BlockFunction sharing the generate fn's inputs to reuse as Enqueue inputs
            fn_block = next(fn for fn in get_ui_fns(root) if compare_components_with_ids(fn.inputs, dependency["inputs"]))
            fn = self.wrap_register_ui_task()
            inputs = fn_block.inputs.copy()
            inputs.insert(0, self.checkpoint_dropdown)
            args = dict(
                fn=fn,
                _js="submit_enqueue_img2img" if is_img2img else "submit_enqueue",
                inputs=inputs,
                outputs=None,
                show_progress=False,
            )

            # 將 Enqueue 點擊事件綁定到包裝後的註冊函式 / Bind the Enqueue click event to the wrapped registration fn
            self.submit_button.click(**args)

            # 若存在 ControlNet 單元，額外綁定以確保其設定一併送出 / If a ControlNet unit exists, bind it too so its config is submitted
            if cnet_dependency is not None:
                cnet_fn_block = next(
                    fn for fn in get_ui_fns(root) if compare_components_with_ids(fn.inputs, cnet_dependency["inputs"])
                )
                self.submit_button.click(
                    fn=UiControlNetUnit,
                    inputs=cnet_fn_block.inputs,
                    outputs=cnet_fn_block.outputs,
                    queue=False,
                )

    def wrap_register_ui_task(self):
        # 包裝 Enqueue 行為：解析 checkbox、checkpoint、task 名稱後註冊任務 / Wrap Enqueue: resolve checkbox/checkpoint/task name then register the task
        def f(request: gr.Request, *args):
            # 至少需要一個參數，否則視為非法呼叫 / Require at least one arg or treat the call as invalid
            if len(args) == 0:
                raise Exception("Invalid call")

            checkpoint: str = args[0]
            task_id = args[1]
            args = args[1:]
            task_name = None

            # 特別關鍵字：對每一個 checkpoint 各產生一個任務 / Special keyword: create one task per available checkpoint
            if task_id == queue_with_every_checkpoints:
                task_id = str(uuid4())
                checkpoint = list_checkpoint_tiles()
            else:
                # 非 task(...) 開頭表示這是任務名稱，需產生新 id / A non task(...) prefix means a task name; generate a fresh id
                if not task_id.startswith("task("):
                    task_name = task_id
                    task_id = str(uuid4())

                # 依下拉選項決定實際要使用的 checkpoint 清單 / Resolve the actual checkpoint list from the dropdown option
                if checkpoint is None or checkpoint == "" or checkpoint == checkpoint_current:
                    checkpoint = [shared.sd_model.sd_checkpoint_info.title]
                elif checkpoint == checkpoint_runtime:
                    checkpoint = [None]
                elif checkpoint.endswith(" checkpoints)"):
                    checkpoint_dir = " ".join(checkpoint.split(" ")[0:-2])
                    checkpoint = list(filter(lambda c: c.startswith(checkpoint_dir), list_checkpoint_tiles()))
                else:
                    checkpoint = [checkpoint]

            # 多個 checkpoint 時為每個產生獨立 task id / When multiple checkpoints, give each a distinct task id
            for i, c in enumerate(checkpoint):
                t_id = task_id if i == 0 else f"{task_id}.{i}"

                # gr.Info(f"[AgentScheduler] Add new Task {t_id} {task_name or ''}")

                # 註冊單一 UI 任務，帶入 checkpoint 與任務名稱 / Register a single UI task with checkpoint and task name
                task_runner.register_ui_task(
                    t_id,
                    self.is_img2img,
                    *args,
                    checkpoint=c,
                    task_name=task_name,
                    request=request,
                )

            # 註冊後立即觸發背景執行緒處理佇列 / Trigger the background runner after registering
            task_runner.execute_pending_tasks_threading()

        return f


def get_checkpoint_choices():
    """建構 Enqueue 下拉選單的 checkpoint 選項 / Build the checkpoint choices for the Enqueue dropdown.

    除了個別 checkpoint 外，也依目錄分組提供「整個資料夾」選項，方便一次排程多個模型。
    Besides individual checkpoints, group by directory to offer "whole folder" options for batch queuing.
    """
    checkpoints: List[str] = list_checkpoint_tiles()

    # 統計各目錄下的 checkpoint 數量，用於分組選項 / Count checkpoints per directory for grouped options
    checkpoint_dirs = defaultdict(lambda: 0)
    for checkpoint in checkpoints:
        checkpoint_dir = os.path.dirname(checkpoint)
        while checkpoint_dir != "" and checkpoint_dir != "/":
            checkpoint_dirs[checkpoint_dir] += 1
            checkpoint_dir = os.path.dirname(checkpoint_dir)

    choices = checkpoints
    choices.extend([f"{d} ({checkpoint_dirs[d]} checkpoints)" for d in checkpoint_dirs.keys()])
    choices = sorted(choices)

    # 固定將「當前 / 執行時」選項放在首位 / Keep "Current / Runtime" options pinned at the top
    choices.insert(0, checkpoint_runtime)
    choices.insert(0, checkpoint_current)

    return choices


def create_send_to_buttons():
    # 建立把結果送到其他分頁（txt2img/img2img/...）的按鈕組 / Build the "send to" buttons that push results to other tabs
    return {
        "txt2img": ToolButton(
            "➠ text" if is_sdnext else "📝",
            elem_id="agent_scheduler_send_to_txt2img",
            tooltip="Send generation parameters to txt2img tab.",
        ),
        "img2img": ToolButton(
            "➠ image" if is_sdnext else "🖼️",
            elem_id="agent_scheduler_send_to_img2img",
            tooltip="Send image and generation parameters to img2img tab.",
        ),
        "inpaint": ToolButton(
            "➠ inpaint" if is_sdnext else "🎨️",
            elem_id="agent_scheduler_send_to_inpaint",
            tooltip="Send image and generation parameters to img2img inpaint tab.",
        ),
        "extras": ToolButton(
            "➠ process" if is_sdnext else "📐",
            elem_id="agent_scheduler_send_to_extras",
            tooltip="Send image and generation parameters to extras tab.",
        ),
    }


def infotexts_to_geninfo(infotexts: List[str]):
    # 將多筆 infotext 彙整成 geninfo 結構，供前端展示 / Aggregate multiple infotexts into a geninfo structure for the UI
    all_promts = []
    all_seeds = []

    geninfo = {"infotexts": infotexts, "all_prompts": all_promts, "all_seeds": all_seeds, "index_of_first_image": 0}

    for infotext in infotexts:
        # Dynamic prompt breaks layout of infotext
        # 動態提示詞會破壞 infotext 排版，故移除 Template 行 / Dynamic-prompt Template lines break parsing, so strip them
        if "Template: " in infotext:
            lines = infotext.split("\n")
            lines = [l for l in lines if not (l.startswith("Template: ") or l.startswith("Negative Template: "))]
            infotext = "\n".join(lines)

        # 解析單筆 infotext 的產圖參數 / Parse the generation parameters from a single infotext
        params = parse_generation_parameters(infotext)

        # 僅以第一筆作為整體 geninfo 的主要欄位 / Use the first infotext as the overall geninfo main fields
        if "prompt" not in geninfo:
            geninfo["prompt"] = params.get("Prompt", "")
            geninfo["negative_prompt"] = params.get("Negative prompt", "")
            geninfo["seed"] = params.get("Seed", "-1")
            geninfo["sampler_name"] = params.get("Sampler", "")
            geninfo["cfg_scale"] = params.get("CFG scale", "")
            geninfo["steps"] = params.get("Steps", "0")
            geninfo["width"] = params.get("Size-1", "512")
            geninfo["height"] = params.get("Size-2", "512")

        all_promts.append(params.get("Prompt", ""))
        all_seeds.append(params.get("Seed", "-1"))

    return geninfo


def get_task_results(task_id: str, image_idx: int = None):
    # 依 task id 讀取結果並整理成前端要顯示的元件更新 / Load a task's result and build the UI component updates to display it
    task = task_manager.get_task(task_id)

    galerry = None
    geninfo = None
    infotext = None
    # 任務不存在時不顯示內容 / No-op when the task does not exist
    if task is None:
        pass
    # 未完成的任務僅顯示狀態（失敗時附錯誤） / Show status only for unfinished tasks, plus error text on failure
    elif task.status != TaskStatus.DONE:
        infotext = f"Status: {task.status}"
        if task.status == TaskStatus.FAILED and task.result:
            infotext += f"\nError: {task.result}"
    # 已完成的任務解析 result，準備圖片與 geninfo / Parse results for completed tasks to prepare images and geninfo
    elif task.status == TaskStatus.DONE:
        try:
            result: dict = json.loads(task.result)
            images = result.get("images", [])
            geninfo = result.get("geninfo", None)
            if isinstance(geninfo, dict):
                infotexts = geninfo.get("infotexts", [])
            else:
                infotexts = result.get("infotexts", [])
                geninfo = infotexts_to_geninfo(infotexts)

            # 依 image_idx 決定顯示全部圖片或單張，並取對應的 infotext / Show all images or a single one, picking the matching infotext
            galerry = [Image.open(i) for i in images if os.path.exists(i)] if image_idx is None else gr.update()
            idx = image_idx if image_idx is not None else 0
            if idx < len(infotexts):
                infotext = infotexts[idx]
        # 解析失敗時回報錯誤而不中斷整個 UI / Report parse errors gracefully instead of breaking the UI
        except Exception as e:
            log.error(f"[AgentScheduler] Failed to load task result")
            log.error(e)
            infotext = f"Failed to load task result: {str(e)}"

    res = (
        gr.Textbox.update(infotext, visible=infotext is not None),
        gr.Row.update(visible=galerry is not None),
    )

    if image_idx is None:
        geninfo = json.dumps(geninfo) if geninfo else None
        res += (
            galerry,
            gr.Textbox.update(geninfo),
            gr.File.update(None, visible=False),
            gr.HTML.update(None),
        )

    return res


def remove_old_tasks():
    # delete task that are too old
    # 依設定刪除過舊的歷史任務，避免資料庫無限增長 / Delete tasks older than the retention window to keep the DB bounded
    retention_days = 30
    # 若使用者在設定中指定了保留天數則採用該值 / Honor the user-configured retention window if set
    if (
        getattr(shared.opts, "queue_history_retention_days", None)
        and shared.opts.queue_history_retention_days in task_history_retenion_map
    ):
        retention_days = task_history_retenion_map[shared.opts.queue_history_retention_days]

    # 僅在保留天數大於 0 時執行刪除（"Keep forever" 對應 0） / Only delete when retention > 0 ("Keep forever" maps to 0)
    if retention_days > 0:
        deleted_rows = task_manager.delete_tasks(before=datetime.now() - timedelta(days=retention_days))
        if deleted_rows > 0:
            log.debug(f"[AgentScheduler] Deleted {deleted_rows} tasks older than {retention_days} days")


def on_ui_tab(**_kwargs):
    # 建構 Agent Scheduler 的頁籤 UI（任務佇列與歷史） / Build the Agent Scheduler tab UI (queue + history)
    grid_page_size = getattr(shared.opts, "queue_grid_page_size", 0)

    with gr.Blocks(analytics_enabled=False) as scheduler_tab:
        with gr.Tabs(elem_id="agent_scheduler_tabs"):
            with gr.Tab("Task Queue", id=0, elem_id="agent_scheduler_pending_tasks_tab"):
                with gr.Row(elem_id="agent_scheduler_pending_tasks_wrapper"):
                    with gr.Column(scale=1):
                        with gr.Row(elem_id="agent_scheduler_pending_tasks_actions", elem_classes="flex-row"):
                            paused = getattr(shared.opts, "queue_paused", False)

                            gr.Button(
                                "Pause",
                                elem_id="agent_scheduler_action_pause",
                                variant="stop",
                                visible=not paused,
                            )
                            gr.Button(
                                "Resume",
                                elem_id="agent_scheduler_action_resume",
                                variant="primary",
                                visible=paused,
                            )
                            gr.Button(
                                "Refresh",
                                elem_id="agent_scheduler_action_reload",
                                variant="secondary",
                            )
                            gr.Button(
                                "Clear",
                                elem_id="agent_scheduler_action_clear_queue",
                                variant="stop",
                            )
                            gr.Button(
                                "Export",
                                elem_id="agent_scheduler_action_export",
                                variant="secondary",
                            )
                            gr.Button(
                                "Import",
                                elem_id="agent_scheduler_action_import",
                                variant="secondary",
                            )
                            gr.HTML(f'<input type="file" id="agent_scheduler_import_file" style="display: none" accept="application/json" />')

                            with gr.Row(elem_classes=["agent_scheduler_filter_container", "flex-row", "ml-auto"]):
                                gr.Textbox(
                                    max_lines=1,
                                    placeholder="Search",
                                    label="Search",
                                    show_label=False,
                                    min_width=0,
                                    elem_id="agent_scheduler_action_search",
                                )
                        gr.HTML(
                            f'<div id="agent_scheduler_pending_tasks_grid" class="ag-theme-gradio" data-page-size="{grid_page_size}"></div>'
                        )
                    with gr.Column(scale=1):
                        gr.Gallery(
                            elem_id="agent_scheduler_current_task_images",
                            label="Output",
                            show_label=False,
                            columns=2,
                            object_fit="contain",
                        )
            with gr.Tab("Task History", id=1, elem_id="agent_scheduler_history_tab"):
                with gr.Row(elem_id="agent_scheduler_history_wrapper"):
                    with gr.Column(scale=1):
                        with gr.Row(elem_id="agent_scheduler_history_actions", elem_classes="flex-row"):
                            gr.Button(
                                "Requeue Failed",
                                elem_id="agent_scheduler_action_requeue",
                                variant="primary",
                            )
                            gr.Button(
                                "Refresh",
                                elem_id="agent_scheduler_action_refresh_history",
                                elem_classes="agent_scheduler_action_refresh",
                                variant="secondary",
                            )
                            gr.Button(
                                "Clear",
                                elem_id="agent_scheduler_action_clear_history",
                                variant="stop",
                            )

                            with gr.Row(elem_classes=["agent_scheduler_filter_container", "flex-row", "ml-auto"]):
                                status = gr.Dropdown(
                                    elem_id="agent_scheduler_status_filter",
                                    choices=task_filter_choices,
                                    value="All",
                                    show_label=False,
                                    min_width=0,
                                )
                                gr.Textbox(
                                    max_lines=1,
                                    placeholder="Search",
                                    label="Search",
                                    show_label=False,
                                    min_width=0,
                                    elem_id="agent_scheduler_action_search_history",
                                )
                        gr.HTML(
                            f'<div id="agent_scheduler_history_tasks_grid" class="ag-theme-gradio" data-page-size="{grid_page_size}"></div>'
                        )
                    with gr.Column(scale=1, elem_id="agent_scheduler_history_results"):
                        galerry = gr.Gallery(
                            elem_id="agent_scheduler_history_gallery",
                            label="Output",
                            show_label=False,
                            columns=2,
                            preview=True,
                            object_fit="contain",
                        )
                        with gr.Row(
                            elem_id="agent_scheduler_history_result_actions",
                            visible=False,
                        ) as result_actions:
                            if is_sdnext:
                                with gr.Group():
                                    save = ToolButton(
                                        "💾",
                                        elem_id="agent_scheduler_save",
                                        tooltip=f"Save the image to a dedicated directory ({shared.opts.outdir_save}).",
                                    )
                                    save_zip = None
                            else:
                                save = ToolButton(
                                    "💾",
                                    elem_id="agent_scheduler_save",
                                    tooltip=f"Save the image to a dedicated directory ({shared.opts.outdir_save}).",
                                )
                                save_zip = ToolButton(
                                    "🗃️",
                                    elem_id="agent_scheduler_save_zip",
                                    tooltip=f"Save zip archive with images to a dedicated directory ({shared.opts.outdir_save})",
                                )
                            send_to_buttons = create_send_to_buttons()
                        with gr.Group():
                            generation_info = gr.Textbox(visible=False, elem_id=f"agent_scheduler_generation_info")
                            infotext = gr.TextArea(
                                label="Generation Info",
                                elem_id=f"agent_scheduler_history_infotext",
                                interactive=False,
                                visible=True,
                                lines=3,
                            )
                            download_files = gr.File(
                                None,
                                file_count="multiple",
                                interactive=False,
                                show_label=False,
                                visible=False,
                                elem_id=f"agent_scheduler_download_files",
                            )
                            html_log = gr.HTML(elem_id=f"agent_scheduler_html_log", elem_classes="html-log")
                            selected_task = gr.Textbox(
                                elem_id="agent_scheduler_history_selected_task",
                                visible=False,
                                show_label=False,
                            )
                            selected_image_id = gr.Textbox(
                                elem_id="agent_scheduler_history_selected_image",
                                visible=False,
                                show_label=False,
                            )

        # register event handlers
        # 註冊前端互動的事件處理器 / Register the frontend interaction event handlers
        status.change(
            fn=lambda x: None,
            _js="agent_scheduler_status_filter_changed",
            inputs=[status],
        )
        save.click(
            fn=lambda x, y, z: call_queue.wrap_gradio_call(save_files)(x, y, False, int(z)),
            _js="(x, y, z) => [x, y, selected_gallery_index()]",
            inputs=[generation_info, galerry, infotext],
            outputs=[download_files, html_log],
            show_progress=False,
        )
        if save_zip:
            save_zip.click(
                fn=lambda x, y, z: call_queue.wrap_gradio_call(save_files)(x, y, True, int(z)),
                _js="(x, y, z) => [x, y, selected_gallery_index()]",
                inputs=[generation_info, galerry, infotext],
                outputs=[download_files, html_log],
            )
        selected_task.change(
            fn=lambda x: get_task_results(x, None),
            inputs=[selected_task],
            outputs=[infotext, result_actions, galerry, generation_info, download_files, html_log],
        )
        selected_image_id.change(
            fn=lambda x, y: get_task_results(x, image_idx=int(y)),
            inputs=[selected_task, selected_image_id],
            outputs=[infotext, result_actions],
        )
        # 註冊「送往其他分頁」的貼上參數按鈕 / Register the "send to" paste-params buttons
        try:
            for paste_tabname, paste_button in send_to_buttons.items():
                register_paste_params_button(
                    ParamBinding(
                        paste_button=paste_button,
                        tabname=paste_tabname,
                        source_text_component=infotext,
                        source_image_component=galerry,
                    )
                )
        # 貼上按鈕註冊失敗時靜默忽略，不影響主 UI / Silently ignore failures so the main UI still works
        except:
            pass

    return [(scheduler_tab, "Agent Scheduler", "agent_scheduler")]


def on_ui_settings():
    # 註冊所有 Agent Scheduler 的設定項目 / Register all Agent Scheduler settings options
    section = ("agent_scheduler", "Agent Scheduler")
    shared.opts.add_option(
        "queue_paused",
        shared.OptionInfo(
            False,
            "Disable queue auto-processing",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_button_hide_checkpoint",
        shared.OptionInfo(
            True,
            "Hide the custom checkpoint dropdown",
            gr.Checkbox,
            {},
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_button_placement",
        shared.OptionInfo(
            placement_under_generate,
            "Queue button placement",
            gr.Radio,
            lambda: {
                "choices": [
                    placement_under_generate,
                    placement_between_prompt_and_generate,
                ]
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_ui_placement",
        shared.OptionInfo(
            ui_placement_as_tab,
            "Task queue UI placement",
            gr.Radio,
            lambda: {
                "choices": [
                    ui_placement_as_tab,
                    ui_placement_append_to_main,
                ]
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_history_retention_days",
        shared.OptionInfo(
            "30 days",
            "Auto delete queue history (bookmarked tasks excluded)",
            gr.Radio,
            lambda: {
                "choices": list(task_history_retenion_map.keys()),
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_automatic_requeue_failed_task",
        shared.OptionInfo(
            False,
            "Auto requeue failed tasks",
            gr.Checkbox,
            {},
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_grid_page_size",
        shared.OptionInfo(
            0,
            "Task list page size (0 for auto)",
            gr.Slider,
            {"minimum": 0, "maximum": 200, "step": 1},
            section=section,
        ),
    )

    def enqueue_keyboard_shortcut(disabled: bool, modifiers, key_code: str):
        # 根據勾選的修飾鍵與按鍵計算出快捷鍵字串 / Compute the shortcut string from modifiers and the chosen key
        if disabled:
            modifiers.insert(0, "Disabled")

        shortcut = "+".join(sorted(modifiers) + [enqueue_key_codes[key_code]])

        # 停用時連帶停用修飾鍵與按鍵的編輯 / When disabled, also disable editing of modifiers and key
        return (
            shortcut,
            gr.CheckboxGroup.update(interactive=not disabled),
            gr.Dropdown.update(interactive=not disabled),
        )

    def enqueue_keyboard_shortcut_ui(**_kwargs):
        # Enqueue 快捷鍵的設定 UI 元件 / Settings UI for the Enqueue keyboard shortcut
        value = _kwargs.get("value", enqueue_default_hotkey)
        parts = value.split("+")
        key = parts.pop()
        key_code_value = [k for k, v in enqueue_key_codes.items() if v == key]
        modifiers = [m for m in parts if m in enqueue_key_modifiers]
        disabled = "Disabled" in value

        with gr.Group(elem_id="enqueue_keyboard_shortcut_wrapper"):
            modifiers = gr.CheckboxGroup(
                enqueue_key_modifiers,
                value=modifiers,
                label="Enqueue keyboard shortcut",
                elem_id="enqueue_keyboard_shortcut_modifiers",
                interactive=not disabled,
            )
            key_code = gr.Dropdown(
                choices=list(enqueue_key_codes.keys()),
                value="E" if len(key_code_value) == 0 else key_code_value[0],
                elem_id="enqueue_keyboard_shortcut_key",
                label="Key",
                interactive=not disabled,
            )
            shortcut = gr.Textbox(**_kwargs)
            disable = gr.Checkbox(
                value=disabled,
                elem_id="enqueue_keyboard_shortcut_disable",
                label="Disable keyboard shortcut",
            )

        modifiers.change(
            fn=enqueue_keyboard_shortcut,
            inputs=[disable, modifiers, key_code],
            outputs=[shortcut, modifiers, key_code],
        )
        key_code.change(
            fn=enqueue_keyboard_shortcut,
            inputs=[disable, modifiers, key_code],
            outputs=[shortcut, modifiers, key_code],
        )
        disable.change(
            fn=enqueue_keyboard_shortcut,
            inputs=[disable, modifiers, key_code],
            outputs=[shortcut, modifiers, key_code],
        )

        return shortcut

    shared.opts.add_option(
        "queue_keyboard_shortcut",
        shared.OptionInfo(
            enqueue_default_hotkey,
            "Enqueue keyboard shortcut",
            enqueue_keyboard_shortcut_ui,
            {
                "interactive": False,
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_completion_action",
        shared.OptionInfo(
            "Do nothing",
            "Action after queue completion",
            gr.Radio,
            lambda: {
                "choices": completion_action_choices,
            },
            section=section,
        ),
    )


def on_app_started(block: gr.Blocks, app):
    # app 啟動時初始化任務執行器、註冊 API 與清理回呼 / On app start, init the runner, register APIs and cleanup hooks
    global task_runner
    task_runner = get_instance(block)
    task_runner.execute_pending_tasks_threading()
    regsiter_apis(app, task_runner)
    # 任務清空後自動刪除過舊的歷史 / Auto-prune old history whenever tasks are cleared
    task_runner.on_task_cleared(lambda: remove_old_tasks())

    # 當設定為併入主 UI 時，把分頁內容掛到主介面上 / When configured to append to main UI, mount the tab into the main UI
    if getattr(shared.opts, "queue_ui_placement", "") == ui_placement_append_to_main and block:
        with block:
            with block.children[1]:
                # 暫存已註冊的貼上綁定，避免掛載流程複製時重複 / Stash registered bindings so mounting doesn't duplicate them
                bindings = registered_param_bindings.copy()
                registered_param_bindings.clear()
                on_ui_tab()
                connect_paste_params_buttons()
                registered_param_bindings.extend(bindings)


# 若採用獨立分頁（預設），以 on_ui_tabs 註冊 / If using a standalone tab (default), register via on_ui_tabs
if getattr(shared.opts, "queue_ui_placement", "") != ui_placement_append_to_main:
    script_callbacks.on_ui_tabs(on_ui_tab)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)
