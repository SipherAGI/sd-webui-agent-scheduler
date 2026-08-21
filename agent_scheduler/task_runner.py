import os
import ctypes
import json
import subprocess
import time
import traceback
import threading
import gradio as gr

from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Any, Callable, Union, Optional, List, Dict
from fastapi import FastAPI
from PIL import Image

from modules import progress, shared, script_callbacks
from modules.call_queue import queue_lock, wrap_gradio_call
from modules.txt2img import txt2img
from modules.img2img import img2img
from modules.api.api import Api
from modules.api.models import (
    StableDiffusionTxt2ImgProcessingAPI,
    StableDiffusionImg2ImgProcessingAPI,
)
from modules.devices import torch_gc

from .db import TaskStatus, Task, task_manager
from .helpers import (
    log,
    detect_control_net,
    get_component_by_elem_id,
    get_dict_attribute,
    is_windows,
    is_macos,
    _exit,
)
from .task_helpers import (
    encode_image_to_base64,
    serialize_img2img_image_args,
    deserialize_img2img_image_args,
    serialize_script_args,
    deserialize_script_args,
    serialize_api_task_args,
    map_ui_task_args_list_to_named_args,
    map_named_args_to_ui_task_args_list,
)
from .shared_opts_backup import SharedOptsBackup
from pathlib import Path

class OutOfMemoryError(Exception):
    def __init__(self, message="CUDA out of memory") -> None:
        self.message = message
        super().__init__(message)


class FakeRequest:
    def __init__(self, username: str = None):
        self.username = username


class ParsedTaskArgs(BaseModel):
    is_ui: bool
    named_args: Dict[str, Any]
    script_args: List[Any]
    checkpoint: Optional[str] = None
    vae: Optional[str] = None


class TaskRunner:
    instance = None

    def __init__(self, UiControlNetUnit=None, block: gr.Blocks=None):
        self.UiControlNetUnit = UiControlNetUnit

        self.__total_pending_tasks: int = 0
        self.__current_thread: threading.Thread = None
        self.__api = Api(FastAPI(), queue_lock)

        self.__block = block

        self.__saved_images_path: List[str] = []
        script_callbacks.on_image_saved(self.__on_image_saved)

        self.script_callbacks = {
            "task_registered": [],
            "task_started": [],
            "task_finished": [],
            "task_cleared": [],
        }

        # Mark this to True when reload UI
        self.dispose = False
        self.interrupted = None

        if TaskRunner.instance is not None:
            raise Exception("TaskRunner instance already exists")
        TaskRunner.instance = self

    @property
    def current_task_id(self) -> Union[str, None]:
        return progress.current_task

    @property
    def is_executing_task(self) -> bool:
        return self.__current_thread and self.__current_thread.is_alive()

    @property
    def paused(self) -> bool:
        return getattr(shared.opts, "queue_paused", False)

    def __serialize_ui_task_args(
        self,
        is_img2img: bool,
        *args,
        checkpoint: str = None,
        vae: str = None,
        request: gr.Request = None,
    ):
        named_args, script_args = map_ui_task_args_list_to_named_args(list(args), is_img2img)

        # loop through named_args and serialize images
        if is_img2img:
            serialize_img2img_image_args(named_args)

        if "request" in named_args:
            named_args["request"] = {"username": request.username}

        params = json.dumps(
            {
                "args": named_args,
                "checkpoint": checkpoint,
                "vae": vae,
                "is_ui": True,
                "is_img2img": is_img2img,
            }
        )
        script_params = serialize_script_args(script_args)

        return (params, script_params)

    def __serialize_api_task_args(
        self,
        is_img2img: bool,
        checkpoint: str = None,
        vae: str = None,
        **api_args,
    ):
        named_args = serialize_api_task_args(api_args, is_img2img, checkpoint=checkpoint, vae=vae)
        checkpoint = get_dict_attribute(named_args, "override_settings.sd_model_checkpoint", None)
        script_args = named_args.pop("script_args", [])

        params = json.dumps(
            {
                "args": named_args,
                "checkpoint": checkpoint,
                "is_ui": False,
                "is_img2img": is_img2img,
            }
        )
        script_params = serialize_script_args(script_args)
        return (params, script_params)

    def __deserialize_ui_task_args(
        self,
        is_img2img: bool,
        named_args: Dict,
        script_args: List,
        checkpoint: str = None,
        vae: str = None,
    ):
        """
        Deserialize UI task arguments
        In-place update named_args and script_args
        """

        # Apply checkpoint override
        if checkpoint is not None:
            override: List[str] = named_args.get("override_settings_texts", [])
            override = [x for x in override if not x.startswith("Model hash: ")]
            if checkpoint != "System":
                override.append("Model hash: " + checkpoint)
            named_args["override_settings_texts"] = override

        # Apply VAE override
        if vae is not None:
            override: List[str] = named_args.get("override_settings_texts", [])
            override = [x for x in override if not x.startswith("VAE: ")]
            override.append("VAE: " + vae)
            named_args["override_settings_texts"] = override

        # A1111 1.5.0-RC has new request field
        if "request" in named_args:
            named_args["request"] = FakeRequest(**named_args["request"])

        # loop through image_args and deserialize images
        if is_img2img:
            deserialize_img2img_image_args(named_args)

        # loop through script_args and deserialize images
        script_args = deserialize_script_args(script_args, self.UiControlNetUnit)

        return (named_args, script_args)

    def __deserialize_api_task_args(
        self,
        is_img2img: bool,
        named_args: Dict,
        script_args: List,
        checkpoint: str = None,
        vae: str = None,
    ):
        # Apply checkpoint override
        if checkpoint is not None:
            override: Dict = named_args.get("override_settings", {})
            if checkpoint != "System":
                override["sd_model_checkpoint"] = checkpoint
            else:
                override.pop("sd_model_checkpoint", None)
            named_args["override_settings"] = override

        # Apply VAE override
        if vae is not None:
            override: Dict = named_args.get("override_settings", {})
            override["sd_vae"] = vae
            named_args["override_settings"] = override

        # load images from disk
        if is_img2img:
            init_images = named_args.get("init_images")
            for i, img in enumerate(init_images):
                if isinstance(img, str) and os.path.isfile(img):
                    image = Image.open(img)
                    init_images[i] = encode_image_to_base64(image)

        # force image saving
        named_args.update({"save_images": True, "send_images": False})

        script_args = deserialize_script_args(script_args)
        return (named_args, script_args)

    def parse_task_args(self, task: Task, deserialization: bool = True):
        parsed: Dict[str, Any] = json.loads(task.params)

        is_ui = parsed.get("is_ui", True)
        is_img2img = parsed.get("is_img2img", None)
        checkpoint = parsed.get("checkpoint", None)
        vae = parsed.get("vae", None)
        named_args: Dict[str, Any] = parsed["args"]
        script_args: List[Any] = parsed.get("script_args", task.script_params)

        if is_ui and deserialization:
            named_args, script_args = self.__deserialize_ui_task_args(
                is_img2img, named_args, script_args, checkpoint=checkpoint, vae=vae
            )
        elif deserialization:
            named_args, script_args = self.__deserialize_api_task_args(
                is_img2img, named_args, script_args, checkpoint=checkpoint, vae=vae
            )
        else:
            # ignore script_args if not deserialization
            script_args = []

        return ParsedTaskArgs(
            is_ui=is_ui,
            named_args=named_args,
            script_args=script_args,
            checkpoint=checkpoint,
            vae=vae,
        )

    def register_ui_task(
        self,
        task_id: str,
        is_img2img: bool,
        *args,
        checkpoint: str = None,
        task_name: str = None,
        request: gr.Request = None,
    ):
        progress.add_task_to_queue(task_id)

        vae = getattr(shared.opts, "sd_vae", "Automatic")

        (params, script_args) = self.__serialize_ui_task_args(
            is_img2img, *args, checkpoint=checkpoint, vae=vae, request=request
        )

        task_type = "img2img" if is_img2img else "txt2img"
        task = Task(
            id=task_id,
            name=task_name,
            type=task_type,
            params=params,
            script_params=script_args,
        )
        task_manager.add_task(task)

        self.__run_callbacks("task_registered", task_id, is_img2img=is_img2img, is_ui=True, args=params)
        self.__total_pending_tasks += 1

        return task

    def register_api_task(
        self,
        task_id: str,
        api_task_id: str,
        is_img2img: bool,
        args: Dict,
        checkpoint: str = None,
        vae: str = None,
    ):
        progress.add_task_to_queue(task_id)

        (params, script_params) = self.__serialize_api_task_args(is_img2img, checkpoint=checkpoint, vae=vae, **args)

        task_type = "img2img" if is_img2img else "txt2img"
        task = Task(
            id=task_id,
            api_task_id=api_task_id,
            type=task_type,
            params=params,
            script_params=script_params,
        )
        task_manager.add_task(task)

        self.__run_callbacks("task_registered", task_id, is_img2img=is_img2img, is_ui=False, args=params)
        self.__total_pending_tasks += 1

        return task

    def execute_task(self, task: Task, get_next_task: Callable[[], Task]):
        while True:
            if self.dispose:
                break

            if progress.current_task is None:
                task_id = task.id
                is_img2img = task.type == "img2img"
                # gr.Error(f"[AgentScheduler] Executing task {task_id}")
                log.info(f"[AgentScheduler] Executing task {task_id}")

                task_args = self.parse_task_args(task)
                task_meta = {
                    "is_img2img": is_img2img,
                    "is_ui": task_args.is_ui,
                    "task": task,
                }

                self.interrupted = None
                self.__saved_images_path = []
                self.__run_callbacks("task_started", task_id, **task_meta)

                shared_opts_backup = SharedOptsBackup(shared.opts)

                shared_opts_backup.set_shared_opts(samples_save=True, grid_save=True)

                def change_output_dir():
                    if is_img2img:
                        key_samples = "outdir_img2img_samples"
                        key_grids = "outdir_img2img_grids"
                    else:
                        key_samples = "outdir_txt2img_samples"
                        key_grids = "outdir_txt2img_grids"

                    outdir_path_samples_old = Path(shared_opts_backup.get_backup_value(key_samples))

                    outdir_path_root_top = outdir_path_samples_old.joinpath('..')
                    outdir_path_root = outdir_path_root_top.joinpath('agent-scheduler')

                    outdir_path_root_task = outdir_path_root

                    save_to_dirs = False
                    if save_to_dirs:
                        directories_filename_pattern_new = "[datetime<%Y-%m-%d_%H-%M-%S>]_" + str(task_id)

                        shared_opts_backup.set_shared_opts_core("directories_filename_pattern",
                                                                directories_filename_pattern_new)
                    else:
                        outdir_label = time.strftime("%Y-%m-%d/%H-%M-%S") + '_' + str(task_id)

                        outdir_path_root_task = outdir_path_root.joinpath(outdir_label)

                    outdir_path_samples_new = outdir_path_root_task.joinpath(outdir_path_samples_old.name)

                    def _output_to_root_task(key: str):
                        name = Path(shared_opts_backup.get_backup_value(key)).name
                        outdir = outdir_path_root_task.joinpath(name)
                        shared_opts_backup.set_shared_opts_core(key, outdir)
                        os.makedirs(outdir, exist_ok=True)

                    shared_opts_backup.set_shared_opts_core(key_samples, outdir_path_samples_new)
                    os.makedirs(outdir_path_samples_new, exist_ok=True)

                    _output_to_root_task(key_grids)
                    _output_to_root_task("outdir_init_images")
                    _output_to_root_task("ad_save_images_dir")

                    shared_opts_backup.set_shared_opts_core("grid_only_if_multiple", True)
                    shared_opts_backup.set_shared_opts_core("grid_prevent_empty_spots", True)

                    shared_opts_backup.set_shared_opts_core("save_to_dirs", save_to_dirs)
                    shared_opts_backup.set_shared_opts_core("grid_save_to_dirs", save_to_dirs)

                    control_net_detectedmap_dir = Path("..").joinpath("detected_maps")

                    shared_opts_backup.set_shared_opts_core("control_net_detectedmap_dir", control_net_detectedmap_dir)
                    shared_opts_backup.set_shared_opts_core("control_net_detectmap_autosaving", True)

                    shared_opts_backup.set_shared_opts_core("save_mask", True)
                    shared_opts_backup.set_shared_opts_core("save_mask_composite", True)

                    shared_opts_backup.set_shared_opts_core("dp_write_prompts_to_file", False)

                    shared_opts_backup.set_shared_opts_core("enable_pnginfo", True)
                    shared_opts_backup.set_shared_opts_core("save_txt", True)

                    shared_opts_backup.set_shared_opts_core("save_images_before_highres_fix", True)

                    shared_opts_backup.set_shared_opts_core("save_init_img", is_img2img)

                    if is_img2img:
                        shared_opts_backup.set_shared_opts_core("ad_save_previews", True)
                        shared_opts_backup.set_shared_opts_core("ad_save_images_before", True)

                change_output_dir()

                res = self.__execute_task(task_id, is_img2img, task_args)

                shared_opts_backup.restore_shared_opts()

                if not res or isinstance(res, Exception):
                    if isinstance(res, OutOfMemoryError):
                        gr.Error(f"[AgentScheduler] Task {task_id} failed: CUDA OOM. Queue will be paused.")
                        log.error(f"[AgentScheduler] Task {task_id} failed: CUDA OOM. Queue will be paused.")
                        shared.opts.queue_paused = True
                    else:
                        gr.Error(f"[AgentScheduler] Task {task_id} failed: {res}")
                        log.error(f"[AgentScheduler] Task {task_id} failed: {res}")
                        log.debug(traceback.format_exc())

                    if getattr(shared.opts, "queue_automatic_requeue_failed_task", False):
                        log.info(f"[AgentScheduler] Requeue task {task_id}")
                        task.status = TaskStatus.PENDING
                        task.priority = int(datetime.now(timezone.utc).timestamp() * 1000)
                        task_manager.update_task(task)
                    else:
                        task.status = TaskStatus.FAILED
                        task.result = str(res) if res else None
                        task_manager.update_task(task)
                        self.__run_callbacks("task_finished", task_id, status=TaskStatus.FAILED, **task_meta)
                else:
                    is_interrupted = self.interrupted == task_id
                    if is_interrupted:
                        log.info(f"\n[AgentScheduler] Task {task.id} interrupted")
                        task.status = TaskStatus.INTERRUPTED
                        task_manager.update_task(task)
                        self.__run_callbacks(
                            "task_finished",
                            task_id,
                            status=TaskStatus.INTERRUPTED,
                            **task_meta,
                        )
                    else:
                        geninfo = json.loads(res)
                        result = {
                            "images": self.__saved_images_path.copy(),
                            "geninfo": geninfo,
                        }

                        task.status = TaskStatus.DONE
                        task.result = json.dumps(result)
                        task_manager.update_task(task)
                        self.__run_callbacks(
                            "task_finished",
                            task_id,
                            status=TaskStatus.DONE,
                            result=result,
                            **task_meta,
                        )
                        # gr.Info(f"[AgentScheduler] Task {task_id} finished")

                self.__saved_images_path = []
            else:
                time.sleep(2)
                continue

            task = get_next_task()
            if not task:
                if not self.paused:
                    time.sleep(1)
                    self.__on_completed()
                break

    def execute_pending_tasks_threading(self):
        if self.paused:
            gr.Warning("[AgentScheduler] Runner is paused")
            log.info("[AgentScheduler] Runner is paused")
            return

        if self.is_executing_task:
            # gr.Warning("[AgentScheduler] Runner already started")
            log.info("[AgentScheduler] Runner already started")
            return

        pending_task = self.__get_pending_task()
        if pending_task:
            # Start the infinite loop in a separate thread
            self.__current_thread = threading.Thread(
                target=self.execute_task,
                args=(
                    pending_task,
                    self.__get_pending_task,
                ),
            )
            self.__current_thread.daemon = True
            self.__current_thread.start()

    def __execute_task(self, task_id: str, is_img2img: bool, task_args: ParsedTaskArgs):
        if task_args.is_ui:
            ui_args = map_named_args_to_ui_task_args_list(task_args.named_args, task_args.script_args, is_img2img)

            return self.__execute_ui_task(task_id, is_img2img, *ui_args)
        else:
            return self.__execute_api_task(
                task_id,
                is_img2img,
                script_args=task_args.script_args,
                **task_args.named_args,
            )

    def __execute_ui_task(self, task_id: str, is_img2img: bool, *args):
        func = wrap_gradio_call(img2img if is_img2img else txt2img, add_stats=True)

        with queue_lock:
            shared.state.begin()
            progress.start_task(task_id)

            res = None
            try:
                result = func(*args)
                if result[0] is None and hasattr(shared.state, "oom") and shared.state.oom:
                    res = OutOfMemoryError()
                elif "CUDA out of memory" in result[2]:
                    res = OutOfMemoryError()
                else:
                    res = result[1]


            except Exception as e:
                res = e
            finally:
                progress.finish_task(task_id)

            shared.state.end()

            return res

    def __execute_api_task(self, task_id: str, is_img2img: bool, **kwargs):
        progress.start_task(task_id)

        res = None
        try:
            result = (
                self.__api.img2imgapi(StableDiffusionImg2ImgProcessingAPI(**kwargs))
                if is_img2img
                else self.__api.text2imgapi(StableDiffusionTxt2ImgProcessingAPI(**kwargs))
            )
            res = result.info
        except Exception as e:
            if "CUDA out of memory" in str(e):
                res = OutOfMemoryError()
            else:
                res = e
        finally:
            progress.finish_task(task_id)

        return res

    def __get_pending_task(self):
        if self.dispose:
            return None

        if self.paused:
            log.info("[AgentScheduler] Runner is paused")
            return None

        # # delete task that are too old
        # retention_days = 30
        # if (
        #     getattr(shared.opts, "queue_history_retention_days", None)
        #     and shared.opts.queue_history_retention_days in task_history_retenion_map
        # ):
        #     retention_days = task_history_retenion_map[shared.opts.queue_history_retention_days]

        # if retention_days > 0:
        #     deleted_rows = task_manager.delete_tasks(before=datetime.now() - timedelta(days=retention_days))
        #     if deleted_rows > 0:
        #         log.debug(f"[AgentScheduler] Deleted {deleted_rows} tasks older than {retention_days} days")

        self.__total_pending_tasks = task_manager.count_tasks(status="pending")

        # get more task if needed
        if self.__total_pending_tasks > 0:
            log.info(f"[AgentScheduler] Total pending tasks: {self.__total_pending_tasks}")
            pending_tasks = task_manager.get_tasks(status="pending", limit=1)
            if len(pending_tasks) > 0:
                return pending_tasks[0]
        else:
            log.info("[AgentScheduler] Task queue is empty")
            self.__run_callbacks("task_cleared")

    def __on_image_saved(self, data: script_callbacks.ImageSaveParams):
        if self.current_task_id is None:
            return

        outpath_grids = shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids
        if data.filename.startswith(outpath_grids):
            self.__saved_images_path.insert(0, data.filename)
        else:
            self.__saved_images_path.append(data.filename)
    
    def __on_completed(self):
        action = getattr(shared.opts, "queue_completion_action", "Do nothing")

        if action == "Do nothing":
            return

        command = None
        if action == "Shut down":
            log.info("[AgentScheduler] Shutting down...")
            if is_windows:
                command = ["shutdown", "/s", "/hybrid", "/t", "0"]
            elif is_macos:
                command = ["osascript", "-e", 'tell application "Finder" to shut down']
            else:
                command = ["systemctl", "poweroff"]
        elif action == "Restart":
            log.info("[AgentScheduler] Restarting...")
            if is_windows:
                command = ["shutdown", "/r", "/t", "0"]
            elif is_macos:
                command = ["osascript", "-e", 'tell application "Finder" to restart']
            else:
                command = ["systemctl", "reboot"]
        elif action == "Sleep":
            log.info("[AgentScheduler] Sleeping...")
            if is_windows:
                if not ctypes.windll.PowrProf.SetSuspendState(False, False, False):
                    print(f"Couldn't sleep: {ctypes.GetLastError()}")
            elif is_macos:
                command = ["osascript", "-e", 'tell application "Finder" to sleep']
            else:
                command = ["sh", "-c", 'systemctl hybrid-sleep || (echo "Couldn\'t hybrid sleep, will try to suspend instead: $?"; systemctl suspend)']
        elif action == "Hibernate":
            log.info("[AgentScheduler] Hibernating...")
            if is_windows:
                command = ["shutdown", "/h"]
            elif is_macos:
                command = ["osascript", "-e", 'tell application "Finder" to sleep']
            else:
                command = ["systemctl", "hibernate"]
        elif action == "Stop webui":
            log.info("[AgentScheduler] Stopping webui...")
            _exit(0)
        elif action == "Restart webui":
            log.info("[AgentScheduler] Restarting webui...")
            with self.__block:
                gr.HTML("""
    <script type="text/javascript">
    console.info(`Restart webui...after 5sec`);
    
    var requestPing = function() {
        requestGet("./internal/ping", {}, function(data) {
            location.reload();
        }, function() {
            console.info(`Waiting webui start...`);
            setTimeout(requestPing, 500);
        });
    };
    
    setTimeout(requestPing, 5000);
    </script>
                """)
            shared.state.request_restart()

        if command:
            subprocess.Popen(command)

        if action in {"Shut down", "Restart"}:
            _exit(0)

    def on_task_registered(self, callback: Callable):
        """Callback when a task is registered

        Callback signature: callback(task_id: str, is_img2img: bool, is_ui: bool, args: Dict)
        """

        self.script_callbacks["task_registered"].append(callback)

    def on_task_started(self, callback: Callable):
        """Callback when a task is started

        Callback signature: callback(task_id: str, is_img2img: bool, is_ui: bool)
        """

        self.script_callbacks["task_started"].append(callback)

    def on_task_finished(self, callback: Callable):
        """Callback when a task is finished

        Callback signature: callback(task_id: str, is_img2img: bool, is_ui: bool, status: TaskStatus, result: Dict)
        """

        self.script_callbacks["task_finished"].append(callback)

    def on_task_cleared(self, callback: Callable):
        self.script_callbacks["task_cleared"].append(callback)

    def __run_callbacks(self, name: str, *args, **kwargs):
        # print(f"[AgentScheduler] __run_callbacks {name}...")

        for callback in self.script_callbacks[name]:
            callback(*args, **kwargs)

        if name == "task_finished" or name == "task_started":
            print(f"[AgentScheduler] {name} torch_gc...")
            try:
                torch_gc()
            except Exception as e:
                print(e)


def get_instance(block) -> TaskRunner:
    if TaskRunner.instance is None:
        if block is not None:
            txt2img_submit_button = get_component_by_elem_id(block, "txt2img_generate")
            UiControlNetUnit = detect_control_net(block, txt2img_submit_button)
            TaskRunner(UiControlNetUnit, block)
        else:
            TaskRunner()

        if not hasattr(script_callbacks, "on_before_reload"):
            log.warning(
                "*****************************************************************************************\n"
                + "[AgentScheduler] YOUR SD WEBUI IS OUTDATED AND AGENT SCHEDULER WILL NOT WORKING PROPERLY."
                + "*****************************************************************************************\n",
            )
        else:

            def on_before_reload():
                # Tell old instance to stop
                TaskRunner.instance.dispose = True
                # force recreate the instance
                TaskRunner.instance = None

            script_callbacks.on_before_reload(on_before_reload)

    return TaskRunner.instance
