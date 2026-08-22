"""Agent Scheduler 的 FastAPI 介面 / Agent Scheduler FastAPI endpoints.

對外提供佇列管理、歷史查詢、任務 CRUD、結果下載與佇列控制等 HTTP API，
並在任務完成時呼叫使用者指定的 callback。
Exposes HTTP APIs for queue management, history queries, task CRUD, result
downloads and queue control, calling a user callback when a task finishes.
"""

import io
import os
import json
import base64
import requests
import threading
from uuid import uuid4
from zipfile import ZipFile
from pathlib import Path
from secrets import compare_digest
from typing import Optional, Dict, List
from datetime import datetime, timezone
from collections import defaultdict
from gradio.routes import App
from PIL import Image
from fastapi import Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from modules import shared, progress, sd_models, sd_samplers

from .db import Task, TaskStatus, task_manager
from .models import (
    Txt2ImgApiTaskArgs,
    Img2ImgApiTaskArgs,
    QueueTaskResponse,
    QueueStatusResponse,
    HistoryResponse,
    TaskModel,
    UpdateTaskArgs,
)
from .task_runner import TaskRunner
from .helpers import log, request_with_retry
from .task_helpers import encode_image_to_base64, img2img_image_args_by_mode


def api_callback(callback_url: str, task_id: str, status: TaskStatus, images: list):
    # 將任務結果圖片以 multipart 表單 POST 回使用者的 callback 網址 / POST result images back to the user's callback URL as multipart form data
    files = []
    for img in images:
        img_path = Path(img)
        ext = img_path.suffix.lower()
        content_type = f"image/{ext[1:]}"
        files.append(
            (
                "files",
                (img_path.name, open(os.path.abspath(img), "rb"), content_type),
            )
        )

    return requests.post(
        callback_url,
        timeout=5,
        data={"task_id": task_id, "status": status.value},
        files=files,
    )


def on_task_finished(
    task_id: str,
    task: Task,
    status: TaskStatus = None,
    result: dict = None,
    **_,
):
    # 任務完成回呼：若有設定 callback 則推送結果 / Task-finished hook: push results if a callback URL is configured
    # handle api task callback
    if not task.api_task_callback:
        return

    # 以 lambda 封裝上傳邏輯以便重試 / Wrap upload in a lambda so it can be retried
    upload = lambda: api_callback(
        task.api_task_callback,
        task_id=task_id,
        status=status,
        images=result["images"],
    )

    request_with_retry(upload)


def regsiter_apis(app: App, task_runner: TaskRunner):
    """註冊所有 Agent Scheduler 的 HTTP API / Register all Agent Scheduler HTTP APIs.

    依啟動參數決定是否啟用 Basic Auth，並逐一掛載佇列、歷史、任務與結果相關路由。
    Enables HTTP Basic Auth when configured, then mounts queue/history/task/result routes.
    """
    api_credentials = {}
    deps = None

    def auth(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        # 以 compare_digest 進行常數時間比對，避免時序側信道 / Use constant-time compare_digest to avoid timing side-channels
        if credentials.username in api_credentials:
            if compare_digest(credentials.password, api_credentials[credentials.username]):
                return True

        raise HTTPException(
            status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"}
        )

    # 若啟用了 API 認證，解析帳密並將 auth 依賴套用到各路由 / When API auth is on, parse credentials and apply the auth dependency
    if shared.cmd_opts.api_auth:
        api_credentials = {}

        for cred in shared.cmd_opts.api_auth.split(","):
            user, password = cred.split(":")
            api_credentials[user] = password

        deps = [Depends(auth)]

    log.info("[AgentScheduler] Registering APIs")

    @app.get("/agent-scheduler/v1/samplers", response_model=List[str])
    def get_samplers():
        # 回傳可用的 sampler 名稱清單 / Return the list of available sampler names
        return [sampler[0] for sampler in sd_samplers.all_samplers]

    @app.get("/agent-scheduler/v1/sd-models", response_model=List[str])
    def get_sd_models():
        # 回傳可用的 checkpoint 標題清單 / Return the list of available checkpoint titles
        return [x.title for x in sd_models.checkpoints_list.values()]

    @app.post("/agent-scheduler/v1/queue/txt2img", response_model=QueueTaskResponse, dependencies=deps)
    def queue_txt2img(body: Txt2ImgApiTaskArgs):
        # 接收 txt2img 任務並排入佇列，回傳新任務 id / Accept a txt2img task, enqueue it, and return its new id
        task_id = str(uuid4())
        args = body.dict()
        # 自參數中抽出特殊欄位，剩餘交給執行器 / Extract special fields; the rest goes to the runner
        checkpoint = args.pop("checkpoint", None)
        vae = args.pop("vae", None)
        callback_url = args.pop("callback_url", None)
        task = task_runner.register_api_task(
            task_id,
            api_task_id=None,
            is_img2img=False,
            args=args,
            checkpoint=checkpoint,
            vae=vae,
        )
        # 若有 callback 則記錄到任務，完成時會被呼叫 / Persist the callback URL so it fires on completion
        if callback_url:
            task.api_task_callback = callback_url
            task_manager.update_task(task)

        task_runner.execute_pending_tasks_threading()

        return QueueTaskResponse(task_id=task_id)

    @app.post("/agent-scheduler/v1/queue/img2img", response_model=QueueTaskResponse, dependencies=deps)
    def queue_img2img(body: Img2ImgApiTaskArgs):
        # 接收 img2img 任務並排入佇列，回傳新任務 id / Accept an img2img task, enqueue it, and return its new id
        task_id = str(uuid4())
        args = body.dict()
        checkpoint = args.pop("checkpoint", None)
        vae = args.pop("vae", None)
        callback_url = args.pop("callback_url", None)
        task = task_runner.register_api_task(
            task_id,
            api_task_id=None,
            is_img2img=True,
            args=args,
            checkpoint=checkpoint,
            vae=vae,
        )
        if callback_url:
            task.api_task_callback = callback_url
            task_manager.update_task(task)

        task_runner.execute_pending_tasks_threading()

        return QueueTaskResponse(task_id=task_id)

    def format_task_args(task):
        # 將任務參數整理為前端回傳用的精簡結構 / Trim task args into a compact shape for API responses
        task_args = TaskRunner.instance.parse_task_args(task, deserialization=False)
        named_args = task_args.named_args
        named_args["checkpoint"] = task_args.checkpoint
        # remove unused args to reduce payload size
        # 移除前端不需的大型欄位以縮減回傳體積 / Drop heavy fields the frontend doesn't need to shrink the payload
        named_args.pop("alwayson_scripts", None)
        named_args.pop("script_args", None)
        named_args.pop("init_images", None)
        for image_args in img2img_image_args_by_mode.values():
            for keys in image_args:
                named_args.pop(keys[0], None)
        return named_args

    @app.get("/agent-scheduler/v1/queue", response_model=QueueStatusResponse, dependencies=deps)
    def queue_status_api(limit: int = 20, offset: int = 0):
        # 回傳目前佇列狀態：進行中任務、待處理清單與總數 / Return queue status: running task, pending list, and totals
        current_task_id = progress.current_task
        total_pending_tasks = task_manager.count_tasks(status="pending")
        pending_tasks = task_manager.get_tasks(status=TaskStatus.PENDING, limit=limit, offset=offset)
        position = offset
        parsed_tasks = []
        # 逐筆標註位置與進行中狀態 / Annotate position and "running" status per task
        for task in pending_tasks:
            params = format_task_args(task)
            task_data = task.dict()
            task_data["params"] = params
            if task.id == current_task_id:
                task_data["status"] = "running"

            task_data["position"] = position
            parsed_tasks.append(TaskModel(**task_data))
            position += 1

        return QueueStatusResponse(
            current_task_id=current_task_id,
            pending_tasks=parsed_tasks,
            total_pending_tasks=total_pending_tasks,
            paused=TaskRunner.instance.paused,
        )

    @app.get("/agent-scheduler/v1/export")
    def export_queue(limit: int = 1000, offset: int = 0):
        # 匯出待處理任務為 JSON 清單 / Export pending tasks as a JSON list
        pending_tasks = task_manager.get_tasks(status=TaskStatus.PENDING, limit=limit, offset=offset)
        pending_tasks = [Task.from_table(t).to_json() for t in pending_tasks]
        return pending_tasks

    class StringRequestBody(BaseModel):
        # 匯入時接收的原始 JSON 字串本體 / Raw JSON body received for queue import
        content: str

    @app.post("/agent-scheduler/v1/import")
    def import_queue(queue: StringRequestBody):
        # 匯入任務清單：建立/更新任務並回報成功或失敗 / Import a task list: create/update tasks, report success or failure
        try:
            objList = json.loads(queue.content)
            taskList: List[Task] = []
            # 缺 id 的項目補上新的 uuid，並重置為待處理 / Generate an id for items missing one and reset to pending
            for obj in objList:
                if "id" not in obj or not obj["id"] or obj["id"] == "":
                    obj["id"] = str(uuid4())
                obj["result"] = None
                obj["status"] = TaskStatus.PENDING
                task = Task.from_json(obj)
                taskList.append(task)

            # 已存在則更新，否則新增 / Update if exists, otherwise add
            for task in taskList:
                exists = task_manager.get_task(task.id)
                if exists:
                    task_manager.update_task(task)
                else:
                    task_manager.add_task(task)
            return {"success": True, "message": "Queue imported"}
        # 解析或寫入失敗時回傳失敗訊息 / On parse/write failure, report import failure
        except Exception as e:
            print(e)
            return {"success": False, "message": "Import Failed"}

    @app.get("/agent-scheduler/v1/history", response_model=HistoryResponse, dependencies=deps)
    def history_api(status: str = None, limit: int = 20, offset: int = 0):
        # 查詢歷史任務；依 status 過濾（bookmarked 只撈收藏項） / Query history tasks, filtering by status (bookmarked = only saved items)
        bookmarked = True if status == "bookmarked" else None
        # 未指定或 "all" 時涵蓋已完成/失敗/中斷三類 / Unspecified or "all" covers done/failed/interrupted
        if not status or status == "all" or bookmarked:
            status = [
                TaskStatus.DONE,
                TaskStatus.FAILED,
                TaskStatus.INTERRUPTED,
            ]

        total = task_manager.count_tasks(status=status)
        tasks = task_manager.get_tasks(
            status=status,
            bookmarked=bookmarked,
            limit=limit,
            offset=offset,
            order="desc",
        )
        parsed_tasks = []
        # 逐筆整理成標準的 TaskModel 回傳結構 / Normalize each task into the standard TaskModel response
        for task in tasks:
            params = format_task_args(task)
            task_data = task.dict()
            task_data["params"] = params
            parsed_tasks.append(TaskModel(**task_data))

        return HistoryResponse(
            total=total,
            tasks=parsed_tasks,
        )

    @app.get("/agent-scheduler/v1/task/{id}", dependencies=deps)
    def get_task(id: str):
        # 取得單一任務詳情，含進行中狀態與佇列位置 / Get a single task's details, including running state and queue position
        task = task_manager.get_task(id)
        if task is None:
            return {"success": False, "message": "Task not found"}

        params = format_task_args(task)
        task_data = task.dict()
        task_data["params"] = params
        if task.id == progress.current_task:
            task_data["status"] = "running"
        if task_data["status"] == TaskStatus.PENDING:
            task_data["position"] = task_manager.get_task_position(id)

        return {"success": True, "data": TaskModel(**task_data)}

    @app.get("/agent-scheduler/v1/task/{id}/position", dependencies=deps)
    def get_task_position(id: str):
        # 僅回傳任務的狀態與佇列位置 / Return only the task's status and queue position
        task = task_manager.get_task(id)
        if task is None:
            return {"success": False, "message": "Task not found"}

        position = None if task.status != TaskStatus.PENDING else task_manager.get_task_position(id)
        return {"success": True, "data": {"status": task.status, "position": position}}

    @app.put("/agent-scheduler/v1/task/{id}", dependencies=deps)
    def update_task(id: str, body: UpdateTaskArgs):
        # 更新任務的名稱或參數（checkpoint / 生成參數） / Update a task's name or params (checkpoint / generation args)
        task = task_manager.get_task(id)
        if task is None:
            return {"success": False, "message": "Task not found"}

        should_save = False
        # 名稱有變更才更新 / Only update name when provided
        if body.name is not None:
            task.name = body.name
            should_save = True

        # checkpoint 或 params 有提供時合併進現有參數 / Merge checkpoint/params when either is supplied
        if body.checkpoint or body.params:
            params: Dict = json.loads(task.params)
            if body.checkpoint is not None:
                params["checkpoint"] = body.checkpoint
            if body.checkpoint is not None:
                params["args"].update(body.params)

            task.params = json.dumps(params)
            should_save = True

        if should_save:
            task_manager.update_task(task)

        return {"success": True, "message": "Task updated."}

    @app.post("/agent-scheduler/v1/run/{id}", dependencies=deps, deprecated=True)
    @app.post("/agent-scheduler/v1/task/{id}/run", dependencies=deps)
    def run_task(id: str):
        # 立即執行指定任務（若佇列忙則提升優先權） / Run a task now, or bump its priority if the queue is busy
        if progress.current_task is not None:
            if progress.current_task == id:
                return {"success": False, "message": "Task is running"}
            else:
                # move task up in queue
                # 已在執行其他任務時，把該任務提到最前等待下一輪 / If another task runs, prioritize this one for the next slot
                task_manager.prioritize_task(id, 0)
                return {
                    "success": True,
                    "message": "Task is scheduled to run next",
                }
        else:
            # run task
            # 目前閒置則直接以背景執行緒執行 / Idle: run directly in a background thread
            task = task_manager.get_task(id)
            current_thread = threading.Thread(
                target=TaskRunner.instance.execute_task,
                args=(
                    task,
                    lambda: None,
                ),
            )
            current_thread.daemon = True
            current_thread.start()

            return {"success": True, "message": "Task is executing"}

    @app.post("/agent-scheduler/v1/requeue/{id}", dependencies=deps, deprecated=True)
    @app.post("/agent-scheduler/v1/task/{id}/requeue", dependencies=deps)
    def requeue_task(id: str):
        # 以新 id 重新排入已完成/失敗的任務（保留原設定） / Re-enqueue a finished/failed task under a fresh id, keeping its config
        task = task_manager.get_task(id)
        if task is None:
            return {"success": False, "message": "Task not found"}

        task.id = str(uuid4())
        task.result = None
        task.status = TaskStatus.PENDING
        task.bookmarked = False
        task.name = f"Copy of {task.name}" if task.name else None
        task_manager.add_task(task)
        task_runner.execute_pending_tasks_threading()

        return {"success": True, "message": "Task requeued"}

    @app.post("/agent-scheduler/v1/task/requeue-failed", dependencies=deps)
    def requeue_failed_tasks():
        # 批次將所有失敗任務重新排入佇列 / Batch-requeue every failed task
        failed_tasks = task_manager.get_tasks(status=TaskStatus.FAILED)
        if (len(failed_tasks)) == 0:
            return {"success": False, "message": "No failed tasks"}

        for task in failed_tasks:
            task.status = TaskStatus.PENDING
            task.result = None
            # 以當前時間作為優先權，避免插隊到使用者手動任務前 / Use current time as priority so they don't jump ahead of manual tasks
            task.priority = int(datetime.now(timezone.utc).timestamp() * 1000)
            task_manager.update_task(task)

        return {"success": True, "message": f"Requeued {len(failed_tasks)} failed tasks"}

    @app.post("/agent-scheduler/v1/delete/{id}", dependencies=deps, deprecated=True)
    @app.delete("/agent-scheduler/v1/task/{id}", dependencies=deps)
    def delete_task(id: str):
        # 刪除任務；若正在執行則改為中斷 / Delete a task, or interrupt it if currently running
        if progress.current_task == id:
            shared.state.interrupt()
            task_runner.interrupted = id
            return {"success": True, "message": "Task interrupted"}

        task_manager.delete_task(id)
        return {"success": True, "message": "Task deleted"}

    @app.post("/agent-scheduler/v1/move/{id}/{over_id}", dependencies=deps, deprecated=True)
    @app.post("/agent-scheduler/v1/task/{id}/move/{over_id}", dependencies=deps)
    def move_task(id: str, over_id: str):
        # 調整任務在佇列中的順序（頂端/底端/指定任務之前） / Reorder a task in the queue (top / bottom / before a given task)
        task = task_manager.get_task(id)
        if task is None:
            return {"success": False, "message": "Task not found"}

        # 頂端/底端使用特殊關鍵字，否則插入到目標任務的優先權 / top/bottom use keywords; otherwise insert at the target's priority
        if over_id == "top":
            task_manager.prioritize_task(id, 0)
            return {"success": True, "message": "Task moved to top"}
        elif over_id == "bottom":
            task_manager.prioritize_task(id, -1)
            return {"success": True, "message": "Task moved to bottom"}
        else:
            over_task = task_manager.get_task(over_id)
            if over_task is None:
                return {"success": False, "message": "Task not found"}

            task_manager.prioritize_task(id, over_task.priority)
            return {"success": True, "message": "Task moved"}

    @app.post("/agent-scheduler/v1/bookmark/{id}", dependencies=deps, deprecated=True)
    @app.post("/agent-scheduler/v1/task/{id}/bookmark", dependencies=deps)
    def pin_task(id: str):
        # 收藏任務以免被自動清理 / Bookmark a task so auto-pruning skips it
        task = task_manager.get_task(id)
        if task is None:
            return {"success": False, "message": "Task not found"}

        task.bookmarked = True
        task_manager.update_task(task)
        return {"success": True, "message": "Task bookmarked"}

    @app.post("/agent-scheduler/v1/unbookmark/{id}", dependencies=deps, deprecated=True)
    @app.post("/agent-scheduler/v1/task/{id}/unbookmark")
    def unpin_task(id: str):
        # 取消收藏任務 / Unbookmark a task
        task = task_manager.get_task(id)
        if task is None:
            return {"success": False, "message": "Task not found"}

        task.bookmarked = False
        task_manager.update_task(task)
        return {"success": True, "message": "Task unbookmarked"}

    @app.post("/agent-scheduler/v1/rename/{id}", dependencies=deps, deprecated=True)
    @app.post("/agent-scheduler/v1/task/{id}/rename", dependencies=deps)
    def rename_task(id: str, name: str):
        # 重新命名任務 / Rename a task
        task = task_manager.get_task(id)
        if task is None:
            return {"success": False, "message": "Task not found"}

        task.name = name
        task_manager.update_task(task)
        return {"success": True, "message": "Task renamed."}

    @app.get("/agent-scheduler/v1/results/{id}", dependencies=deps, deprecated=True)
    @app.get("/agent-scheduler/v1/task/{id}/results", dependencies=deps)
    def get_task_results(id: str, zip: Optional[bool] = False):
        # 取得已完成任務的結果圖片與 infotext（可打包成 zip） / Get a finished task's result images and infotext, optionally zipped
        task = task_manager.get_task(id)
        if task is None:
            return {"success": False, "message": "Task not found"}

        # 僅已完成且有結果才可下載 / Only done tasks with results are downloadable
        if task.status != TaskStatus.DONE:
            return {"success": False, "message": f"Task is {task.status}"}

        if task.result is None:
            return {"success": False, "message": "Task result is not available"}

        result: dict = json.loads(task.result)
        infotexts = result.get("infotexts", None)
        # 相容舊格式：從 geninfo 中取得 infotexts / Fall back to geninfo for the older result layout
        if infotexts is None:
            geninfo = result.get("geninfo", {})
            infotexts = geninfo.get("infotexts", defaultdict(lambda: ""))

        if zip:
            zip_buffer = io.BytesIO()

            # Create a new zip file in the in-memory buffer
            # 在記憶體中建立 zip 並把每張圖片寫入 / Build the zip in memory and add each image
            with ZipFile(zip_buffer, "w") as zip_file:
                # Loop through the files in the directory and add them to the zip file
                for image in result["images"]:
                    if Path(image).is_file():
                        zip_file.write(Path(image), Path(image).name)

            # Reset the buffer position to the beginning to avoid truncation issues
            # 將緩衝區指標移回開頭避免截斷 / Rewind the buffer so the stream isn't truncated
            zip_buffer.seek(0)

            # Return the in-memory buffer as a streaming response with the appropriate headers
            # 以串流方式回傳 zip 附件 / Stream the zip back as an attachment
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename=results-{id}.zip"},
            )
        else:
            # 否則以 base64 圖片 + infotext 清單回傳 / Otherwise return base64 images with infotext list
            data = [
                {
                    "image": encode_image_to_base64(Image.open(image)),
                    "infotext": infotexts[i],
                }
                for i, image in enumerate(result["images"])
                if Path(image).is_file()
            ]

            return {"success": True, "data": data}

    @app.post("/agent-scheduler/v1/pause", dependencies=deps, deprecated=True)
    @app.post("/agent-scheduler/v1/queue/pause", dependencies=deps)
    def pause_queue():
        # 暫停佇列自動處理 / Pause automatic queue processing
        shared.opts.queue_paused = True
        return {"success": True, "message": "Queue paused."}

    @app.post("/agent-scheduler/v1/resume", dependencies=deps, deprecated=True)
    @app.post("/agent-scheduler/v1/queue/resume", dependencies=deps)
    def resume_queue():
        # 恢復佇列並立即處理待辦 / Resume the queue and process pending tasks immediately
        shared.opts.queue_paused = False
        TaskRunner.instance.execute_pending_tasks_threading()
        return {"success": True, "message": "Queue resumed."}

    @app.post("/agent-scheduler/v1/queue/clear", dependencies=deps)
    def clear_queue():
        # 清空所有待處理任務 / Clear all pending tasks
        task_manager.delete_tasks(status=TaskStatus.PENDING)
        return {"success": True, "message": "Queue cleared."}

    @app.post("/agent-scheduler/v1/history/clear", dependencies=deps)
    def clear_history():
        # 清空歷史（已完成/失敗/中斷） / Clear history (done/failed/interrupted)
        task_manager.delete_tasks(
            status=[
                TaskStatus.DONE,
                TaskStatus.FAILED,
                TaskStatus.INTERRUPTED,
            ]
        )
        return {"success": True, "message": "History cleared."}

    # 註冊任務完成回呼，用於觸發 API callback / Register the finished hook that fires the API callback
    task_runner.on_task_finished(on_task_finished)
