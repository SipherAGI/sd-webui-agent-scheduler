"""任務排程相關的 pydantic 資料模型 / Task-scheduling related pydantic data models

定義 queue API 與任務佇列所需的請求/回應模型，以及對外曝露的
任務參數模型（Txt2Img / Img2Img）。為相容 A1111 與 Forge Neo
兩套 sd-webui 環境，pydantic v1 / v2 的 config 寫法差異統一由
compat_a1111_forge 套件處理。
Defines request/response models for the queue API and the task-queue,
plus the externally exposed task-argument models (Txt2Img / Img2Img).
To stay compatible with both A1111 and Forge Neo sd-webui, the pydantic
v1/v2 config-style differences are handled centrally by compat_a1111_forge.
"""

from datetime import datetime, timezone
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field

from modules import sd_samplers
from modules.api.models import (
    StableDiffusionTxt2ImgProcessingAPI,
    StableDiffusionImg2ImgProcessingAPI,
)

from agent_scheduler.compat_a1111_forge.pydantic import PYDANTIC_V2, api_task_schema_extra

if PYDANTIC_V2:
    from pydantic import ConfigDict


def convert_datetime_to_iso_8601_with_z_suffix(dt: datetime) -> str:
    """將 datetime 轉為帶 Z 尾碼的 ISO 8601 字串 / Convert datetime to ISO 8601 string with Z suffix

    sd-webui 前端的任務時間通常以毫秒精度、UTC 的 Z 尾碼格式呈現，
    None 時回傳 None 以避免對未設定時間的欄位拋錯。
    sd-webui's frontend typically shows task times in millisecond-precision UTC
    with a Z suffix; None inputs return None so unset time fields don't error.
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if dt else None


def transform_to_utc_datetime(dt: datetime) -> datetime:
    """將任意時區的 datetime 轉換為 UTC / Convert a datetime in any timezone to UTC

    queue 內部統一以 UTC 儲存與比較時間，避免時區錯亂導致排序或顯示錯誤。
    The queue stores and compares times in UTC uniformly to avoid timezone
    confusion in ordering or display.
    """
    return dt.astimezone(tz=timezone.utc)


class QueueStatusAPI(BaseModel):
    """查詢佇列狀態時的分頁參數 / Pagination params for querying queue status

    提供 limit/offset 讓前端以分頁方式拉取任務，避免一次回傳過多資料。
    Provides limit/offset so the frontend can page through tasks instead of
    receiving the entire queue at once.
    """
    limit: Optional[int] = Field(title="Limit", description="The maximum number of tasks to return", default=20)
    offset: Optional[int] = Field(title="Offset", description="The offset of the tasks to return", default=0)


class TaskModel(BaseModel):
    """佇列中單一任務的完整資料模型 / Full data model for a single queued task

    對外曝露給 API 的任務結構，涵蓋識別、類型、狀態、參數與時間等欄位，
    作為 queue 與 history 列表共用的核心模型。
    The task structure exposed to the API, covering identity, type, status,
    params and timestamps; the core model shared by queue and history listings.
    """
    id: str = Field(title="Task Id")
    api_task_id: Optional[str] = Field(title="API Task Id", default=None)
    api_task_callback: Optional[str] = Field(title="API Task Callback", default=None)
    name: Optional[str] = Field(title="Task Name", default=None)
    type: str = Field(title="Task Type", description="Either txt2img or img2img")
    status: str = Field(
        "pending",
        title="Task Status",
        description="Either pending, running, done or failed",
    )
    params: Dict[str, Any] = Field(title="Task Parameters", description="The parameters of the task in JSON format")
    priority: Optional[int] = Field(title="Task Priority", default=None)
    position: Optional[int] = Field(title="Task Position", default=None)
    result: Optional[str] = Field(title="Task Result", description="The result of the task in JSON format", default=None)
    bookmarked: Optional[bool] = Field(title="Is task bookmarked", default=None)
    created_at: Optional[datetime] = Field(
        title="Task Created At",
        description="The time when the task was created",
        default=None,
    )
    updated_at: Optional[datetime] = Field(
        title="Task Updated At",
        description="The time when the task was updated",
        default=None,
    )


class Txt2ImgApiTaskArgs(StableDiffusionTxt2ImgProcessingAPI):
    """txt2img 任務參數模型 / Task-argument model for txt2img

    繼承 sd-webui 內部的 Txt2Img API 模型，額外加入 checkpoint / vae /
    callback_url 等欄位，並依 pydantic 版本隱藏內部專用的 send_images /
    save_images 欄位。
    Extends sd-webui's internal Txt2Img API model, adding checkpoint / vae /
    callback_url fields, and hides the internal-only send_images / save_images
    fields depending on the pydantic version.
    """
    checkpoint: Optional[str] = Field(
        None,
        title="Custom checkpoint.",
        description="Custom checkpoint hash. If not specified, the latest checkpoint will be used.",
    )
    vae: Optional[str] = Field(
        None,
        title="Custom VAE.",
        description="Custom VAE. If not specified, the current VAE will be used.",
    )
    sampler_index: Optional[str] = Field(sd_samplers.samplers[0].name, title="Sampler name", alias="sampler_name")
    callback_url: Optional[str] = Field(
        None,
        title="Callback URL",
        description="The callback URL to send the result to.",
    )

    """
    依 pydantic 版本選擇 schema 過濾的掛載方式 / Choose schema-filter mounting by pydantic version

    PYDANTIC_V2 旗標決定 pydantic 設定寫法，而過濾邏輯本身
    統一由 api_task_schema_extra 提供：
    The PYDANTIC_V2 flag decides the pydantic config style, while the
    filtering logic itself is always provided by api_task_schema_extra:

    - pydantic v2 (Forge Neo)：以 model_config + ConfigDict(json_schema_extra=)
      直接引用函式，pydantic 會以 (schema, model) 呼叫它。
      pydantic v2 (Forge Neo): reference the function directly via
      model_config + ConfigDict(json_schema_extra=); pydantic calls it with (schema, model).
    - pydantic v1 (A1111 classic)：需透過 Config.schema_extra 靜態方法包裝，
      再轉呼叫相同的 api_task_schema_extra。
      pydantic v1 (A1111 classic): wrap it via the Config.schema_extra static method,
      which then calls the same api_task_schema_extra.

    兩者執行的是同一段欄位移除邏輯（移除 send_images / save_images）。
     Both execute the same field-removal logic (removing send_images / save_images).
     """
    # 依 pydantic 版本選擇 schema 掛載方式：v2 用 model_config，v1 用 Config.schema_extra / Choose schema mounting by pydantic version: v2 model_config vs v1 Config.schema_extra
    if PYDANTIC_V2:
        model_config = ConfigDict(json_schema_extra=api_task_schema_extra)
    else:
        class Config(StableDiffusionTxt2ImgProcessingAPI.__config__):
            @staticmethod
            def schema_extra(schema: Dict[str, Any], model) -> None:
                api_task_schema_extra(schema, model)


class Img2ImgApiTaskArgs(StableDiffusionImg2ImgProcessingAPI):
    """img2img 任務參數模型 / Task-argument model for img2img

    與 Txt2ImgApiTaskArgs 對稱，繼承 img2img 內部 API 模型並補上額外欄位，
    同樣依 pydantic 版本隱藏內部專用欄位。
    Symmetric to Txt2ImgApiTaskArgs: extends the img2img internal API model with
    extra fields and hides internal-only fields by pydantic version.
    """
    checkpoint: Optional[str] = Field(
        None,
        title="Custom checkpoint.",
        description="Custom checkpoint hash. If not specified, the latest checkpoint will be used.",
    )
    vae: Optional[str] = Field(
        None,
        title="Custom VAE.",
        description="Custom VAE. If not specified, the current VAE will be used.",
    )
    sampler_index: Optional[str] = Field(sd_samplers.samplers[0].name, title="Sampler name", alias="sampler_name")
    callback_url: Optional[str] = Field(
        None,
        title="Callback URL",
        description="The callback URL to send the result to.",
    )

    """
    依 pydantic 版本選擇 schema 過濾的掛載方式 / Choose schema-filter mounting by pydantic version

    與 Txt2ImgApiTaskArgs 相同的版本判斷：v2 用 model_config，
    v1 用 Config.schema_extra 包裝，兩者共用 api_task_schema_extra。
     Same version branching as Txt2ImgApiTaskArgs: v2 uses model_config,
     v1 wraps via Config.schema_extra, both share api_task_schema_extra.
     """
    # 依 pydantic 版本選擇 schema 掛載方式：v2 用 model_config，v1 用 Config.schema_extra / Choose schema mounting by pydantic version: v2 model_config vs v1 Config.schema_extra
    if PYDANTIC_V2:
        model_config = ConfigDict(json_schema_extra=api_task_schema_extra)
    else:
        class Config(StableDiffusionImg2ImgProcessingAPI.__config__):
            @staticmethod
            def schema_extra(schema: Dict[str, Any], model) -> None:
                api_task_schema_extra(schema, model)


class QueueTaskResponse(BaseModel):
    """排入佇列後的回應 / Response returned after enqueuing a task

    僅回傳新建立任務的 id 讓呼叫端後續追蹤狀態。
    Returns only the newly created task id so the caller can track it later.
    """
    task_id: str = Field(title="Task Id")


class QueueStatusResponse(BaseModel):
    """佇列狀態回應 / Queue status response

    彙整目前進行中任務、待處理任務列表與佇列長度、暫停旗標，
    供前端渲染佇列畫面。
    Aggregates the in-progress task, the pending task list, queue length and the
    paused flag for the frontend to render the queue view.
    """
    current_task_id: Optional[str] = Field(title="Current Task Id", description="The on progress task id")
    pending_tasks: List[TaskModel] = Field(title="Pending Tasks", description="The pending tasks in the queue")
    total_pending_tasks: int = Field(title="Queue length", description="The total pending tasks in the queue")
    paused: bool = Field(title="Paused", description="Whether the queue is paused")

    class Config:
        json_encoders = {datetime: lambda dt: int(dt.timestamp() * 1e3)}


class HistoryResponse(BaseModel):
    """歷史任務列表回應 / History task list response

    回傳分頁後的任務列表與總數，供 history 頁面顯示與分頁。
    Returns the (paged) task list and total count for the history page.
    """
    tasks: List[TaskModel] = Field(title="Tasks")
    total: int = Field(title="Task count")

    class Config:
        json_encoders = {datetime: lambda dt: int(dt.timestamp() * 1e3)}


class UpdateTaskArgs(BaseModel):
    """更新任務時可修改的欄位 / Fields editable when updating a task

    僅開放 name / checkpoint / params 等可變更欄位，其餘皆不可經由更新 API 修改。
    Only exposes mutable fields like name / checkpoint / params; the rest cannot
    be changed via the update API.
    """
    name: Optional[str] = Field(title="Task Name")
    checkpoint: Optional[str]
    params: Optional[Dict[str, Any]] = Field(
        title="Task Parameters", description="The parameters of the task in JSON format"
    )
