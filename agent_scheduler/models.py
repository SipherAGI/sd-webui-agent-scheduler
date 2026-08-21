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
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if dt else None


def transform_to_utc_datetime(dt: datetime) -> datetime:
    return dt.astimezone(tz=timezone.utc)


class QueueStatusAPI(BaseModel):
    limit: Optional[int] = Field(title="Limit", description="The maximum number of tasks to return", default=20)
    offset: Optional[int] = Field(title="Offset", description="The offset of the tasks to return", default=0)


class TaskModel(BaseModel):
    id: str = Field(title="Task Id")
    api_task_id: Optional[str] = Field(title="API Task Id", default=None)
    api_task_callback: Optional[str] = Field(title="API Task Callback", default=None)
    name: Optional[str] = Field(title="Task Name")
    type: str = Field(title="Task Type", description="Either txt2img or img2img")
    status: str = Field(
        "pending",
        title="Task Status",
        description="Either pending, running, done or failed",
    )
    params: Dict[str, Any] = Field(title="Task Parameters", description="The parameters of the task in JSON format")
    priority: Optional[int] = Field(title="Task Priority")
    position: Optional[int] = Field(title="Task Position")
    result: Optional[str] = Field(title="Task Result", description="The result of the task in JSON format")
    bookmarked: Optional[bool] = Field(title="Is task bookmarked")
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

    if PYDANTIC_V2:
        model_config = ConfigDict(json_schema_extra=api_task_schema_extra)
    else:
        class Config(StableDiffusionTxt2ImgProcessingAPI.__config__):
            @staticmethod
            def schema_extra(schema: Dict[str, Any], model) -> None:
                api_task_schema_extra(schema, model)


class Img2ImgApiTaskArgs(StableDiffusionImg2ImgProcessingAPI):
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

    if PYDANTIC_V2:
        model_config = ConfigDict(json_schema_extra=api_task_schema_extra)
    else:
        class Config(StableDiffusionImg2ImgProcessingAPI.__config__):
            @staticmethod
            def schema_extra(schema: Dict[str, Any], model) -> None:
                api_task_schema_extra(schema, model)


class QueueTaskResponse(BaseModel):
    task_id: str = Field(title="Task Id")


class QueueStatusResponse(BaseModel):
    current_task_id: Optional[str] = Field(title="Current Task Id", description="The on progress task id")
    pending_tasks: List[TaskModel] = Field(title="Pending Tasks", description="The pending tasks in the queue")
    total_pending_tasks: int = Field(title="Queue length", description="The total pending tasks in the queue")
    paused: bool = Field(title="Paused", description="Whether the queue is paused")

    class Config:
        json_encoders = {datetime: lambda dt: int(dt.timestamp() * 1e3)}


class HistoryResponse(BaseModel):
    tasks: List[TaskModel] = Field(title="Tasks")
    total: int = Field(title="Task count")

    class Config:
        json_encoders = {datetime: lambda dt: int(dt.timestamp() * 1e3)}


class UpdateTaskArgs(BaseModel):
    name: Optional[str] = Field(title="Task Name")
    checkpoint: Optional[str]
    params: Optional[Dict[str, Any]] = Field(
        title="Task Parameters", description="The parameters of the task in JSON format"
    )
