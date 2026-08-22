import json
import base64
from enum import Enum
from datetime import datetime, timezone
from typing import Optional, Union, List, Dict

from sqlalchemy import (
    TypeDecorator,
    Column,
    String,
    Text,
    Integer,
    DateTime as DateTimeImpl,
    LargeBinary,
    Boolean,
    text,
    func,
)
from sqlalchemy.orm import Session

from .base import BaseTableManager, Base
from ..models import TaskModel
from pydantic import Field


class DateTime(TypeDecorator):
    # 自訂 SQLAlchemy 型別：統一將時間欄位以 UTC 儲存與讀取，避免時區不一致 / Custom type that normalizes datetimes to UTC on both write and read to avoid timezone drift.
    impl = DateTimeImpl
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], _):
        # 寫入前轉為 UTC；空值直接放行 / Convert to UTC before persisting; pass through None.
        if value is None:
            return None
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Optional[datetime], _):
        # 讀出時確保帶有 UTC 時區資訊，缺漏則補上 / Ensure returned datetimes carry UTC; fill in tzinfo when missing.
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class TaskStatus(str, Enum):
    # 任務的生命週期狀態，用於排程與篩選 / Task lifecycle states used for scheduling and filtering.
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Task(TaskModel):
    """
    任務的業務層模型：繼承自 pydantic 的 TaskModel，並補上與資料表對應的欄位。
    script_params 以 Field(exclude=True) 排除於 dict 序列化之外，
    此寫法在 pydantic v1 (A1111 classic) 與 v2 (Forge Neo) 皆相容，
    取代 v1 專用的 Config.exclude 寫法。
    Business-layer task model extending the pydantic TaskModel with DB-bound fields.
    script_params is excluded from dict serialization via Field(exclude=True),
    which is compatible with both pydantic v1 (A1111 classic) and v2 (Forge Neo),
    replacing the v1-only Config.exclude syntax.
    """
    script_params: bytes = Field(None, exclude=True)
    params: str

    def __init__(self, **kwargs):
        # 未指定優先權時，以目前 UTC 時間戳作為預設，使後加入的任務自然排到佇列尾端 / Default priority to the current UTC timestamp so new tasks naturally land at the queue tail.
        priority = kwargs.pop("priority", int(datetime.now(timezone.utc).timestamp() * 1000))
        super().__init__(priority=priority, **kwargs)

    @staticmethod
    def from_table(table: "TaskTable"):
        # 由 ORM 資料列重建業務模型 / Reconstruct the business model from an ORM row.
        return Task(
            id=table.id,
            api_task_id=table.api_task_id,
            api_task_callback=table.api_task_callback,
            name=table.name,
            type=table.type,
            params=table.params,
            script_params=table.script_params,
            priority=table.priority,
            status=table.status,
            result=table.result,
            bookmarked=table.bookmarked,
            created_at=table.created_at,
            updated_at=table.updated_at,
        )

    def to_table(self):
        # 轉換為 ORM 資料列以便寫入資料庫 / Convert to an ORM row for persistence.
        return TaskTable(
            id=self.id,
            api_task_id=self.api_task_id,
            api_task_callback=self.api_task_callback,
            name=self.name,
            type=self.type,
            params=self.params,
            script_params=self.script_params,
            priority=self.priority,
            status=self.status,
            result=self.result,
            bookmarked=self.bookmarked,
        )

    def from_json(json_obj: Dict):
        # 由外部 JSON（可能來自 API）解析出任務；script_params 以 base64 解碼還原為位元組 / Parse a task from external JSON (e.g. API); decode base64 script_params back to bytes.
        return Task(
            id=json_obj.get("id"),
            api_task_id=json_obj.get("api_task_id", None),
            api_task_callback=json_obj.get("api_task_callback", None),
            name=json_obj.get("name", None),
            type=json_obj.get("type"),
            status=json_obj.get("status", TaskStatus.PENDING),
            params=json.dumps(json_obj.get("params")),
            script_params=base64.b64decode(json_obj.get("script_params")),
            priority=json_obj.get("priority", int(datetime.now(timezone.utc).timestamp() * 1000)),
            result=json_obj.get("result", None),
            bookmarked=json_obj.get("bookmarked", False),
            created_at=datetime.fromtimestamp(json_obj.get("created_at", datetime.now(timezone.utc).timestamp())),
            updated_at=datetime.fromtimestamp(json_obj.get("updated_at", datetime.now(timezone.utc).timestamp())),
        )

    def to_json(self):
        # 序列化為對外 JSON；params 轉回 dict、script_params 以 base64 編碼 / Serialize to external JSON; reload params to dict and base64-encode script_params.
        return {
            "id": self.id,
            "api_task_id": self.api_task_id,
            "api_task_callback": self.api_task_callback,
            "name": self.name,
            "type": self.type,
            "status": self.status,
            "params": json.loads(self.params),
            "script_params": base64.b64encode(self.script_params).decode("utf-8"),
            "priority": self.priority,
            "result": self.result,
            "bookmarked": self.bookmarked,
            "created_at": int(self.created_at.timestamp()),
            "updated_at": int(self.updated_at.timestamp()),
        }


class TaskTable(Base):
    # 任務資料表的 ORM 對應；欄位涵蓋排程、狀態、參數與時間戳 / ORM mapping of the task table covering scheduling, status, params and timestamps.
    __tablename__ = "task"

    id = Column(String(64), primary_key=True)
    api_task_id = Column(String(64), nullable=True)
    api_task_callback = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    type = Column(String(20), nullable=False)  # txt2img or img2txt
    params = Column(Text, nullable=False)  # task args
    script_params = Column(LargeBinary, nullable=False)  # script args
    priority = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, done, failed
    result = Column(Text)  # task result
    bookmarked = Column(Boolean, nullable=True, default=False)
    created_at = Column(
        DateTime,
        nullable=False,
        server_default=text("(datetime('now'))"),
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=text("(datetime('now'))"),
        onupdate=text("(datetime('now'))"),
    )

    def __repr__(self):
        return f"Task(id={self.id!r}, type={self.type!r}, params={self.params!r}, status={self.status!r}, created_at={self.created_at!r})"


class TaskManager(BaseTableManager):
    # 任務資料表的 CRUD 與排程輔助操作，統一管理 session 生命週期 / Task table CRUD and scheduling helpers with unified session lifecycle management.
    def get_task(self, id: str) -> Union[TaskTable, None]:
        session = Session(self.engine)
        try:
            task = session.get(TaskTable, id)

            # 命中則轉為業務模型，否則回傳 None / Convert to business model when found, otherwise None.
            return Task.from_table(task) if task else None
        except Exception as e:
            print(f"Exception getting task from database: {e}")
            raise e
        finally:
            session.close()

    def get_task_position(self, id: str) -> int:
        session = Session(self.engine)
        try:
            task = session.get(TaskTable, id)
            # 任務存在時，計算其前方仍為 pending 的任務數量作為佇列位置 / When found, count preceding pending tasks as its queue position.
            if task:
                return (
                    session.query(func.count(TaskTable.id))
                    .filter(TaskTable.status == TaskStatus.PENDING)
                    .filter(TaskTable.priority < task.priority)
                    .scalar()
                )
            else:
                raise Exception(f"Task with id {id} not found")
        except Exception as e:
            print(f"Exception getting task position from database: {e}")
            raise e
        finally:
            session.close()

    def get_tasks(
        self,
        type: str = None,
        status: Union[str, List[str]] = None,
        bookmarked: bool = None,
        api_task_id: str = None,
        limit: int = None,
        offset: int = None,
        order: str = "asc",
    ) -> List[TaskTable]:
        # 多條件查詢任務列表，支援型別、狀態、書籤、分頁與排序 / Query tasks with filters on type/status/bookmark plus pagination and ordering.
        session = Session(self.engine)
        try:
            query = session.query(TaskTable)
            # 依任務型別過濾（如 txt2img / img2txt） / Filter by task type when provided.
            if type:
                query = query.filter(TaskTable.type == type)

            # 狀態可為單值或列表，列表時用 IN 查詢 / Accept a single status or a list; use IN for lists.
            if status is not None:
                if isinstance(status, list):
                    query = query.filter(TaskTable.status.in_(status))
                else:
                    query = query.filter(TaskTable.status == status)

            # 依外部 API 任務編號過濾 / Filter by external API task id when provided.
            if api_task_id:
                query = query.filter(TaskTable.api_task_id == api_task_id)

            # 指定書籤時只回傳書籤項，否則把書籤項排在前面 / When bookmarked is requested, filter to those; otherwise float bookmarked items first.
            if bookmarked == True:
                query = query.filter(TaskTable.bookmarked == bookmarked)
            else:
                query = query.order_by(TaskTable.bookmarked.asc())

            # 依優先權升/降冪排序，作為主要排程順序 / Order by priority (asc/desc) as the primary schedule order.
            query = query.order_by(TaskTable.priority.asc() if order == "asc" else TaskTable.priority.desc())

            # 分頁限制 / Apply pagination limits.
            if limit:
                query = query.limit(limit)

            if offset:
                query = query.offset(offset)

            all = query.all()
            # 批次轉換為業務模型回傳 / Convert all rows to business models.
            return [Task.from_table(t) for t in all]
        except Exception as e:
            print(f"Exception getting tasks from database: {e}")
            raise e
        finally:
            session.close()

    def count_tasks(
        self,
        type: str = None,
        status: Union[str, List[str]] = None,
        api_task_id: str = None,
    ) -> int:
        # 計算符合條件的任務數量，用於清單總數與分頁 / Count tasks matching filters for list totals and pagination.
        session = Session(self.engine)
        try:
            query = session.query(TaskTable)
            if type:
                query = query.filter(TaskTable.type == type)

            if status is not None:
                if isinstance(status, list):
                    query = query.filter(TaskTable.status.in_(status))
                else:
                    query = query.filter(TaskTable.status == status)

            if api_task_id:
                query = query.filter(TaskTable.api_task_id == api_task_id)

            return query.count()
        except Exception as e:
            print(f"Exception counting tasks from database: {e}")
            raise e
        finally:
            session.close()

    def add_task(self, task: Task) -> TaskTable:
        session = Session(self.engine)
        try:
            item = task.to_table()
            session.add(item)
            session.commit()
            return task
        except Exception as e:
            print(f"Exception adding task to database: {e}")
            raise e
        finally:
            session.close()

    def update_task(self, task: Task) -> TaskTable:
        session = Session(self.engine)
        try:
            current = session.get(TaskTable, task.id)
            # 目標不存在則拋錯，避免靜默更新失敗 / Raise when the target is missing to avoid silent no-ops.
            if current is None:
                raise Exception(f"Task with id {id} not found")

            # 以傳入模型合併既有列，更新所有欄位 / Merge incoming model into the existing row to update all fields.
            session.merge(task.to_table())
            session.commit()
            return task

        except Exception as e:
            print(f"Exception updating task in database: {e}")
            raise e
        finally:
            session.close()

    def prioritize_task(self, id: str, priority: int) -> TaskTable:
        """0 means move to top, -1 means move to bottom, otherwise set the exact priority"""
        # 調整單一任務的排程優先權 / Reorder a single task's scheduling priority.

        session = Session(self.engine)
        try:
            result = session.get(TaskTable, id)
            # 僅在任務存在時進行優先權調整 / Only adjust priority when the task exists.
            if result:
                # 0：移到佇列最前（取最小 pending 優先權再減一） / 0 = move to the very top (just below the current minimum pending priority).
                if priority == 0:
                    result.priority = self.__get_min_priority(status=TaskStatus.PENDING) - 1
                # -1：移到佇列最後（使用當前時間戳作為最大優先權） / -1 = move to the bottom (use current timestamp as the largest priority).
                elif priority == -1:
                    result.priority = int(datetime.now(timezone.utc).timestamp() * 1000)
                else:
                    # 其他值：先將該優先權及其之後的任務下移，騰出空位 / Otherwise make room by shifting tasks at/after that priority down by one.
                    self.__move_tasks_down(priority)
                    session.execute(text("SELECT 1"))
                    result.priority = priority

                session.commit()
                return result
            else:
                raise Exception(f"Task with id {id} not found")
        except Exception as e:
            print(f"Exception updating task in database: {e}")
            raise e
        finally:
            session.close()

    def delete_task(self, id: str):
        session = Session(self.engine)
        try:
            result = session.get(TaskTable, id)
            # 存在才刪除，否則拋錯以提示呼叫端 / Delete only when present; otherwise raise to alert the caller.
            if result:
                session.delete(result)
                session.commit()
            else:
                raise Exception(f"Task with id {id} not found")
        except Exception as e:
            print(f"Exception deleting task from database: {e}")
            raise e
        finally:
            session.close()

    def delete_tasks(
        self,
        before: datetime = None,
        status: Union[str, List[str]] = [
            TaskStatus.DONE,
            TaskStatus.FAILED,
            TaskStatus.INTERRUPTED,
        ],
    ):
        # 批次清理任務：排除書籤項，並可指定時間與狀態範圍 / Batch-clean tasks, always excluding bookmarked ones, optionally by age and status.
        session = Session(self.engine)
        try:
            # 書籤項視為使用者保留，絕不批次刪除 / Bookmarked tasks are user-pinned and must never be batch-deleted.
            query = session.query(TaskTable).filter(TaskTable.bookmarked == False)

            # 僅清除早於指定時間建立的任務 / Only purge tasks created before the given time.
            if before:
                query = query.filter(TaskTable.created_at < before)

            # 依狀態（單值或列表）過濾要清理的任務 / Filter by status (single value or list) for cleanup scope.
            if status is not None:
                if isinstance(status, list):
                    query = query.filter(TaskTable.status.in_(status))
                else:
                    query = query.filter(TaskTable.status == status)

            # 執行刪除並回傳影響列數 / Execute deletion and report affected row count.
            deleted_rows = query.delete()
            session.commit()

            return deleted_rows
        except Exception as e:
            print(f"Exception deleting tasks from database: {e}")
            raise e
        finally:
            session.close()

    def __get_min_priority(self, status: str = None) -> int:
        # 取得指定狀態的最小優先權值，用於「移到最前」計算 / Get the minimum priority for a status, used when moving a task to the top.
        session = Session(self.engine)
        try:
            query = session.query(func.min(TaskTable.priority))
            if status is not None:
                query = query.filter(TaskTable.status == status)

            min_priority = query.scalar()
            # 無資料時視為 0，確保新優先權為負數也能排在最前 / Default to 0 when empty so the new priority stays ahead.
            return min_priority if min_priority else 0
        except Exception as e:
            print(f"Exception getting min priority from database: {e}")
            raise e
        finally:
            session.close()

    def __move_tasks_down(self, priority: int):
        # 將優先權大於等於指定值的任務全部 +1，騰出插入空間 / Shift all tasks with priority >= given value down by one to free a slot.
        session = Session(self.engine)
        try:
            session.query(TaskTable).filter(TaskTable.priority >= priority).update(
                {TaskTable.priority: TaskTable.priority + 1}
            )
            session.commit()
        except Exception as e:
            print(f"Exception moving tasks down in database: {e}")
            raise e
        finally:
            session.close()
