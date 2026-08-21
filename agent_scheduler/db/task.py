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
    impl = DateTimeImpl
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], _):
        if value is None:
            return None
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Optional[datetime], _):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Task(TaskModel):
    """
    script_params 以 Field(exclude=True) 排除於 dict 序列化之外，
    此寫法在 pydantic v1 (A1111 classic) 與 v2 (Forge Neo) 皆相容，
    取代 v1 專用的 Config.exclude 寫法。
    script_params is excluded from dict serialization via Field(exclude=True),
    which is compatible with both pydantic v1 (A1111 classic) and v2 (Forge Neo),
    replacing the v1-only Config.exclude syntax.
    """
    script_params: bytes = Field(None, exclude=True)
    params: str

    def __init__(self, **kwargs):
        priority = kwargs.pop("priority", int(datetime.now(timezone.utc).timestamp() * 1000))
        super().__init__(priority=priority, **kwargs)

    @staticmethod
    def from_table(table: "TaskTable"):
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
    def get_task(self, id: str) -> Union[TaskTable, None]:
        session = Session(self.engine)
        try:
            task = session.get(TaskTable, id)

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

            if bookmarked == True:
                query = query.filter(TaskTable.bookmarked == bookmarked)
            else:
                query = query.order_by(TaskTable.bookmarked.asc())

            query = query.order_by(TaskTable.priority.asc() if order == "asc" else TaskTable.priority.desc())

            if limit:
                query = query.limit(limit)

            if offset:
                query = query.offset(offset)

            all = query.all()
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
            if current is None:
                raise Exception(f"Task with id {id} not found")

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

        session = Session(self.engine)
        try:
            result = session.get(TaskTable, id)
            if result:
                if priority == 0:
                    result.priority = self.__get_min_priority(status=TaskStatus.PENDING) - 1
                elif priority == -1:
                    result.priority = int(datetime.now(timezone.utc).timestamp() * 1000)
                else:
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
        session = Session(self.engine)
        try:
            query = session.query(TaskTable).filter(TaskTable.bookmarked == False)

            if before:
                query = query.filter(TaskTable.created_at < before)

            if status is not None:
                if isinstance(status, list):
                    query = query.filter(TaskTable.status.in_(status))
                else:
                    query = query.filter(TaskTable.status == status)

            deleted_rows = query.delete()
            session.commit()

            return deleted_rows
        except Exception as e:
            print(f"Exception deleting tasks from database: {e}")
            raise e
        finally:
            session.close()

    def __get_min_priority(self, status: str = None) -> int:
        session = Session(self.engine)
        try:
            query = session.query(func.min(TaskTable.priority))
            if status is not None:
                query = query.filter(TaskTable.status == status)

            min_priority = query.scalar()
            return min_priority if min_priority else 0
        except Exception as e:
            print(f"Exception getting min priority from database: {e}")
            raise e
        finally:
            session.close()

    def __move_tasks_down(self, priority: int):
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
