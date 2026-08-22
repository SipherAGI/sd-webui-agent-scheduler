# preload.py 用於註冊命令列啟動參數 / preload.py registers command-line launch arguments
def preload(parser):
    """註冊 Agent Scheduler 專屬的命令列參數 / Register Agent Scheduler's command-line arguments

    sd-webui 啟動時會呼叫此 hook 注入擴充所需的啟動選項，
    此處提供 sqlite 資料庫檔案路徑的自訂參數。
    sd-webui calls this hook at startup to inject extension launch options;
    here we add a customizable path for the sqlite database file.
    """
    parser.add_argument(
        "--agent-scheduler-sqlite-file",
        help="sqlite file to use for the database connection. It can be abs or relative path(from base path) default: task_scheduler.sqlite3",
        default="task_scheduler.sqlite3",
    )