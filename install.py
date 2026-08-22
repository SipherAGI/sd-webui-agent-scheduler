# install.py 在擴充載入時被呼叫，確保執行期相依套件已安裝 / install.py runs at extension load to ensure runtime dependencies are present
import launch

# 任務排程依賴 sqlalchemy 做為資料庫層；若環境尚未安裝則透過 pip 補裝 / Task scheduling relies on sqlalchemy; install it via pip if missing
if not launch.is_installed("sqlalchemy"):
    launch.run_pip("install sqlalchemy", "requirement for task-scheduler")
