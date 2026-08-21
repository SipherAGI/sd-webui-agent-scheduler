# AGENTS.md — sd-webui-agent-scheduler

本專案是 sd-webui 的擴充（Agent Scheduler 排程器），需要在**兩套不同的 sd-webui 環境**下皆可運作。本文件說明兩個版本的差異、開發路徑，以及本專案採用的相容性策略模式。

This project is an sd-webui extension (Agent Scheduler) that must run under **two different sd-webui environments**. This document explains the differences between the two versions, their development paths, and the compatibility strategy pattern used in this project.

## 環境路徑 / Environment Paths

| 版本 / Version | 路徑 / Path | gradio | pydantic | 關鍵模組 / Key module |
|---|---|---|---|---|
| **A1111** | `D:\Users\WebstormProjects\ai\stable-diffusion-webui-a1111` | 3.41.2 | v1 | `modules\infotext_utils.py` |
| **Forge Neo** | `D:\Program Files (AI)\sd-webui-forge-classic` | 4.40.0 | v2.10.6 | `modules\infotext_utils.py` |

注意 / Note:
- 兩個版本皆可執行 `modules\infotext_utils.py` 內的符號（`parse_generation_parameters`、`ParamBinding`、`register_paste_params_button`、`connect_paste_params_buttons`、`create_override_settings_dict` 等）。
- A1111 的 venv 未放在專案內，執行驗證時請以各環境實際的 Python 直譯器與已安裝套件為準。

## 版本差異摘要 / Version Differences

### 1. 模組位置 / Module Location

| 符號 / Symbol | A1111 (新) / Forge Neo | A1111 (舊版相容) |
|---|---|---|
| `parse_generation_parameters` | `modules.infotext_utils` | `modules.generation_parameters_copypaste` |
| `ParamBinding` 等 paste 按鈕符號 | `modules.infotext_utils` | `modules.generation_parameters_copypaste` |
| `create_override_settings_dict` | `modules.infotext_utils` | `modules.generation_parameters_copypaste` |

**處理方式 / Approach:** 優先嘗試新版位置 `modules.infotext_utils`，`ImportError` 時回退至舊版 `modules.generation_parameters_copypaste`。

### 2. pydantic 版本差異 / pydantic Version Differences

| 功能 / Feature | pydantic v1 (A1111) | pydantic v2 (Forge Neo) |
|---|---|---|
| 設定寫法 / Config style | `class Config(...)` | `model_config = ConfigDict(...)` |
| 繼承動態模型的 config | `class Config(Base.__config__):` | `model_config = ConfigDict(json_schema_extra=...)` |
| 序列化排除欄位 | `Config.exclude = [...]` | `Field(..., exclude=True)` |
| schema 過濾 | `Config.schema_extra` | `ConfigDict(json_schema_extra=...)` |

**處理方式 / Approach:**
- 使用 `PYDANTIC_V2` 旗標（依 `pydantic.VERSION` 判斷）在類別內分支選用寫法。
- 排除欄位統一用 `Field(exclude=True)`（兩版本皆支援），避免版本分支。
- schema 過濾邏輯抽成共用函式 `api_task_schema_extra`，v1/v2 各自掛載。

### 3. gradio 版本差異 / gradio Version Differences

| 功能 / Feature | gradio 3.x (A1111) | gradio 4.x (Forge Neo) |
|---|---|---|
| 事件相依列表 | `root.dependencies`（list of dict） | 不存在；位於 `root.fns`（dict of `BlockFunction`） |
| 相依 dict 欄位 | `trigger` / `targets` / `inputs` / `outputs` 皆為 id | `BlockFunction.targets` 為 `(id, event_name)` 元組；`inputs`/`outputs` 為元件物件 |
| 函式物件列表 | `root.fns` 可迭代 | `root.fns` 是 dict，迭代會得到 id 鍵 |

**處理方式 / Approach:** 透過 `get_ui_dependencies()` 將新舊結構正規化為舊版 dict 格式，透過 `get_ui_fns()` 統一回傳 `BlockFunction` 列表。

## 相容性策略模式 / Compatibility Strategy Pattern

所有版本差異的相容處理集中在 `agent_scheduler/compat_a1111_forge/` 套件內，依**差異主題**拆分為子模組：

```
agent_scheduler/compat_a1111_forge/
├── __init__.py        # 僅匯出 infotext 相關（供最常見需求）
├── pydantic.py        # PYDANTIC_V2 旗標、api_task_schema_extra
├── infotext.py        # parse_generation_parameters
├── paste_params.py    # ParamBinding 等 paste 按鈕符號、create_override_settings_dict
└── ui.py              # get_ui_dependencies、get_ui_fns（gradio 3.x / 4.x 相容）
```

### 核心原則 / Core Principles

1. **子模組按主題拆分**：每個子模組只處理一個「差異主題」（pydantic / infotext / paste params / gradio ui）。移植到其他專案時，可只複製需要的子模組。
2. **`__init__.py` 最小化**：只匯出最常用的 infotext 功能，避免引入呼叫端不需要的相依。其他功能一律直接從子模組匯入，例如：
   ```python
   from agent_scheduler.compat_a1111_forge.pydantic import PYDANTIC_V2
   from agent_scheduler.compat_a1111_forge.paste_params import create_override_settings_dict
   from agent_scheduler.compat_a1111_forge.ui import get_ui_dependencies
   ```
3. **模組位置差異 → try/except 回退**：新版位置優先，舊版位置作為 `ImportError` 回退。
4. **行為差異 → 版本旗標分支**：需要不同寫法時，用 `PYDANTIC_V2` 等旗標在呼叫端類別內分支；能共用者（如 `Field(exclude=True)`）一律共用以減少分支。
5. **共用邏輯抽離**：跨版本相同的工作（如 schema 過濾）抽成單一函式，各版本只負責「掛載方式」的差異。
6. **呼叫端不直接引用版本特定模組**：一律透過 compat 套件取得符號，保持呼叫端與版本無關。

### 新增相容符號的步驟 / Steps to Add a New Compat Symbol

1. 判斷該符號的「差異主題」，放入對應子模組（無對應者新建子模組）。
2. 若只是「位置不同」：仿照 `paste_params.py` 的 try/except 雙來源匯出。
3. 若涉及「寫法不同」：仿照 `pydantic.py` / `ui.py` 提供版本旗標或正規化函式。
4. 更新呼叫端 import，改為從 compat 子模組匯入。
5. 用兩套環境的 Python 執行 `py_compile` 驗證語法。

## 驗證 / Verification

- 語法驗證：以各環境 Python 執行 `python -m py_compile <檔案>`。
  - Forge Neo 範例：`"D:/Program Files (AI)/sd-webui-forge-classic/venv/Scripts/python.exe" -m py_compile <檔案>`
- 注意：`modules.*` 相關 import 需要完整的 sd-webui 啟動流程才能載入；僅做語法檢查時請勿直接執行含模組 import 的檔案，改以 `py_compile` 驗證。
- 完整驗證需在目標環境實際啟動 sd-webui（`webui-user.bat` / `webui.bat`），確認無 script load 錯誤。