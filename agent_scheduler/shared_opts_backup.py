from pathlib import Path, PurePath
import gradio as gr
from typing import Union


def simplify_path(input_path: Union[PurePath, str]) -> Path:
    """以純文字方式正規化路徑：逐段移除 '.' 與成對的 '..'，全程不觸碰檔案系統。
    Textually normalize a path: drop '.' segments and cancel paired '..' segments
    segment-by-segment, without any filesystem access.

    設計原因 / Rationale:
    此函式只做字串層級的清理，因此即使路徑不存在、掛載點損毀或權限不足也能安全執行；
    junction／符號連結的實際展開交由後續 Path.resolve() 處理，兩者互補。
    Because this works purely on strings, it stays safe even when the target does not
    exist, the mount point is broken, or access is denied; actual expansion of
    junctions / symlinks is delegated to the follow-up Path.resolve().
    """
    # 如果输入是字符串，将其转换为 Path 对象
    # If the input is a string, convert it into a Path object first.
    if isinstance(input_path, str):
        input_path = Path(input_path)

    # 逐段處理：遇到 '..' 時，若前一段是一般目錄則向上一層（彈出），否則保留
    # Process segment by segment: a '..' cancels the previous ordinary directory
    # segment (pop), otherwise it is kept as a leading or dangling '..'.
    parts = []
    for part in input_path.parts:
        if part == '..':
            # 僅當前一段為一般目錄（非 '..'、非斜線、非根）才彈出，避免誤刪根路徑
            # Only pop when the previous segment is an ordinary directory (not '..',
            # not a slash, not the root) so the root path is never removed by mistake.
            if parts and parts[-1] != '..' and parts[-1] != '/' and parts[-1] != input_path.root:
                parts.pop()
            else:
                parts.append(part)
        elif part != '.' and part != '':
            parts.append(part)

    # 如果路径是绝对路径，保留根路径
    # If the original path is absolute, re-attach its root prefix to the result.
    if input_path.is_absolute():
        simplified_path = Path(input_path.root, *parts)
    else:
        simplified_path = Path(*parts)

    return simplified_path

class SharedOptsBackup:
    """
    任務執行期間隔離並備份／還原 shared.opts 的工具類別。
    A class used to backup and restore shared options.

    Attributes
    ----------
    shared_opts : dict
        The shared options to be backed up and restored.
    backup : dict
        A dictionary to store the backup of shared options.

    Methods
    -------
    set_shared_opts_core(key: str, value)
        Sets a shared option and backs it up only if it is not already backed up.

    set_shared_opts(**kwargs)
        Sets multiple shared options and backs them up.

    restore_shared_opts()
        Restores the shared options from the backup.
    """

    def __init__(self, shared_opts):
        """
        Constructs all the necessary attributes for the SharedOptsBackup object.

        Parameters
        ----------
        shared_opts : dict
            The shared options to be backed up and restored.
        """
        self.shared_opts = shared_opts
        self.backup = {}

        # gr.Info(f"[AgentScheduler] backup shared opts")

    def set_shared_opts_core(self, key: str, value):
        """
        Sets a shared option and backs it up only if it is not already backed up.

        參數 / Parameters
        ----------
        key : str
            The key of the shared option to be set.
        value : any
            The value to be set for the shared option.

        回傳 / Returns
        -------
        最終存入的值；路徑型別會先展開（解析 junction／符號連結並轉絕對路徑）再儲存。
        The final stored value; path-like values are expanded (junction /
        symlink resolved, made absolute) before storing.

        設計原因（延後展開） / Design rationale (deferred expansion):
        路徑展開集中在此處進行，呼叫端可先以相對路徑（含 '..'）傳入，
        直到真正寫入 shared.opts 時才 resolve。這能避免任務執行中途因 CWD 位於
        junction 捷徑（Windows MAX_PATH=260）而超過路徑長度上限，導致存圖時
        builtins.open 拋出 FileNotFoundError。
        Path expansion is centralized here so callers may pass relative paths
        (including '..') and only get resolved when written to shared.opts. This
        avoids exceeding the Windows MAX_PATH limit (260) mid-task when the CWD
        sits on a junction symlink, which would otherwise make builtins.open raise
        FileNotFoundError during image saving.
        """
        # 備份原值（每個 key 只備份一次），還原時才能還原成任務執行前的設定
        # Back up the original value once per key so restore can revert exactly
        # to the pre-task setting.
        if not self.is_backup_exists(key):
            old = getattr(self.shared_opts, key, None)
            self.backup[key] = old
            print(f"[AgentScheduler] [backup] {key}: {old}")

        # 路徑型值：依序做文字正規化、junction 展開、絕對化、POSIX 正斜線化
        # Path-like values: textual simplify -> junction resolve -> absolute ->
        # POSIX forward-slash normalized string.
        if isinstance(value, (Path, PurePath)):
            # control_net_detectedmap_dir 刻意保持相對：該設定預期為相對於 CWD 的路徑，
            # resolve 成絕對會破壞其相對語意
            # control_net_detectedmap_dir is intentionally kept relative: that option
            # expects a CWD-relative path, so resolving it to absolute would break it.
            if key != "control_net_detectedmap_dir":
                value = simplify_path(value)
                value = value.resolve()

            value = str(value.as_posix())

        # 寫入 shared.opts；僅在與備份值不同時印出變更提示
        # Write into shared.opts; only log a change when it differs from the backup.
        self.shared_opts.set(key, value)
        if self.backup[key] != value:
            print(f"\33[32m[AgentScheduler] [change] {key}: {value}\33[0m")

        # 回傳最終（可能已展開的）值，方便呼叫端直接拿去做 makedirs 等後續操作
        # Return the final (possibly expanded) value so callers can directly use it
        # for follow-up steps such as os.makedirs.
        return value

    def set_shared_opts(self, **kwargs):
        """
        Sets multiple shared options and backs them up.

        Parameters
        ----------
        kwargs : dict
            The key-value pairs of shared options to be set.
        """
        for attr, value in kwargs.items():
            self.set_shared_opts_core(attr, value)

    def is_backup_exists(self, key: str):
        return key in self.backup

    def get_backup_value(self, key: str):
        """取得備份值；若該 key 未曾備份則回傳 shared.opts 目前值作為後備。
        Get the backed-up value; if the key was never backed up, fall back to the
        current shared.opts value.
        """
        # 已備份者優先取備份；未備份者直接回傳當前設定值，避免 KeyError
        # Prefer the backup when present; otherwise return the live option to avoid
        # a KeyError for keys that were never touched.
        return self.backup.get(key) if self.is_backup_exists(key) else getattr(self.shared_opts, key, None)

    def restore_shared_opts(self):
        """
        將備份中的設定一併還原回 shared.opts，使任務結束後恢復成執行前狀態。
        Restore every backed-up option into shared.opts so the environment returns
        to its pre-task state after the task finishes.
        """
        # 逐 key 還原並印出日誌；若當初未變更，值會與備份相同（等同無效操作）
        # Restore each key and log it; unchanged keys restore to the same value.
        for attr, value in self.backup.items():
            self.shared_opts.set(attr, value)
            print(f"\33[32m[AgentScheduler] [restore] {attr}: {value}\33[0m")

        # gr.Info(f"[AgentScheduler] restore shared opts")
