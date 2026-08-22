from pathlib import Path, PurePath
import gradio as gr
from typing import Union


def simplify_path(input_path: Union[PurePath, str]) -> Path:
    # 如果输入是字符串，将其转换为 Path 对象
    if isinstance(input_path, str):
        input_path = Path(input_path)

    parts = []
    for part in input_path.parts:
        if part == '..':
            if parts and parts[-1] != '..' and parts[-1] != '/' and parts[-1] != input_path.root:
                parts.pop()
            else:
                parts.append(part)
        elif part != '.' and part != '':
            parts.append(part)

    # 如果路径是绝对路径，保留根路径
    if input_path.is_absolute():
        simplified_path = Path(input_path.root, *parts)
    else:
        simplified_path = Path(*parts)

    return simplified_path

class SharedOptsBackup:
    """
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

        Parameters
        ----------
        key : str
            The key of the shared option to be set.
        value : any
            The value to be set for the shared option.

        Returns
        -------
        The final stored value; path-like values are expanded (junction /
        symlink resolved, made absolute) before storing.
        """
        if not self.is_backup_exists(key):
            old = getattr(self.shared_opts, key, None)
            self.backup[key] = old
            print(f"[AgentScheduler] [backup] {key}: {old}")

        if isinstance(value, (Path, PurePath)):
            if key != "control_net_detectedmap_dir":
                value = simplify_path(value)
                value = value.resolve()

            value = str(value.as_posix())

        self.shared_opts.set(key, value)
        if self.backup[key] != value:
            print(f"\33[32m[AgentScheduler] [change] {key}: {value}\33[0m")

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
        return self.backup.get(key) if self.is_backup_exists(key) else getattr(self.shared_opts, key, None)

    def restore_shared_opts(self):
        """
        Restores the shared options from the backup.
        """
        for attr, value in self.backup.items():
            self.shared_opts.set(attr, value)
            print(f"\33[32m[AgentScheduler] [restore] {attr}: {value}\33[0m")

        # gr.Info(f"[AgentScheduler] restore shared opts")
