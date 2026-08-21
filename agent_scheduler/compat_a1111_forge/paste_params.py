"""paste 參數按鈕相關符號的版本相容匯出 / Version-compatible re-exports for paste-parameter symbols"""

"""
統一匯出 paste 參數相關符號 / Re-export paste-parameter symbols uniformly

在 Forge Neo 中這些符號位於 modules.infotext_utils，
在 A1111 (v1.7 以前) 則位於 modules.generation_parameters_copypaste。
In Forge Neo these symbols live in modules.infotext_utils,
while in A1111 (pre-v1.7) they live in modules.generation_parameters_copypaste.

優先嘗試新版 (Neo) 模組位置，失敗時回退至舊版 A1111 位置。
Prefer the newer (Neo) module location, falling back to legacy A1111.
"""
try:
    from modules.infotext_utils import (
        ParamBinding,
        connect_paste_params_buttons,
        register_paste_params_button,
        registered_param_bindings,
    )
except ImportError:
    from modules.generation_parameters_copypaste import (
        ParamBinding,
        connect_paste_params_buttons,
        register_paste_params_button,
        registered_param_bindings,
    )