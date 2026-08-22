"""
task_helpers.py — 任務參數的序列化與轉換輔助函式 / Task parameter serialization and conversion helpers.

本模組負責在 sd-webui 的 UI 任務參數、API 任務參數，以及可序列化格式之間互相轉換，
以便在 Agent Scheduler 中保存、還原與排程生成任務。
This module converts between sd-webui UI task args, API task args, and serializable
formats so the Agent Scheduler can persist, restore, and schedule generation tasks.
"""

import io
import zlib
import base64
import pickle
import inspect
import requests
import numpy as np
import torch
from typing import Union, List, Dict
from enum import Enum
from PIL import Image, ImageOps, ImageChops, ImageEnhance, ImageFilter, PngImagePlugin
from numpy import ndarray
from torch import Tensor

from modules import sd_samplers, scripts, shared, sd_vae, images, txt2img, img2img
from agent_scheduler.compat_a1111_forge.paste_params import create_override_settings_dict
from modules.sd_models import CheckpointInfo, get_closet_checkpoint_match
from modules.api.models import (
    StableDiffusionTxt2ImgProcessingAPI,
    StableDiffusionImg2ImgProcessingAPI,
)

from .helpers import log, get_dict_attribute

# img2img 各模式對應的圖片參數鍵名映射 / Maps each img2img mode to its image argument key paths.
img2img_image_args_by_mode: Dict[int, List[List[str]]] = {
    0: [["init_img"]],
    1: [["sketch"]],
    2: [["init_img_with_mask", "image"], ["init_img_with_mask", "mask"]],
    3: [["inpaint_color_sketch"], ["inpaint_color_sketch_orig"]],
    4: [["init_img_inpaint"], ["init_mask_inpaint"]],
}


def get_script_by_name(script_name: str, is_img2img: bool = False, is_always_on: bool = False) -> scripts.Script:
    """
    依腳本名稱取得對應的 scripts.Script 實例 / Resolve a scripts.Script instance by its title.

    為什麼需要：Agent Scheduler 在序列化/反序列化任務時，需要根據名稱找到具體的腳本物件，
    以便正確地轉換其參數格式（tx2img / img2img、alwayson / selectable）。
    Why: when (de)serializing tasks the scheduler must locate the concrete script object
    by name to correctly map its argument format (txt2img/img2img, alwayson/selectable).
    """
    # 依是否為 img2img 選擇正確的 script runner / Pick the correct script runner for txt2img vs img2img.
    script_runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img
    # 依是否為 alwayson 腳本選擇可取得的腳本清單 / Choose the script list based on alwayson vs selectable.
    available_scripts = script_runner.alwayson_scripts if is_always_on else script_runner.selectable_scripts

    # 以不區分大小寫的方式比對標題，找不到時回傳 None / Case-insensitive title match; return None if absent.
    return next(
        (s for s in available_scripts if s.title().lower() == script_name.lower()),
        None,
    )


def load_image_from_url(url: str):
    """
    從 URL 下載並開啟圖片 / Download and open an image from a URL.

    為什麼需要：任務參數中的圖片可能以網址形式提供，需先轉為 PIL 影像才能進一步處理。
    Why: image args may be supplied as URLs and must be fetched into a PIL image before processing.
    """
    # 嘗試下載並開啟圖片；任何失敗都記錄錯誤並回傳 None，避免中斷排程 / Download and open; on any failure log and return None to keep scheduling alive.
    try:
        response = requests.get(url)
        buffer = io.BytesIO(response.content)
        return Image.open(buffer)
    except Exception as e:
        log.error(f"[AgentScheduler] Error downloading image from url: {e}")
        return None


def encode_image_to_base64(image):
    """
    將圖片編碼為 base64 data URI / Encode an image into a base64 data URI.

    為什麼需要：API 任務參數需要以純文字（base64）攜帶圖片，並保留生成資訊以利復原。
    Why: API task args must carry images as plain-text base64 and preserve generation metadata for reproducibility.
    """
    # 若傳入 ndarray 則先轉為 PIL 影像 / Convert ndarray input into a PIL image first.
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    # 若傳入的是網址字串，則先下載為影像 / If given a URL string, fetch it into an image first.
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = load_image_from_url(image)

    # 非 PIL 影像（可能已是 base64 字串）直接原樣回傳 / Non-PIL input (e.g. already base64) is returned as-is.
    if not isinstance(image, Image.Image):
        return image

    # 讀取圖片內嵌的生成資訊（geninfo） / Read embedded generation info from the image.
    geninfo, _ = images.read_info_from_image(image)
    pnginfo = PngImagePlugin.PngInfo()
    # 若有生成資訊，加入 PNG 中繼資料以便後續復原 / If present, embed geninfo into PNG metadata for later restore.
    if geninfo:
        pnginfo.add_text("parameters", geninfo)

    with io.BytesIO() as output_bytes:
        # 有生成資訊才寫入 pnginfo；否則省略以節省空間 / Only attach pnginfo when geninfo exists, else skip to save space.
        if geninfo:
            image.save(output_bytes, format="PNG", pnginfo=pnginfo)
        else:
            image.save(output_bytes, format="PNG") # remove pnginfo to save space
        bytes_data = output_bytes.getvalue()
        return "data:image/png;base64," + base64.b64encode(bytes_data).decode("utf-8")


def serialize_image(image):
    """
    將各種影像型別序列化為可 JSON 儲存的字典 / Serialize various image types into a JSON-safe dict.

    為什麼需要：任務排程需將影像保存為純文字，序列化後再放入任務檔案中。
    Why: scheduled tasks must persist images as plain text, so images are serialized into task files.
    """
    # 處理 numpy ndarray：儲存形狀、資料型別與壓縮後的 bytes / Handle ndarray: keep shape, dtype and compressed bytes.
    if isinstance(image, np.ndarray):
        shape = image.shape
        dtype = image.dtype
        data = base64.b64encode(zlib.compress(image.tobytes())).decode()
        return {"shape": shape, "data": data, "cls": "ndarray", "dtype": str(dtype)}
    # 處理 torch Tensor：額外記錄所在裝置以便還原 / Handle torch Tensor: also record its device for restore.
    elif isinstance(image, torch.Tensor):
        shape = image.shape
        dtype = image.dtype
        data = base64.b64encode(zlib.compress(image.detach().numpy().tobytes())).decode()
        return {
            "shape": shape,
            "data": data,
            "cls": "Tensor",
            "device": image.device.type,
            "dtype": str(dtype),
        }
    # 處理 PIL 影像：儲存尺寸、模式與壓縮 bytes / Handle PIL image: keep size, mode and compressed bytes.
    elif isinstance(image, Image.Image):
        size = image.size
        mode = image.mode
        data = base64.b64encode(zlib.compress(image.tobytes())).decode()
        return {
            "size": size,
            "mode": mode,
            "data": data,
            "cls": "Image",
        }
    # 其餘型別（如已序列化字串）原樣回傳 / Other types (e.g. already-serialized) are returned unchanged.
    else:
        return image


def deserialize_image(image_str):
    """
    將 serialize_image 產生的字典還原為原始影像物件 / Restore an image dict produced by serialize_image.

    為什麼需要：讀取已保存的任務時，需將純文字資料還原成可傳入生成管線的影像型別。
    Why: when loading saved tasks, the plain-text data must be restored into an image type usable by the pipeline.
    """
    # 僅處理含有 cls 標記的字典，否則視為非序列化資料直接回傳 / Only dicts with a "cls" tag are serialized; otherwise return as-is.
    if isinstance(image_str, dict) and image_str.get("cls", None):
        cls = image_str["cls"]
        data = zlib.decompress(base64.b64decode(image_str["data"]))

        # 還原 numpy ndarray / Restore numpy ndarray.
        if cls == "ndarray":
            # warn if required fields are missing
            # 缺少 dtype 時給預設值並記錄警告，避免崩潰 / Fall back to a default dtype and warn rather than crash.
            if image_str.get("dtype", None) is None:
                log.warning(f"Missing dtype for ndarray")
            shape = tuple(image_str["shape"])
            dtype = np.dtype(image_str.get("dtype", "uint8"))
            image = np.frombuffer(data, dtype=dtype)
            return image.reshape(shape)
        # 還原 torch Tensor，並回到原始裝置 / Restore torch Tensor and move it back to its device.
        elif cls == "Tensor":
            # 缺少 device 時回退至 cpu 並記錄警告 / Fall back to cpu and warn when device is missing.
            if image_str.get("device", None) is None:
                log.warning(f"Missing device for Tensor")
            shape = tuple(image_str["shape"])
            dtype = np.dtype(image_str.get("dtype", "uint8"))
            image_np = np.frombuffer(data, dtype=dtype)
            return torch.from_numpy(image_np.reshape(shape)).to(device=image_str.get("device", "cpu"))
        # 其餘視為 PIL 影像（Image 類別） / Otherwise treat as a PIL image.
        else:
            size = tuple(image_str["size"])
            mode = image_str["mode"]
            return Image.frombytes(mode, size, data)
    # 非序列化內容直接回傳 / Non-serialized content is returned unchanged.
    else:
        return image_str


def serialize_img2img_image_args(args: Dict):
    """
    序列化 img2img 任務中的圖片參數 / Serialize the image args inside an img2img task dict.

    為什麼需要：依當前模式只保留對應的圖片參數，其餘設為 None 以節省儲存空間。
    Why: keep only the image args relevant to the current mode and null the rest to save storage.
    """
    for mode, image_args in img2img_image_args_by_mode.items():
        for keys in image_args:
            # 非當前模式的圖片參數設為 None，減少儲存體積 / Null out image args belonging to unused modes.
            if mode != args["mode"]:
                # set None to unused image args to save space
                args[keys[0]] = None
            # 單一鍵的直接序列化該圖片 / Single-key path: serialize the image directly.
            elif len(keys) == 1:
                image = args.get(keys[0], None)
                args[keys[0]] = serialize_image(image)
            # 巢狀鍵（如 mask）則取出子字典中的圖片再序列化 / Nested key (e.g. mask): serialize the image within the sub-dict.
            else:
                value = args.get(keys[0], {})
                image = value.get(keys[1], None)
                value[keys[1]] = serialize_image(image)
                args[keys[0]] = value


def deserialize_img2img_image_args(args: Dict):
    """
    反序列化 img2img 任務中的圖片參數 / Deserialize the image args inside an img2img task dict.

    為什麼需要：僅針對當前模式還原對應的圖片參數，符合序列化時的處理邏輯。
    Why: restore only the image args for the active mode, mirroring the serialization logic.
    """
    for mode, image_args in img2img_image_args_by_mode.items():
        # 只處理與當前模式相符的圖片參數 / Only process image args matching the current mode.
        if mode != args["mode"]:
            continue

        for keys in image_args:
            # 單一鍵直接還原圖片 / Single-key path: restore the image directly.
            if len(keys) == 1:
                image = args.get(keys[0], None)
                args[keys[0]] = deserialize_image(image)
            # 巢狀鍵則還原子字典中的圖片 / Nested key: restore the image within the sub-dict.
            else:
                value = args.get(keys[0], {})
                image = value.get(keys[1], None)
                value[keys[1]] = deserialize_image(image)
                args[keys[0]] = value


def serialize_controlnet_args(cnet_unit):
    """
    將 ControlNet 單元序列化為可儲存的字典 / Serialize a ControlNet unit into a storable dict.

    為什麼需要：ControlNet 單元含 Enum 等非 JSON 型別，需轉為純值才能保存。
    Why: ControlNet units contain Enums and other non-JSON types that must be flattened to plain values.
    """
    args: Dict = cnet_unit.__dict__
    serialized_args = {"is_cnet": True}
    for k, v in args.items():
        # Enum 需取其 value 才能序列化；其餘型別直接保留 / Enums must be stored as their value; other types kept as-is.
        if isinstance(v, Enum):
            serialized_args[k] = v.value
        else:
            serialized_args[k] = v

    return serialized_args


def deserialize_controlnet_args(args: Dict):
    """
    移除 ControlNet 序列化時加入的標記欄位 / Strip the markers added during ControlNet serialization.

    為什麼需要：is_cnet / is_ui 只是序列化標記，重建單元時不應作為實際參數傳入。
    Why: is_cnet/is_ui are serialization markers and must not be passed as real unit arguments.
    """
    new_args = args.copy()
    # 移除序列化標記 is_cnet，避免被誤認為真實參數 / Drop the is_cnet marker.
    new_args.pop("is_cnet", None)
    # 移除 is_ui 標記 / Drop the is_ui marker.
    new_args.pop("is_ui", None)

    return new_args


def serialize_script_args(script_args: List):
    """
    將腳本參數清單序列化為壓縮位元組 / Serialize a script args list into compressed bytes.

    為什麼需要：腳本參數含自訂物件（如 ControlNet 單元），需轉為可 pickle 的形式再壓縮保存。
    Why: script args may contain custom objects (e.g. ControlNet units) that must be pickled then compressed.
    """
    # convert UiControlNetUnit to dict to make it serializable
    # 將 ControlNet 單元先轉成字典，才能被 pickle 序列化 / Convert ControlNet units to dicts so they can be pickled.
    for i, a in enumerate(script_args):
        if type(a).__name__ == "UiControlNetUnit":
            script_args[i] = serialize_controlnet_args(a)

    return zlib.compress(pickle.dumps(script_args))


def deserialize_script_args(script_args: Union[bytes, List], UiControlNetUnit = None):
    """
    將腳本參數反序列化，並重建 ControlNet 單元 / Deserialize script args and rebuild ControlNet units.

    為什麼需要：還原任務時需把壓縮位元組解回參數清單，並將標記字典還原為 ControlNet 單元。
    Why: restoring tasks requires decompressing bytes back into args and rebuilding ControlNet units from marker dicts.
    """
    # 若傳入的是壓縮位元組則先解壓縮並反 pickle / If given compressed bytes, decompress and unpickle first.
    if type(script_args) is bytes:
        script_args = pickle.loads(zlib.decompress(script_args))

    for i, a in enumerate(script_args):
        # 僅處理標記為 ControlNet 的字典 / Only handle dicts flagged as ControlNet.
        if isinstance(a, dict) and a.get("is_cnet", False):
            unit = deserialize_controlnet_args(a)
            skip_controlnet = False
            # 若提供 ControlNet 型別，則嘗試重建單元並驗證 Enum 值 / If the unit type is available, rebuild and validate enums.
            if UiControlNetUnit is not None:
                u = UiControlNetUnit()
                for k, v in unit.items():
                    if isinstance(getattr(u, k, None), Enum):
                        # check if v is a valid enum value
                        # 檢查 v 是否為合法的 Enum 值，避免無效資料導致重建失敗 / Validate v against the enum to avoid bad rebuilds.
                        enum_obj: Enum= getattr(u, k)
                        if v not in [e.value for e in enum_obj.__class__]:
                            log.error(f"Invalid enum value {v} for {k} encountered, valid value is {enum_obj.__class__}")
                            skip_controlnet = True
                            break
                        # 將合法值轉回對應 Enum 型別 / Convert the valid value back into its Enum type.
                        unit[k] = type(getattr(u, k))(v)
                # 所有 Enum 皆合法才重建單元 / Rebuild the unit only when all enums were valid.
                if not skip_controlnet: # valid 
                    unit = UiControlNetUnit(**unit)
            # 通過驗證才用重建後的單元取代原字典 / Replace the dict with the rebuilt unit when valid.
            if not skip_controlnet: # valid
                script_args[i] = unit

    return script_args


def map_controlnet_args_to_api_task_args(args: Dict):
    """
    將 ControlNet 參數對應為 API 任務參數格式 / Map ControlNet args into API task arg format.

    為什麼需要：API 需要 base64 圖片與純值 Enum，故在此統一轉換影像與列舉欄位。
    Why: the API expects base64 images and plain enum values, so images and enums are converted here.
    """
    # 若傳入的是 ControlNet 單元物件，先取其实例字典 / If given a unit object, use its instance dict first.
    if type(args).__name__ == "UiControlNetUnit":
        args = args.__dict__

    for k, v in args.items():
        # 影像欄位需轉為 base64（含可選 mask） / The image field must be base64-encoded (with optional mask).
        if k == "image" and v is not None:
            args[k] = {
                "image": encode_image_to_base64(v["image"]),
                "mask": encode_image_to_base64(v["mask"]) if v.get("mask", None) is not None else None,
            }
        # Enum 欄位轉為其 value 純值 / Enum fields are flattened to their value.
        if isinstance(v, Enum):
            args[k] = v.value

    return args


def map_ui_task_args_list_to_named_args(args: List, is_img2img: bool):
    """
    將 UI 任務參數清單對應為具名參數與腳本參數 / Map a UI task arg list into named args plus script args.

    為什麼需要：sd-webui 內部函式以位置參數接收，需依簽章把清單拆分為具名與腳本兩部分。
    Why: sd-webui internal functions take positional args, so the list is split into named args and script args by signature.
    """
    # 依 img2img 或 txt2img 選擇對應的處理函式（相容舊版無 create_processing 的環境）/ Pick the processing fn, falling back to the old name for compatibility.
    fn = (
        getattr(img2img, "img2img_create_processing", img2img.img2img)
        if is_img2img
        else getattr(txt2img, "txt2img_create_processing", txt2img.txt2img)
    )
    arg_names = inspect.getfullargspec(fn).args

    # SD WebUI 1.5.0 has new request arg
    # 新版 WebUI 多了 request 參數，若簽章含 request 則插入 None 佔位 / Newer WebUI adds a "request" arg; insert a None placeholder if present.
    if "request" in arg_names:
        args.insert(arg_names.index("request"), None)

    # 依函式簽章切分具名參數與剩餘的腳本參數 / Split named args from the remaining script args by the signature.
    named_args = dict(zip(arg_names, args[0 : len(arg_names)]))
    script_args = args[len(arg_names) :]

    override_settings_texts: List[str] = named_args.get("override_settings_texts") or []
    # add clip_skip if not exist in args (vlad fork has this arg)
    # 若參數中缺少 clip_skip，則從設定中補上（部分 fork 才有此參數）/ If clip_skip is missing, backfill it from options (some forks have this arg).
    if named_args.get("clip_skip", None) is None:
        clip_skip = next((s for s in override_settings_texts if s.startswith("Clip skip:")), None)
        if clip_skip is None and hasattr(shared.opts, "CLIP_stop_at_last_layers"):
            override_settings_texts.append(f"Clip skip: {shared.opts.CLIP_stop_at_last_layers}")

    named_args["override_settings_texts"] = override_settings_texts

    # 將 sampler_index 轉為可讀的 sampler_name / Convert sampler_index into a human-readable sampler_name.
    sampler_index = named_args.get("sampler_index", None)
    if sampler_index is not None:
        available_samplers = sd_samplers.samplers_for_img2img if is_img2img else sd_samplers.samplers
        sampler_name = available_samplers[named_args["sampler_index"]].name
        named_args["sampler_name"] = sampler_name
        log.debug(f"serialize sampler index: {str(sampler_index)} as {sampler_name}")

    return (
        named_args,
        script_args,
    )


def map_named_args_to_ui_task_args_list(named_args: Dict, script_args: List, is_img2img: bool):
    """
    將具名參數還原為 UI 任務參數清單（map_ui_task_args_list_to_named_args 的反向操作）/ Rebuild a UI task arg list from named args.

    為什麼需要：還原任務時需把具名參數與腳本參數重新組合成 sd-webui 內部函式所需的位置清單。
    Why: restoring tasks must reassemble named args and script args into the positional list the internal fn expects.
    """
    # 依 img2img 或 txt2img 選擇對應處理函式（相容舊版環境）/ Pick the processing fn, falling back to the old name for compatibility.
    fn = (
        getattr(img2img, "img2img_create_processing", img2img.img2img)
        if is_img2img
        else getattr(txt2img, "txt2img_create_processing", txt2img.txt2img)
    )
    arg_names = inspect.getfullargspec(fn).args

    # 將 sampler_name 轉回對應的索引值 / Convert sampler_name back into its index.
    sampler_name = named_args.get("sampler_name", None)
    if sampler_name is not None:
        available_samplers = sd_samplers.samplers_for_img2img if is_img2img else sd_samplers.samplers
        sampler_index = next((i for i, x in enumerate(available_samplers) if x.name == sampler_name), 0)
        named_args["sampler_index"] = sampler_index

    # 依簽章依序取出具名參數，再串接腳本參數 / Collect named args by signature then append script args.
    args = [named_args.get(name, None) for name in arg_names]
    args.extend(script_args)

    return args


def map_script_args_list_to_named(script: scripts.Script, args: List):
    """
    將腳本的位置參數清單對應為具名參數 / Map a script's positional args into named args.

    為什麼需要：ControlNet 與一般腳本的參數結構不同，需個別處理成 API 可接受的具名格式。
    Why: ControlNet and normal scripts have different arg shapes, so each is mapped into API-friendly named args.
    """
    script_name = script.title().lower()

    # ControlNet 需先將每個單元對應為 API 格式 / ControlNet needs each unit mapped into API format first.
    if script_name == "controlnet":
        for i, cnet_args in enumerate(args):
            args[i] = map_controlnet_args_to_api_task_args(cnet_args)

        return args

    # 一般腳本：依 process/run 簽章（略過前兩個 self/args 參數）對應 / Normal scripts: map by process/run signature, skipping self and the args param.
    fn = script.process if script.alwayson else script.run
    inspection = inspect.getfullargspec(fn)
    arg_names = inspection.args[2:]
    named_script_args = dict(zip(arg_names, args[: len(arg_names)]))
    # 若有 *args 可變參數，將剩餘部分放入對應鍵 / If the script accepts *args, store the remainder under that key.
    if inspection.varargs is not None:
        named_script_args[inspection.varargs] = args[len(arg_names) :]

    return named_script_args


def map_named_script_args_to_list(script: scripts.Script, named_args: Union[dict, list]):
    """
    將具名腳本參數對應回位置參數清單（map_script_args_list_to_named 的反向操作）/ Rebuild a script arg list from named args.

    為什麼需要：還原任務時需把 API 具名格式重新組回 sd-webui 腳本所需的位置清單。
    Why: restoring tasks must reassemble API named args back into the positional list the script expects.
    """
    script_name = script.title().lower()

    # 具名字典：依簽章逐一取回值並組成清單 / Dict form: collect values by signature into a list.
    if isinstance(named_args, dict):
        fn = script.process if script.alwayson else script.run
        inspection = inspect.getfullargspec(fn)
        arg_names = inspection.args[2:]
        args = [named_args.get(name, None) for name in arg_names]
        # 若有 *args 可變參數，補上剩餘清單 / If *args present, append the remaining list.
        if inspection.varargs is not None:
            args.extend(named_args.get(inspection.varargs, []))

        return args

    # 清單形式：ControlNet 需先對應為 API 格式 / List form: ControlNet units are mapped into API format first.
    if isinstance(named_args, list):
        if script_name == "controlnet":
            for i, cnet_args in enumerate(named_args):
                named_args[i] = map_controlnet_args_to_api_task_args(cnet_args)

        return named_args


def map_ui_task_args_to_api_task_args(named_args: Dict, script_args: List, is_img2img: bool):
    """
    將 UI 任務參數對應為 sd-webui API 任務參數格式 / Map UI task args into sd-webui API task args.

    為什麼需要：API 使用的欄位名稱與預處理邏輯與 UI 不同（如 styles、sampler_name、override_settings），
    且 img2img 各模式需轉成 init_images/mask。此函式統一完成這些轉換以便呼叫 API。
    Why: the API uses different field names and preprocessing (styles, sampler_name, override_settings) and
    img2img modes must be normalized to init_images/mask; this centralizes those conversions for API calls.
    """
    api_task_args: Dict = named_args.copy()

    # 將 prompt_styles 改名為 styles 以符合 API 欄位 / Rename prompt_styles to styles for the API.
    prompt_styles = api_task_args.pop("prompt_styles", [])
    api_task_args["styles"] = prompt_styles

    # 將 sampler_index 轉為 sampler_name / Convert sampler_index into sampler_name.
    sampler_index = api_task_args.pop("sampler_index", 0)
    api_task_args["sampler_name"] = sd_samplers.samplers[sampler_index].name

    # 將 override_settings_texts 轉為 override_settings 字典 / Convert override_settings_texts into an override_settings dict.
    override_settings_texts = api_task_args.pop("override_settings_texts", [])
    api_task_args["override_settings"] = create_override_settings_dict(override_settings_texts)

    if is_img2img:
        # 取出當前 img2img 模式 / Read the current img2img mode.
        mode = api_task_args.pop("mode", 0)
        # 移除非當前模式所屬的圖片參數，避免多餘欄位 / Drop image args belonging to other modes to avoid extra fields.
        for arg_mode, image_args in img2img_image_args_by_mode.items():
            if mode != arg_mode:
                for keys in image_args:
                    api_task_args.pop(keys[0], None)

        # the logic below is copied from modules/img2img.py
        # 依不同 img2img 模式從對應欄位取出 image 與 mask / Extract image and mask per img2img mode.
        if mode == 0:
            image = api_task_args.pop("init_img")
            image = image.convert("RGB") if image else None
            mask = None
        elif mode == 1:
            image = api_task_args.pop("sketch")
            image = image.convert("RGB") if image else None
            mask = None
        elif mode == 2:
            init_img_with_mask: Dict = api_task_args.pop("init_img_with_mask") or {}
            image = init_img_with_mask.get("image", None)
            image = image.convert("RGB") if image else None
            mask = init_img_with_mask.get("mask", None)
            # 由影像 alpha 通道合成遮罩，確保與原 mask 對齊 / Build the mask from the alpha channel and merge with the provided mask.
            if mask:
                alpha_mask = (
                    ImageOps.invert(image.split()[-1]).convert("L").point(lambda x: 255 if x > 0 else 0, mode="1")
                )
                mask = ImageChops.lighter(alpha_mask, mask.convert("L")).convert("L")
        elif mode == 3:
            image = api_task_args.pop("inpaint_color_sketch")
            orig = api_task_args.pop("inpaint_color_sketch_orig") or image
            if image is not None:
                mask_alpha = api_task_args.pop("mask_alpha", 0)
                mask_blur = api_task_args.get("mask_blur", 4)
                # 比對影像與原圖差異以產生遮罩 / Derive the mask from pixel differences between image and original.
                pred = np.any(np.array(image) != np.array(orig), axis=-1)
                mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
                mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
                blur = ImageFilter.GaussianBlur(mask_blur)
                image = Image.composite(image.filter(blur), orig, mask.filter(blur))
                image = image.convert("RGB")
        elif mode == 4:
            image = api_task_args.pop("init_img_inpaint")
            mask = api_task_args.pop("init_mask_inpaint")
        else:
            raise Exception(f"Batch mode is not supported yet")

        # 修正 EXIF 方向並將 image/mask 編碼為 base64 / Fix EXIF orientation then encode image/mask as base64.
        image = ImageOps.exif_transpose(image) if image else None
        api_task_args["init_images"] = [encode_image_to_base64(image)] if image else []
        api_task_args["mask"] = encode_image_to_base64(mask) if mask else None

        # 若使用「按比例縮放」分頁，依 scale_by 計算目標寬高 / When the scale-by tab is active, derive width/height from scale_by.
        selected_scale_tab = api_task_args.pop("selected_scale_tab", 0)
        scale_by = api_task_args.get("scale_by", 1)
        if selected_scale_tab == 1 and image:
            api_task_args["width"] = int(image.width * scale_by)
            api_task_args["height"] = int(image.height * scale_by)
    else:
        # txt2img 的高解析度修復取樣器索引轉為名稱 / Convert hr_sampler_index into hr_sampler_name for txt2img.
        hr_sampler_index = api_task_args.pop("hr_sampler_index", 0)
        api_task_args["hr_sampler_name"] = (
            sd_samplers.samplers_for_img2img[hr_sampler_index - 1].name if hr_sampler_index != 0 else None
        )

    # script
    # 選擇對應的 script runner 並取出選用腳本 / Pick the script runner and read the selectable script id.
    script_runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img
    script_id = script_args[0]
    # script_id 為 0 代表未選用腳本 / script_id 0 means no selectable script was chosen.
    if script_id == 0:
        api_task_args["script_name"] = None
        api_task_args["script_args"] = []
    else:
        script: scripts.Script = script_runner.selectable_scripts[script_id - 1]
        api_task_args["script_name"] = script.title().lower()
        current_script_args = script_args[script.args_from : script.args_to]
        api_task_args["script_args"] = map_script_args_list_to_named(script, current_script_args)

    # alwayson scripts
    # 確保 alwayson_scripts 為字典，避免後續處理時為空 / Ensure alwayson_scripts is a dict to avoid later None handling.
    alwayson_scripts = api_task_args.get("alwayson_scripts", None)
    if not alwayson_scripts:
        api_task_args["alwayson_scripts"] = {}
        alwayson_scripts = api_task_args["alwayson_scripts"]

    # 將每個 alwayson 腳本的參數對應為具名格式（跳過本排程器自身）/ Map each alwayson script's args to named form, skipping the scheduler's own script.
    for script in script_runner.alwayson_scripts:
        alwayson_script_args = script_args[script.args_from : script.args_to]
        script_name = script.title().lower()
        if script_name != "agent scheduler":
            named_script_args = map_script_args_list_to_named(script, alwayson_script_args)
            alwayson_scripts[script_name] = {"args": named_script_args}

    return api_task_args


def serialize_api_task_args(
    params: Dict,
    is_img2img: bool,
    checkpoint: str = None,
    vae: str = None,
) -> Dict:
    """
    將 API 任務參數序列化為 API 模型可接受並可 dict 化的格式 / Serialize API task args into a dict-compatible API model.

    為什麼需要：在呼叫 sd-webui API 模型前，需把具名腳本參數、alwayson 參數、checkpoint/VAE 覆寫與
    img2img 圖片都準備好，最後產出模型所需的 dict。
    Why: before invoking the sd-webui API model we must resolve named/alwayson script args, checkpoint/VAE
    overrides and img2img images, then produce the dict the model expects.
    """
    # handle named script args
    # 將選用腳本的具名參數轉回列表形式 / Convert the selectable script's named args back into a list.
    script_name = params.get("script_name", None)
    if script_name is not None and script_name != "":
        script = get_script_by_name(script_name, is_img2img)
        # 找不到腳本則拋錯，避免產生無效任務 / Raise if the script cannot be resolved, to avoid invalid tasks.
        if script is None:
            raise Exception(f"Not found script {script_name}")

        script_args = params.get("script_args", {})
        params["script_args"] = map_named_script_args_to_list(script, script_args)

    # handle named alwayson script args
    # 取得 alwayson 腳本參數並確保為字典 / Read alwayson script args and ensure they are a dict.
    alwayson_scripts = get_dict_attribute(params, "alwayson_scripts", {})
    assert type(alwayson_scripts) is dict

    script_runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img
    # 建立允許的 alwayson 腳本對照表（以小寫名稱為鍵）/ Build an allow-list of alwayson scripts keyed by lowercase name.
    allowed_alwayson_scripts = {s.title().lower(): s for s in script_runner.alwayson_scripts}

    valid_alwayson_scripts = {}
    for script_name, script_args in alwayson_scripts.items():
        # 跳過排程器自身的腳本 / Skip the scheduler's own script.
        if script_name.lower() == "agent scheduler":
            continue

        # 過濾不存在於允許清單的腳本，避免傳入無效名稱 / Filter out scripts not in the allow-list.
        if script_name.lower() not in allowed_alwayson_scripts:
            log.warning(f"Script {script_name} is not in script_runner.alwayson_scripts")
            continue

        script = allowed_alwayson_scripts[script_name.lower()]
        script_args = get_dict_attribute(script_args, "args", [])
        arg_list = map_named_script_args_to_list(script, script_args)
        valid_alwayson_scripts[script_name] = {"args": arg_list}

    # 以過濾後的 alwayson 腳本取代原參數 / Replace with the filtered alwayson scripts.
    params["alwayson_scripts"] = valid_alwayson_scripts

    # 依 img2img 或 txt2img 建立對應的 API 模型實例 / Build the matching API model for img2img or txt2img.
    args = (
        StableDiffusionImg2ImgProcessingAPI(**params) if is_img2img else StableDiffusionTxt2ImgProcessingAPI(**params)
    )

    # 確保 override_settings 為字典，便於後續加入覆寫 / Ensure override_settings is a dict so overrides can be added.
    if args.override_settings is None:
        args.override_settings = {}

    # 若有指定 checkpoint，則加入 override（找不到時退回系統模型）/ Apply checkpoint override, falling back to system model if not found.
    if checkpoint is not None:
        checkpoint_info: CheckpointInfo = get_closet_checkpoint_match(checkpoint)
        if not checkpoint_info:
            log.warning(f"Checkpoint {checkpoint} not found, use current system model")
        else:
            args.override_settings["sd_model_checkpoint"] = checkpoint_info.title

    # 若有指定 VAE，則加入 override（找不到時退回系統 VAE）/ Apply VAE override, falling back to system VAE if not found.
    if vae is not None:
        if vae not in sd_vae.vae_dict:
            log.warning(f"VAE {vae} not found, use current system vae")
        else:
            args.override_settings["sd_vae"] = vae

    # load images from url or file if needed
    # img2img 需將圖片與遮罩編碼為 base64，並處理批次大小 / For img2img, encode init images and mask as base64 and handle batch size.
    if is_img2img:
        init_images = args.init_images
        # 至少需要一張初始圖片，否則無法生成 / At least one init image is required to generate.
        if len(init_images) == 0:
            raise Exception("At least one init image is required")

        # 逐張編碼初始圖片 / Encode each init image to base64.
        for i, image in enumerate(init_images):
            init_images[i] = encode_image_to_base64(image)

        # 編碼遮罩（可能為 None）/ Encode the mask (which may be None).
        args.mask = encode_image_to_base64(args.mask)
        # 多張圖片時自動設定 batch_size / Set batch_size when multiple init images are provided.
        if len(init_images) > 1:
            args.batch_size = len(init_images)

    return args.dict()
