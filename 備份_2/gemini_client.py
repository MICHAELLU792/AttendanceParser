# python
from typing import Optional, Any
import threading

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

from app_config import ENV_API_KEY
from config.logging_gemini import log_gemini_usage


def ensure_genai_installed() -> None:
    if genai is None:
        raise RuntimeError("缺少 google-generativeai 套件，請先安裝：pip install -U google-generativeai")


def list_available_models() -> list[str]:
    try:
        ensure_genai_installed()
        models = genai.list_models()
        available = []
        for m in models:
            if hasattr(m, "supported_generation_methods") and "generateContent" in m.supported_generation_methods:
                model_name = getattr(m, "name", "").replace("models/", "")
                if model_name:
                    available.append(model_name)
        return sorted(available)
    except Exception:
        # fallback：提供常用模型清單（含 flash-lite）
        return ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]


def build_instructions() -> str:
    from app_config import CSV_HEADERS
    header = ", ".join(CSV_HEADERS)
    return (
        # 角色與輸出格式
        "你是嚴謹的出勤/請假表解析器，輸出內容只能是 CSV 純文字。\n"
        f"第一列欄位必須為：{header}\n"
        f"之後每一列都要剛好 {len(CSV_HEADERS)} 欄，以英文逗號 (,) 分隔；沒有值的欄位留空但保留逗號。\n"
        "禁止使用全形逗號、分號或 tab，欄位順序不可更動，也不能多增加自訂欄位。\n\n"
        # 日期與時間
        "日期一律使用 YYYY-MM-DD（民國年需 +1911 轉成西元）。\n"
        "時間一律使用 24 小時制 HH:MM，例如 09:00、18:30。\n"
        "請假時數或天數只在原文明確提供時填寫，不得自行換算推估。\n"
        "同一筆請假就算跨多日，也維持一列資料，不拆成多列。\n\n"
        # 分類規則
        "【記錄類型判斷】\n"
        "- 出勤：文字中有 上班/下班/打卡/刷卡/遲到/早退/加班/班別/工號 等關鍵字 → 記錄類型=出勤。\n"
        "- 請假：文字中有 請假單/假別/申請人/代理人/主管/核准/起訖日期/時數/天數 等 → 記錄類型=請假。\n"
        "若內容模糊無法判定，就直接略過，不要硬塞成請假或出勤。\n\n"
        # 出勤列
        "【出勤列】\n"
        "- 記錄類型=出勤。\n"
        "- 填入：派駐單位、姓名、日期、上班時間、下班時間。\n"
        "- 所有請假相關欄位（假別、請假起日/迄日、時間、時數、天數）都留空。\n\n"
        # 請假列
        "【請假列】\n"
        "- 記錄類型=請假。\n"
        "- 日期 欄位填『請假起日』；請假起日/迄日 依文件標示的起迄日期填寫（跨日仍是一列）。\n"
        "- 假別 正規化為以下其中一種：事假, 病假, 特休, 公假, 喪假, 婚假, 產假, 陪產假, 育嬰假, 家庭照顧假, 補休, 半薪病假, 其他。\n"
        "- 若有請假時間區間，填入『請假時間(起)』與『請假時間(迄)』。\n"
        "- 若原文給了請假時數，填入『請假時數(小時)』；若給了天數，填入『請假天數(天)』；若同時都有，兩欄皆可填。\n"
        "- 跨午夜的區間（例如 22:00–02:00），日期區間使用起日與次日，但仍維持一列資料。\n\n"
        # 備註
        "【備註】\n"
        "備註欄只放原文中的補充說明：例如單據號、簽核意見、原始假別文字、特殊說明等。\n"
        "不要在備註重複填已經出現在其他欄位（日期、時間、假別、時數/天數）中的資訊。\n"
    )


def _resp_text_safe(resp: Any) -> str:
    try:
        if hasattr(resp, "text"):
            return (resp.text or "").strip()
        if isinstance(resp, dict):
            return (resp.get("text", "") or "").strip()
        return str(resp)
    except Exception:
        return ""


# Internal functions that assume genai is already configured appropriately
def _generate_image_csv_internal(image_bytes: bytes, model: str = "gemini-2.0-flash-lite", source: str = "") -> str:
    instructions = build_instructions()
    img_part = {"mime_type": "image/jpeg", "data": image_bytes}
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(
        [instructions, img_part],
        generation_config=genai.GenerationConfig(temperature=0),
    )
    text = _resp_text_safe(resp)
    try:
        usage = getattr(resp, "usage_metadata", None)
        log_gemini_usage(model, usage, uploaded_filename=source or "", extra_info=instructions[:500])
    except Exception:
        pass
    return text


def _generate_text_csv_internal(text: str, model: str = "gemini-pro", source: str = "") -> str:
    instructions = build_instructions()
    prompt = instructions + "以下為從 PDF 或 OCR 取得的純文字內容（可能包含表格展平）：\n\n" + text[:200000]
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0),
    )
    out_text = _resp_text_safe(resp)
    try:
        usage = getattr(resp, "usage_metadata", None)
        log_gemini_usage(model, usage, uploaded_filename=source or "", extra_info=prompt[:1000])
    except Exception:
        pass
    return out_text


# Worker wrappers that use the global semaphores/lock. They won't block submission beyond the bounded queue.
def call_gemini_for_image_csv_worker(image_bytes: bytes, model: str, source: str, api_key: str, lock: threading.Lock, sem: Optional[threading.Semaphore]) -> str:
    ensure_genai_installed()
    if not api_key:
        raise RuntimeError("缺少 GEMINI API Key")
    if sem:
        sem.acquire()
    try:
        # 只有在需要時覆寫 global configure；若 ENV_API_KEY 與使用者 key 相同則跳過重設
        need_configure = (api_key != ENV_API_KEY)
        if need_configure:
            with lock:
                genai.configure(api_key=api_key)
                return _generate_image_csv_internal(image_bytes, model=model, source=source)
        else:
            # 已在啟動時設定好或使用環境 key，直接呼叫
            return _generate_image_csv_internal(image_bytes, model=model, source=source)
    finally:
        if sem:
            sem.release()


def call_gemini_for_text_csv_worker(text: str, model: str, source: str, api_key: str, lock: threading.Lock, sem: Optional[threading.Semaphore]) -> str:
    ensure_genai_installed()
    if not api_key:
        raise RuntimeError("缺少 GEMINI API Key")
    if sem:
        sem.acquire()
    try:
        need_configure = (api_key != ENV_API_KEY)
        if need_configure:
            with lock:
                genai.configure(api_key=api_key)
                return _generate_text_csv_internal(text, model=model, source=source)
        else:
            return _generate_text_csv_internal(text, model=model, source=source)
    finally:
        if sem:
            sem.release()

