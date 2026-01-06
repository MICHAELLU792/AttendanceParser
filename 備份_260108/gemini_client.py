# python
from typing import Optional, Any
import threading
import time

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
    # 生成详细的字段位置说明
    header_with_index = "\n".join([f"  {i+1}. {col}" for i, col in enumerate(CSV_HEADERS)])
    return (
        # 角色與輸出格式
        "你是嚴謹的出勤/請假表解析器，輸出內容只能是 CSV 純文字。\n"
        f"第一列欄位必須為（嚴格按照此順序，不可更改）：\n{header_with_index}\n\n"
        f"之後每一列都要剛好 {len(CSV_HEADERS)} 欄，以英文逗號 (,) 分隔；沒有值的欄位留空但保留逗號。\n"
        "【極重要】欄位順序絕對不可更動！每個欄位必須放在正確的位置：\n"
        "- 第1欄：記錄類型\n"
        "- 第2欄：派駐單位\n"
        "- 第3欄：姓名\n"
        "- 第4欄：日期\n"
        "- 第5欄：上班時間\n"
        "- 第6欄：下班時間\n"
        "- 第7欄：總時數\n"
        "- 第8欄：假別\n"
        "- 第9欄：請假起日\n"
        "- 第10欄：請假迄日\n"
        "- 第11欄：請假時間(起)\n"
        "- 第12欄：請假時間(迄)\n"
        "- 第13欄：請假時數(小時)\n"
        "- 第14欄：請假天數(天)\n"
        "- 第15欄：加班起日\n"
        "- 第16欄：加班迄日\n"
        "- 第17欄：加班時間(起) ← 必須放在第17欄，不能放在其他位置！\n"
        "- 第18欄：加班時間(迄) ← 必須放在第18欄，不能放在其他位置！\n"
        "- 第19欄：加班時數(分鐘) ← 必須放在第19欄，不能放在其他位置！\n"
        "- 第20欄：備註\n"
        "禁止使用全形逗號、分號或 tab，也不能多增加自訂欄位。\n\n"
        # 日期與時間
        "日期一律使用 YYYY-MM-DD 格式（民國年必須 +1911 轉成西元年）。\n"
        "重要：民國年轉換範例 - 「114年11月10日」應轉換為「2025-11-10」（114+1911=2025，月份11，日期10）。\n"
        "民國年轉換規則：民國年 + 1911 = 西元年，必須完整保留年月日，不可遺漏或改變。\n"
        "打卡鐘日期特別處理：\n"
        "- 年份和月份通常解析正確，但「日」（日期數字）可能需要特別注意。\n"
        "- 【極重要】如果日期數字在圖像中呈旋轉狀態（例如往左90度倒置），必須仔細識別並正確解析。\n"
        "- 往左90度倒置的數字辨識規則：\n"
        "  * 數字「2」倒置後可能看起來像「5」或「S」\n"
        "  * 數字「3」倒置後可能看起來像「E」或「ε」\n"
        "  * 數字「5」倒置後可能看起來像「2」或「Z」\n"
        "  * 數字「6」倒置後可能看起來像「9」或「g」\n"
        "  * 數字「9」倒置後可能看起來像「6」或「P」\n"
        "  * 數字「0」倒置後可能看起來像「0」或「O」\n"
        "  * 數字「1」倒置後可能看起來像「1」或「I」\n"
        "  * 數字「4」倒置後可能看起來像「h」或「4」\n"
        "  * 數字「7」倒置後可能看起來像「L」或「7」\n"
        "  * 數字「8」倒置後可能看起來像「8」或「∞」\n"
        "- 旋轉的數字可能看起來像其他數字或字母，必須根據以下原則判斷：\n"
        "  * 日期數字只能是0-31之間的數字（根據月份有28/29/30/31天的限制）\n"
        "  * 參考上下文的日期序列進行推斷（例如：如果前一天是10日，後一天是12日，中間的倒置數字很可能是11日）\n"
        "  * 如果倒置數字看起來像字母，但不符合日期邏輯，應該根據上下文和日期範圍推斷正確的數字\n"
        "  * 特別注意：如果看到的字符在正常方向是字母，但倒置後是數字，應該採用倒置後的數字值\n"
        "- 如果日期數字模糊不清，可參考上下文的日期序列進行推斷。\n"
        "- 範例：如果看到「11月1S日」，且1S看起來是倒置的，應該解析為「11月12日」（S倒置後是2）。\n"
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
        "- 總時數欄位由系統自動計算，不需填寫。\n"
        "- 所有請假相關欄位（假別、請假起日/迄日、時間、時數、天數）都留空。\n"
        "- 如果該筆記錄沒有加班，則所有加班相關欄位（加班起日/迄日、時間、時數）都留空。\n"
        "- 如果該筆記錄有加班，則必須填入加班相關欄位（見下方【加班列】說明）。\n\n"
        # 請假列
        "【請假列】\n"
        "- 記錄類型=請假。\n"
        "- 日期 欄位填『請假起日』；請假起日/迄日 依文件標示的起迄日期填寫（跨日仍是一列）。\n"
        "- 假別 正規化為以下其中一種：事假, 病假, 特休, 公假, 喪假, 婚假, 產假, 陪產假, 育嬰假, 家庭照顧假, 補休, 半薪病假, 其他。\n"
        "- 若有請假時間區間，填入『請假時間(起)』與『請假時間(迄)』。\n"
        "- 若原文給了請假時數，填入『請假時數(小時)』；若給了天數，填入『請假天數(天)』；若同時都有，兩欄皆可填。\n"
        "- 跨午夜的區間（例如 22:00–02:00），日期區間使用起日與次日，但仍維持一列資料。\n"
        "- 上班時間、下班時間、總時數欄位留空。\n"
        "- 所有加班相關欄位（加班起日/迄日、時間、時數）都留空。\n\n"
        # 加班列（新增）
        "【加班列】\n"
        "- 記錄類型=出勤（與出勤列相同）。\n"
        "- 若文件中有明確的加班記錄，在出勤列的基礎上，額外填入加班相關欄位。\n"
        "- 加班起日（第15欄）：依文件標示的加班起迄日期填寫（跨日仍是一列）。\n"
        "- 加班迄日（第16欄）：依文件標示的加班迄日期填寫。\n"
        "- 加班時間(起)（第17欄）：這是必填欄位！如果文件中有加班記錄，必須填入加班開始時間（24小時制 HH:MM，例如 18:00）。即使原文沒有明確寫出時間，也要根據上下文推斷並填入。\n"
        "- 加班時間(迄)（第18欄）：這是必填欄位！如果文件中有加班記錄，必須填入加班結束時間（24小時制 HH:MM，例如 20:30）。即使原文沒有明確寫出時間，也要根據上下文推斷並填入。\n"
        "- 加班時數(分鐘)（第19欄）：若原文給了加班時數，填入此欄位（以分鐘為單位，純數字，如「56」，不要使用「00:56」這種時間格式）。若沒有給，可留空由系統自動計算。\n"
        "- 【極重要】欄位位置絕對不能錯：\n"
        "  * 「加班時間(起)」必須放在第17欄（「加班時間(起)」欄位），絕對不能放在第18欄或其他位置！\n"
        "  * 「加班時間(迄)」必須放在第18欄（「加班時間(迄)」欄位），絕對不能放在第17欄、第19欄或其他位置！\n"
        "  * 「加班時數(分鐘)」必須放在第19欄（「加班時數(分鐘)」欄位），絕對不能放在第17欄、第18欄或其他位置！\n"
        "- 若同一筆記錄既有出勤又有加班，則在同一列中同時填入出勤和加班相關欄位。\n"
        "- 重要：只要有加班記錄，就必須同時填入「加班時間(起)」和「加班時間(迄)」兩個欄位，不能留空！\n\n"
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
    
    # 重试机制：最多重试3次，针对429错误
    max_retries = 3
    base_delay = 2  # 初始延迟2秒
    
    for attempt in range(max_retries):
        try:
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
        except Exception as e:
            error_str = str(e)
            # 检查是否为429错误
            if "429" in error_str or "Resource exhausted" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    # 指数退避：2秒, 4秒, 8秒
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    # 最后一次重试失败，抛出带详细信息的异常
                    raise RuntimeError(f"API資源耗盡（429錯誤）。請稍後再試或減少同時處理的檔案數量。詳細錯誤：{error_str}")
            else:
                # 非429错误，直接抛出
                raise


def _generate_text_csv_internal(text: str, model: str = "gemini-pro", source: str = "") -> str:
    instructions = build_instructions()
    prompt = instructions + "以下為從 PDF 或 OCR 取得的純文字內容（可能包含表格展平）：\n\n" + text[:200000]
    model_obj = genai.GenerativeModel(model)
    
    # 重试机制：最多重试3次，针对429错误
    max_retries = 3
    base_delay = 2  # 初始延迟2秒
    
    for attempt in range(max_retries):
        try:
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
        except Exception as e:
            error_str = str(e)
            # 检查是否为429错误
            if "429" in error_str or "Resource exhausted" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    # 指数退避：2秒, 4秒, 8秒
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    # 最后一次重试失败，抛出带详细信息的异常
                    raise RuntimeError(f"API資源耗盡（429錯誤）。請稍後再試或減少同時處理的檔案數量。詳細錯誤：{error_str}")
            else:
                # 非429错误，直接抛出
                raise


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

