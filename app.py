# python
import io
import os
import csv
import re
from typing import List, Optional, Any
import threading
import concurrent.futures
import time

import pandas as pd
import streamlit as st

from config.logging_gemini import log_gemini_usage

# PDF text extraction
try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

# Google Gemini SDK
try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

# PDF rendering (PyMuPDF)
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

# OCR (pytesseract + PIL)
try:
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover
    pytesseract = None
    Image = None


APP_TITLE = "出勤表解析系統"
CSV_HEADERS = [
    "記錄類型",
    "派駐單位",
    "姓名",
    "日期",
    "上班時間",
    "下班時間",
    "假別",
    "請假起日",
    "請假迄日",
    "請假時間(起)",
    "請假時間(迄)",
    "請假時數(小時)",
    "請假天數(天)",
    "備註",
]

# ========== 全域併發控制（可由環境變數微調） ==========
GLOBAL_MAX_WORKERS = int(os.getenv("APP_MAX_WORKERS", "8"))  # 代表可以接受的背景執行緒數（整個程式）
GLOBAL_PENDING_FACTOR = int(os.getenv("APP_PENDING_FACTOR", "4"))  # 每個 worker 的允許待處理任務倍數
GLOBAL_MAX_PENDING = max(16, GLOBAL_MAX_WORKERS * GLOBAL_PENDING_FACTOR)
GLOBAL_API_CONCURRENCY = int(os.getenv("APP_API_CONCURRENCY", "4"))  # 同時對 Gemini 的並發請求上限（process 內）

GLOBAL_SUBMIT_SEMAPHORE = threading.BoundedSemaphore(GLOBAL_MAX_PENDING)
GLOBAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=GLOBAL_MAX_WORKERS)
GLOBAL_API_SEMAPHORE = threading.Semaphore(GLOBAL_API_CONCURRENCY)
GENAI_LOCK = threading.Lock()

# 嘗試在 process 啟動時用環境變數 key 做一次 global configure（減少頻繁覆寫）
ENV_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if ENV_API_KEY:
    try:
        if genai is not None:
            genai.configure(api_key=ENV_API_KEY)
    except Exception:
        # 忽略配置錯誤，工作時會再檢查
        pass

# ========== session defaults ==========
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state["GEMINI_API_KEY"] = None
if "input_gemini_api_key" not in st.session_state:
    st.session_state["input_gemini_api_key"] = ""
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0


def save_api_key():
    key = (st.session_state.get("input_gemini_api_key") or "").strip()
    st.session_state["GEMINI_API_KEY"] = key or None


def clear_api_key():
    st.session_state["GEMINI_API_KEY"] = None
    st.session_state["input_gemini_api_key"] = ""


def get_gemini_api_key() -> Optional[str]:
    try:
        api_key = st.session_state.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    if api_key:
        return api_key

    api_key = ENV_API_KEY
    if api_key:
        return api_key

    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return None


def ensure_genai_installed() -> None:
    if genai is None:
        raise RuntimeError("缺少 google-generativeai 套件，請先安裝：pip install -U google-generativeai")


def list_available_models() -> List[str]:
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
        return ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash"]


def build_instructions() -> str:
    return (
        "你是一個嚴謹的表格解析器。請從提供的內容中找出「出勤」與「請假」兩類紀錄，"
        "並輸出 CSV。第一列必須輸出精確欄位標題：" + ", ".join(CSV_HEADERS) + "。\n\n"
        "【輸出格式要求】\n"
        "- 僅輸出 CSV 純文字。\n"
        "- 第一列必須是表頭：" + ", ".join(CSV_HEADERS) + "。\n"
        "- 每一列都必須嚴格對齊以上欄位數量（共 " + str(len(CSV_HEADERS)) + " 欄），即使某欄無資料也要留空，"
        "例如連續的逗號表示空欄位。\n"
        "- 不得多出或缺少欄位，否則會導致匯入系統失敗。\n"
        "- 每列資料都以英文逗號 ( , ) 分隔；禁止使用全形逗號、分號、tab 或其他符號。\n"
        "- 請確保每列的欄位順序與表頭完全一致。\n"
        "\n"
        "【一般規則】\n"
        "- 日期一律使用 YYYY-MM-DD（西元年），民國年需轉換：西元 = 民國 + 1911。\n"
        "- 時間一律使用 24 小時制 HH:MM（例如 09:00、18:30）。\n"
        "- 僅當來源明載『時數』或『天數』時，才分別填入『請假時數(小時)』或『請假天數(天)』；不得換算。\n"
        "- 同一份文件若包含多種假別，請各假別各輸出一列。\n"
        "- 不可臆測；無法確定的欄位留空（但仍保留逗號）。\n"
        "\n"
        "【分類規則】\n"
        "- 若段落明確顯示出勤關鍵字（上班/下班、打卡、刷卡、遲到、早退、加班、工號、班別等），→ 記錄類型=出勤。\n"
        "- 若段落明確顯示請假關鍵字（請假單/假別、申請人/代理人/主管/核准、起迄日期/時間、天數/時數等），→ 記錄類型=請假。\n"
        "- 若兩者皆不明確，請略過（不要誤判為請假）。\n"
        "- 若同人同日同時有出勤與請假，請分列輸出（出勤一列、請假若干列）。\n"
        "\n"
        "【出勤列】\n"
        "- 記錄類型=出勤。\n"
        "- 填『派駐單位』『姓名』『日期』『上班時間』『下班時間』；其他請假欄位留空。\n"
        "\n"
        "【請假列（不拆天）】\n"
        "- 記錄類型=請假。\n"
        "- 『請假起日』『請假迄日』：依文件原始日期範圍填寫（不拆成多天多列）。\n"
        "- 『日期』請填與『請假起日』相同。\n"
        "- 『假別』：標準化為：事假, 病假, 特休, 公假, 喪假, 婚假, 產假, 陪產假, 育嬰假, 家庭照顧假, 補休, 半薪病假, 其他。\n"
        "- 若文件提供『時間區間』，請填『請假時間(起)/(迄)』；\n"
        "  若文件提供『時數』，填『請假時數(小時)』；若提供『天數』，填『請假天數(天)』；"
        "  若同時提供兩種，皆可保留。\n"
        "- 跨午夜的區間（例如 22:00-02:00），保持一列：『請假起日』為起始日、『請假迄日』為次日。\n"
        "\n"
        "【備註欄位】\n"
        "- 僅放文件中明確標示的資訊（單據號、簽核註記、原文假別、補充說明、原始多段時段等）。\n"
    )


def fix_csv_column_count_and_shift(csv_text: str, headers: list[str]) -> str:
    expected = len(headers)
    out_rows = []
    reader = csv.reader(csv_text.splitlines())
    rows = list(reader)

    if rows and [h.strip() for h in rows[0]] == headers:
        start = 1
        out_rows.append(headers)
    else:
        start = 0
        out_rows.append(headers)

    idx = {name: i for i, name in enumerate(headers)}

    for r in rows[start:]:
        r2 = r[:] + [""] * max(0, expected - len(r))
        extras = r2[expected:] if len(r2) > expected else []
        r2 = r2[:expected]

        if r2 and r2[0].strip() == "出勤":
            pass
        elif r2 and r2[0].strip() == "請假":
            extra_texts = [t.strip() for t in extras if t.strip()]
            if extra_texts:
                r2[idx["備註"]] = (r2[idx["備註"]].strip() + "；" if r2[idx["備註"]].strip() else "") + "；".join(extra_texts)
            note = r2[idx["備註"]].strip()
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:天)?\s*", note)
            if m and not r2[idx["請假天數(天)"]].strip():
                r2[idx["請假天數(天)"]] = m.group(1)
                r2[idx["備註"]] = ""
            else:
                parts = [p.strip() for p in re.split(r"[；;]", note) if p.strip()]
                numeric_tokens = [p for p in parts if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*天)?", p)]
                text_tokens = [p for p in parts if p not in numeric_tokens]
                if numeric_tokens and not r2[idx["請假天數(天)"]].strip():
                    nm = re.search(r"\d+(?:\.\d+)?", numeric_tokens[0])
                    if nm:
                        r2[idx["請假天數(天)"]] = nm.group(0)
                    r2[idx["備註"]] = "；".join(text_tokens)

        if len(r2) < expected:
            r2 += [""] * (expected - len(r2))
        elif len(r2) > expected:
            r2 = r2[:expected]

        out_rows.append(r2)

    from io import StringIO
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in out_rows:
        writer.writerow(row)
    return buf.getvalue()


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
def _generate_image_csv_internal(image_bytes: bytes, model: str = "gemini-2.0-flash", source: str = "") -> str:
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


# ========== 檔案處理 ==========
def pdf_to_jpegs(pdf_bytes: bytes, dpi: int = 220, max_pages: int = 5) -> List[bytes]:
    if fitz is None:
        raise RuntimeError("缺少 PyMuPDF 套件，請先安裝：pip install PyMuPDF")
    images: List[bytes] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i in range(min(len(doc), max_pages)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            images.append(pix.tobytes("jpeg"))
    finally:
        doc.close()
    return images


def extract_pdf_text(file_like: io.BytesIO) -> str:
    if pdfplumber is None:
        raise RuntimeError("缺少 pdfplumber 套件，請先安裝依賴：pip install pdfplumber")
    text_pages: List[str] = []
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
            text_pages.append(extracted)
    return "\n\n".join(t for t in text_pages if t)


def ocr_image_to_text(image_bytes: bytes, lang: str = "eng+chi_tra") -> str:
    if pytesseract is None or Image is None:
        raise RuntimeError("缺少 OCR 依賴，請先安裝：pip install pytesseract pillow，並安裝 Tesseract 執行檔")
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return pytesseract.image_to_string(image, lang=lang)


# ========== CSV 處理 ==========
def normalize_csv_text(csv_text: str) -> str:
    if not csv_text:
        return csv_text
    s = csv_text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            s = "\n".join(lines[1:-1])
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    header = ",".join(CSV_HEADERS)
    first_line = s.split("\n", 1)[0].strip()
    if first_line.replace(" ", "") != header.replace(" ", ""):
        s = s.replace("，", ",").replace("、", ",").replace("\t", ",")
        if not s.startswith(header):
            s = header + "\n" + s
    return s


def csv_to_dataframe(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text))


DANGEROUS_PREFIXES = ("=", "+", "-", "@")


def sanitize_csv_for_excel(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        safe = []
        for c in cells:
            if c and c[0] in DANGEROUS_PREFIXES:
                safe.append("'" + c)
            else:
                safe.append(c)
        out_lines.append(",".join(safe))
    return "\n".join(out_lines)


def clear_uploads():
    st.session_state["uploader_key"] += 1


# helper to submit to global executor while bounding pending tasks
def submit_task(fn, *args, **kwargs):
    # Acquire slot for pending+running tasks
    GLOBAL_SUBMIT_SEMAPHORE.acquire()
    fut = GLOBAL_EXECUTOR.submit(fn, *args, **kwargs)

    # Ensure semaphore released when done
    def _release(_):
        try:
            GLOBAL_SUBMIT_SEMAPHORE.release()
        except Exception:
            pass

    fut.add_done_callback(_release)
    return fut


# ========== Streamlit UI ==========
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🗂️", layout="centered")
    st.title(APP_TITLE)
    st.caption("上傳出勤表（JPG 或 PDF），由模型解析並輸出 CSV。")

    # Sidebar: API key and concurrency settings (UI controls local behavior; heavy tuning via env recommended)
    with st.sidebar:
        st.subheader("設定")

        env_api = ENV_API_KEY
        session_api = st.session_state.get("GEMINI_API_KEY")

        if session_api:
            st.success("✓ 已偵測到暫存的 API Key（session）")
            st.caption("目前使用：暫存於本瀏覽器 session 的 API Key。")
            st.button("清除暫存的 API Key", on_click=clear_api_key)
        elif env_api:
            st.info("✓ 已偵測到系統環境的 API Key（採用環境變數）。")
            st.caption("建議生產環境使用環境變數提供 key。")
            override = st.checkbox("覆蓋環境變數的 API Key（使用自訂 key）", key="override_env_key")
            if override:
                st.text_input("輸入 Gemini API Key", type="password", key="input_gemini_api_key")
                st.button("儲存 API Key", on_click=save_api_key)
            else:
                st.caption("目前將使用系統環境變數提供的 API Key。")
        else:
            st.error("✗ 未偵測到 API Key")
            st.caption("請設定環境變數 GEMINI_API_KEY/GOOGLE_API_KEY；或暫存於下方輸入框（僅供本機測試）")
            st.text_input("輸入 Gemini API Key", type="password", key="input_gemini_api_key")
            st.button("儲存 API Key", on_click=save_api_key)

        model = st.selectbox(
            "Gemini 模型",
            options=["gemini-2.0-flash"],
            index=0,
            disabled=True,
            help="固定使用 Gemini 2.0 Flash 模型",
        )

        max_pages = st.number_input("PDF 解析頁數上限", min_value=1, max_value=30, value=5, step=1)
        # UI 仍允許微調單次操作的 max_workers/api_concurrency，但實際限制會以全域 env 為優先
        local_max_workers = st.number_input("背景工作執行緒數 (本次)", min_value=1, max_value=32, value=min(8, GLOBAL_MAX_WORKERS), step=1)
        local_api_concurrency = st.number_input("同時對 Gemini 的並發請求數 (本次 Semaphore)", min_value=1, max_value=16, value=min(4, GLOBAL_API_CONCURRENCY), step=1)

        st.divider()
        st.subheader("下載 logs（依月份）")

        logs_dir = os.path.join(os.getcwd(), "logs")
        available_months = set()
        scanned_files: list[tuple[str, str, str]] = []  # (fullpath, filename, day_str)

        if os.path.isdir(logs_dir):
            for fname in sorted(os.listdir(logs_dir)):
                if not fname.lower().endswith(".csv"):
                    continue
                full = os.path.join(logs_dir, fname)
                # 先嘗試從檔名擷取 YYYY-MM-DD
                m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", fname)
                if m:
                    day = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    month = f"{m.group(1)}-{m.group(2)}"
                else:
                    # fallback 使用檔案修改時間
                    ts = os.path.getmtime(full)
                    day = time.strftime("%Y-%m-%d", time.localtime(ts))
                    month = time.strftime("%Y-%m", time.localtime(ts))
                available_months.add(month)
                scanned_files.append((full, fname, day))
        else:
            st.info("未找到 `logs` 資料夾，請建立 `logs` 並放入 CSV 檔。")

        months_list = sorted(available_months, reverse=True)
        if months_list:
            sel_month = st.selectbox("選擇月份", options=months_list)
            if st.button("下載選定月份 Logs"):
                # 找出符合選定月份的檔案（以 day 開頭比對 YYYY-MM）
                matched = [(p, fn, day) for (p, fn, day) in scanned_files if day.startswith(sel_month)]
                if not matched:
                    st.warning(f"{sel_month} 無可匯出的 CSV 檔。")
                else:
                    buf = io.BytesIO()
                    try:
                        # 選擇 engine：優先 openpyxl，否則 xlsxwriter
                        engine = "openpyxl"
                        try:
                            import openpyxl  # type: ignore
                            engine = "openpyxl"
                        except Exception:
                            try:
                                import xlsxwriter  # type: ignore
                                engine = "xlsxwriter"
                            except Exception:
                                raise RuntimeError("缺少 openpyxl 或 xlsxwriter，請安裝：pip install openpyxl xlsxwriter")

                        # helper: sanitize sheet name for Excel
                        def _sanitize_sheet_name(name: str) -> str:
                            # 移除控制字元
                            name = "".join(ch for ch in name if ord(ch) >= 32)
                            # 替換不允許的字元 [: \ / ? * [ ]]
                            name = re.sub(r'[\:\?\\\/\*\[\]]', "_", name)
                            # Excel sheet 名稱上限 31 字元
                            name = name[:31] if name else "sheet"
                            return name

                        used_sheets = set()
                        # 以日期為 sheet 名；若同日多檔則加上序號；sheet name 最多 31 字元
                        with pd.ExcelWriter(buf, engine=engine, datetime_format="YYYY-MM-DD") as writer:
                            for path, filename, day in matched:
                                try:
                                    df = pd.read_csv(path)
                                except Exception:
                                    df = pd.DataFrame()

                                base = day or filename or "sheet"
                                sheet_name = _sanitize_sheet_name(base)
                                i = 1
                                original = sheet_name
                                while sheet_name in used_sheets:
                                    suffix = f"_{i}"
                                    # 保持總長度 <=31
                                    sheet_name = (original[: max(0, 31 - len(suffix))] + suffix)[:31]
                                    i += 1
                                # 最終再次保證合法
                                sheet_name = _sanitize_sheet_name(sheet_name)
                                try:
                                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                                except Exception:
                                    # 若寫入失敗，寫入空 sheet 以保留檔案結構
                                    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                                used_sheets.add(sheet_name)

                        buf.seek(0)
                        out_name = f"logs-{sel_month}.xlsx"
                        st.download_button(
                            label="下載 Excel（每日一個 sheet）",
                            data=buf.getvalue(),
                            file_name=out_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    except Exception as e:
                        st.error(f"建立匯出檔案失敗：{e}")
        else:
            st.info("`logs` 中尚無可用的月份資料。")

    uploader_key = st.session_state["uploader_key"]
    uploaded = st.file_uploader(
        "上傳檔案（可多選）",
        type=["jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{uploader_key}",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        parse_clicked = st.button("解析成 CSV", type="primary", use_container_width=True)
    with col2:
        st.button("全部清空", on_click=clear_uploads, use_container_width=True)

    if parse_clicked:
        if not uploaded or len(uploaded) == 0:
            st.warning("請先上傳檔案。")
            return

        files = uploaded if isinstance(uploaded, list) else [uploaded]

        # per-run choices but constrained by global resources
        api_key = get_gemini_api_key()
        if not api_key:
            st.error("缺少有效的 GEMINI API Key，請於側邊欄設定。")
            return

        # choose semaphore for this run: cannot exceed global concurrency
        run_api_concurrency = min(int(local_api_concurrency), GLOBAL_API_CONCURRENCY)
        run_api_sem = GLOBAL_API_SEMAPHORE if run_api_concurrency == GLOBAL_API_CONCURRENCY else threading.Semaphore(run_api_concurrency)

        try:
            all_rows = []
            header = ",".join(CSV_HEADERS)

            futures = []
            # submit tasks to the global executor using submit_task (bounded)
            for file_idx, uploaded_file in enumerate(files, start=1):
                st.divider()
                st.caption(f"📄 處理檔案 {file_idx}/{len(files)}: {uploaded_file.name}")

                if uploaded_file.type in ("image/jpeg", "image/jpg") or uploaded_file.name.lower().endswith((".jpg", ".jpeg")):
                    image_bytes = uploaded_file.read()
                    fut = submit_task(
                        call_gemini_for_image_csv_worker,
                        image_bytes,
                        model,
                        uploaded_file.name,
                        api_key,
                        GENAI_LOCK,
                        run_api_sem,
                    )
                    futures.append(fut)

                elif uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                    pdf_bytes = uploaded_file.read()

                    # define pdf processing function that calls the worker for each page
                    def _process_pdf(pdf_b: bytes, fname: str, model_name: str, max_p: int, key: str, lock: threading.Lock, sema: Optional[threading.Semaphore]):
                        parts = []
                        try:
                            imgs = pdf_to_jpegs(pdf_b, dpi=220, max_pages=max_p)
                        except Exception:
                            # 如果 PDF 轉圖失敗，回傳空字串以避免中斷
                            return ""
                        for idx, img in enumerate(imgs, start=1):
                            try:
                                # 這裡直接使用已在池中執行的呼叫（同步等待），避免再 spawn 太多任務
                                text = call_gemini_for_image_csv_worker(img, model_name, f"{fname}:p{idx}", key, lock, sema)
                                if text:
                                    parts.append(text)
                            except Exception:
                                # 單頁失敗不影響其他頁
                                continue
                        return "\n".join(p for p in parts if p)

                    fut = submit_task(
                        _process_pdf,
                        pdf_bytes,
                        uploaded_file.name,
                        model,
                        int(max_pages),
                        api_key,
                        GENAI_LOCK,
                        run_api_sem,
                    )
                    futures.append(fut)

                else:
                    st.warning(f"檔案 {uploaded_file.name} 格式不支援，已跳過。")
                    continue

            # show spinner while futures complete; collect results as they finish
            if futures:
                with st.spinner("模型解析中（背景執行）..."):
                    for fut in concurrent.futures.as_completed(futures):
                        try:
                            text = fut.result()
                        except Exception as e:
                            st.warning(f"背景工作失敗：{e}")
                            continue
                        if not text:
                            continue
                        try:
                            norm = normalize_csv_text(text)
                            lines = [l for l in norm.splitlines() if l.strip()]
                            all_rows.extend(lines)
                        except Exception:
                            # 若 normalize 失敗，仍把原文字加入以利排查
                            all_rows.append(text)

            # 合併同一份輸出 CSV 的表頭與內容
            merged_lines = []
            for i, line in enumerate(all_rows):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if line_stripped.replace(" ", "") == header.replace(" ", ""):
                    if not merged_lines:
                        merged_lines.append(header)
                    # 如果已存在表頭，忽略重複表頭
                else:
                    merged_lines.append(line_stripped)

            if not merged_lines or merged_lines[0].replace(" ", "") != header.replace(" ", ""):
                merged_lines.insert(0, header)

            csv_text = "\n".join(merged_lines)
            csv_text = fix_csv_column_count_and_shift(csv_text, headers=CSV_HEADERS)

            st.subheader("解析結果 (CSV)")
            st.code(csv_text or "(空白)", language="csv")

            df = None
            try:
                if csv_text:
                    df = csv_to_dataframe(csv_text)
            except Exception:
                st.warning("CSV 預覽失敗，但仍可下載原始文字。請檢查欄位與分隔符號（建議確保逗號分隔與首列表頭）。")

            if df is not None:
                st.dataframe(df, use_container_width=True)

            safe_csv = sanitize_csv_for_excel(csv_text or "")
            st.download_button(
                label="下載 CSV",
                data=safe_csv.encode("utf-8-sig"),
                file_name="attendance.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"解析失敗：{e}")


if __name__ == "__main__":
    main()
