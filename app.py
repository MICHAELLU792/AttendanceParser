# python
import io
import os
from typing import List, Optional

import pandas as pd
import streamlit as st

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
CSV_HEADERS = ["派駐單位", "姓名", "日期", "上班時間", "下班時間", "備註"]


# ========== Gemini 相關 ==========
# ---- 放在 main() 最前面（任何 UI 之前）----
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state["GEMINI_API_KEY"] = None
if "input_gemini_api_key" not in st.session_state:
    st.session_state["input_gemini_api_key"] = ""
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def save_api_key():
    key = (st.session_state.get("input_gemini_api_key") or "").strip()
    st.session_state["GEMINI_API_KEY"] = key or None
    # 不要在這裡 rerun，Streamlit 會自己重跑

def clear_api_key():
    st.session_state["GEMINI_API_KEY"] = None
    st.session_state["input_gemini_api_key"] = ""
    # 這裡也不要 rerun

def get_gemini_api_key() -> Optional[str]:
    """取得 Gemini API Key，優先順序：session_state > 環境變數 > Streamlit secrets（環境變數更安全）"""
    try:
        api_key = st.session_state.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    if api_key:
        return api_key

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key

    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return None


def ensure_gemini_ready() -> None:
    """檢查 SDK 與 API Key 並初始化"""
    if genai is None:
        raise RuntimeError("缺少 google-generativeai 套件，請先安裝：pip install -U google-generativeai")
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("請設定環境變數 GEMINI_API_KEY 或 GOOGLE_API_KEY（或於側邊欄暫存於 session）")
    genai.configure(api_key=api_key)


def list_available_models() -> List[str]:
    """列出可用的 Gemini 模型"""
    ensure_gemini_ready()
    try:
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
        "你是一個嚴謹的表格解析器。請從提供的內容中找出出勤紀錄，"
        "並轉成 CSV。第一列請輸出精確欄位標題：" + ", ".join(CSV_HEADERS) + ".\n\n"
        "規則：\n"
        "- 僅輸出 CSV 純文字，勿包含解釋或額外內容。\n"
        "- 日期格式統一使用 YYYY-MM-DD（西元年）。\n"
        "- 年份轉換：若來源為民國年（例如「民國114年」或「114年」），必須轉換為西元年。\n"
        "  轉換公式：西元年 = 民國年 + 1911\n"
        "  例如：民國114年 → 2025年，民國113年 → 2024年，民國112年 → 2023年。\n"
        "- 上班時間與下班時間使用 24 小時制（例如 09:00、18:30）。\n"
        "- 備註欄位處理：\n"
        "  * 僅根據檔案中明確標示的資訊填入備註，不要自行推測或臆測。\n"
        "  * 若檔案中有明確標示特殊情況，請在備註中說明。\n"
        "  * 若檔案中沒有明確標示任何特殊情況，備註欄位請留空。\n"
        "- 若缺漏資料列，請僅輸出能確定的列。\n"
    )


def call_gemini_for_image_csv(image_bytes: bytes, model: str = "gemini-2.0-flash") -> str:
    ensure_gemini_ready()
    instructions = build_instructions()
    img_part = {"mime_type": "image/jpeg", "data": image_bytes}
    try:
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(
            [instructions, img_part],
            generation_config=genai.GenerationConfig(temperature=0),
        )
        return (resp.text or "").strip()
    except Exception as e:
        available_models = list_available_models()
        raise RuntimeError(
            f"模型 '{model}' 不可用。\n可用模型：{', '.join(available_models[:5])}\n原始錯誤：{str(e)}"
        )


def call_gemini_for_text_csv(text: str, model: str = "gemini-pro") -> str:
    ensure_gemini_ready()
    instructions = build_instructions()
    prompt = instructions + "以下為從 PDF 或 OCR 取得的純文字內容（可能包含表格展平）：\n\n" + text[:200000]
    try:
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0),
        )
        return (resp.text or "").strip()
    except Exception as e:
        available_models = list_available_models()
        raise RuntimeError(
            f"模型 '{model}' 不可用。\n可用模型：{', '.join(available_models[:5])}\n原始錯誤：{str(e)}"
        )


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

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def clear_uploads():
    # 只要變更 key，file_uploader 就會被視為新元件而重置
    st.session_state["uploader_key"] += 1

# ========== Streamlit UI ==========
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🗂️", layout="centered")
    st.title(APP_TITLE)
    st.caption("上傳出勤表（JPG 或 PDF），由模型解析並輸出 CSV。")

    # ---- Sidebar（替換你原本的儲存/清除區塊）----
    with st.sidebar:
        st.subheader("設定")

        api_key = get_gemini_api_key()
        if api_key:
            st.success("✓ 已偵測到 API Key")
            st.caption("已載入 API Key。")
            st.button("清除暫存的 API Key", on_click=clear_api_key)
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

        try:
            all_rows = []
            header = ",".join(CSV_HEADERS)

            for file_idx, uploaded_file in enumerate(files, start=1):
                st.divider()
                st.caption(f"📄 處理檔案 {file_idx}/{len(files)}: {uploaded_file.name}")

                if uploaded_file.type in ("image/jpeg", "image/jpg") or uploaded_file.name.lower().endswith((".jpg", ".jpeg")):
                    image_bytes = uploaded_file.read()
                    with st.spinner(f"模型解析影像中..."):
                        csv_text = call_gemini_for_image_csv(image_bytes, model=model)
                    csv_text = normalize_csv_text(csv_text)
                    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
                    all_rows.extend(lines)

                elif uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                    pdf_bytes = uploaded_file.read()
                    with st.spinner("將 PDF 逐頁轉為影像中..."):
                        page_images = pdf_to_jpegs(pdf_bytes, dpi=220, max_pages=max_pages)
                    for page_idx, img in enumerate(page_images, start=1):
                        st.caption(f"第 {page_idx} 頁解析中…")
                        with st.spinner(f"模型解析第 {page_idx} 頁影像中..."):
                            part_csv = call_gemini_for_image_csv(img, model=model)
                        part_csv = normalize_csv_text(part_csv)
                        lines = [ln for ln in part_csv.splitlines() if ln.strip()]
                        all_rows.extend(lines)

                else:
                    st.warning(f"檔案 {uploaded_file.name} 格式不支援，已跳過。")
                    continue

            merged_lines = []
            for i, line in enumerate(all_rows):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if line_stripped.replace(" ", "") == header.replace(" ", ""):
                    if not merged_lines:
                        merged_lines.append(line_stripped)
                else:
                    merged_lines.append(line_stripped)

            if not merged_lines or merged_lines[0].replace(" ", "") != header.replace(" ", ""):
                merged_lines.insert(0, header)

            csv_text = "\n".join(merged_lines)

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