# python
import io
import os
import re
import time
from typing import Optional, Any
import threading
import concurrent.futures
import base64

import pandas as pd
import streamlit as st

# PDF rendering (PyMuPDF) - needed for PDF preview
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

# Import modules
from app_config import (
    APP_TITLE,
    CSV_HEADERS,
    GLOBAL_MAX_WORKERS,
    GLOBAL_API_CONCURRENCY,
    GLOBAL_SUBMIT_SEMAPHORE,
    GLOBAL_EXECUTOR,
    GLOBAL_API_SEMAPHORE,
    GENAI_LOCK,
    ENV_API_KEY,
)
from gemini_client import (
    call_gemini_for_image_csv_worker,
    call_gemini_for_text_csv_worker,
    list_available_models,
)
from file_processing import pdf_to_jpegs
from csv_processing import (
    fix_csv_column_count_and_shift,
    normalize_csv_text,
    csv_to_dataframe,
    sanitize_csv_for_excel,
)

# Configure Gemini API if ENV_API_KEY is available
if ENV_API_KEY:
    try:
        import google.generativeai as genai
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
if "parsed_dataframe" not in st.session_state:
    st.session_state["parsed_dataframe"] = None
if "file_previews" not in st.session_state:
    st.session_state["file_previews"] = {}  # {filename: file_bytes}
if "row_to_file_mapping" not in st.session_state:
    st.session_state["row_to_file_mapping"] = []  # List of (row_index, filename) tuples


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


def clear_uploads():
    st.session_state["uploader_key"] += 1


# Helper to submit to global executor while bounding pending tasks
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


# --- 互動式影像檢視器（放大/縮小/拖動） ---
def render_image_viewer(image_bytes: bytes, caption: str = ""):
    """
    在 Streamlit 中嵌入可縮放/拖曳的影像檢視器，視覺高度以 80vh 為基準。
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    img_src = f"data:image/jpeg;base64,{b64}"
    # 調整為 80vh 視覺高度
    html = f"""
    <div style="width:100%; height:80vh; border:1px solid rgba(0,0,0,0.08); position:relative; overflow:hidden; touch-action:none;">
      <div id="viewer" style="width:100%; height:100%; position:relative; background:#f6f6f6; display:flex; align-items:center; justify-content:center;">
        <img id="img" src="{img_src}" style="transform-origin:0 0; cursor:grab; position:absolute; left:0; top:0; will-change:transform; user-select:none; -webkit-user-drag:none;"/>
      </div>
      <div style="position:absolute; right:8px; top:8px; background:rgba(255,255,255,0.85); padding:4px 8px; border-radius:6px; font-size:12px;">
        {caption}
      </div>
    </div>
    <script>
    (function(){{
      const viewer = document.getElementById('viewer');
      const img = document.getElementById('img');
      let scale = 1;
      let originX = 0;
      let originY = 0;
      let dragging = false;
      let lastX = 0, lastY = 0;

      img.onload = function() {{
        img.style.transform = `scale(${{scale}}) translate(${{originX}}px, ${{originY}}px)`;
      }};

      viewer.onwheel = function(e) {{
        e.preventDefault();
        const rect = img.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const beforeScale = scale;
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        scale = Math.max(0.1, Math.min(10, scale * delta));
        originX -= (mx / beforeScale) - (mx / scale);
        originY -= (my / beforeScale) - (my / scale);
        img.style.transform = `scale(${{scale}}) translate(${{originX}}px, ${{originY}}px)`;
      }};

      viewer.addEventListener('pointerdown', function(e) {{
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        try {{ viewer.setPointerCapture(e.pointerId); }} catch(e){{}}
        img.style.cursor = 'grabbing';
      }});

      viewer.addEventListener('pointermove', function(e) {{
        if(!dragging) return;
        const dx = (e.clientX - lastX) / scale;
        const dy = (e.clientY - lastY) / scale;
        originX += dx;
        originY += dy;
        lastX = e.clientX;
        lastY = e.clientY;
        img.style.transform = `scale(${{scale}}) translate(${{originX}}px, ${{originY}}px)`;
      }});

      function endDrag(e) {{
        dragging = false;
        img.style.cursor = 'grab';
      }}
      viewer.addEventListener('pointerup', endDrag);
      viewer.addEventListener('pointercancel', endDrag);
      viewer.addEventListener('pointerleave', endDrag);

      viewer.addEventListener('dblclick', function(e) {{
        scale = 1;
        originX = 0;
        originY = 0;
        img.style.transform = `scale(${{scale}}) translate(${{originX}}px, ${{originY}}px)`;
      }});
    }})();
    </script>
    """
    # 指定較大整數高度以匹配 80vh 視覺效果
    st.components.v1.html(html, height=820, scrolling=False)


# Helper: 根據不同儲存格式回傳指定 row 的檔案名稱
def filename_for_row(selected_row: int, mapping: Any) -> Optional[str]:
    """
    支援以下格式的 mapping：
    - list of (row_idx, filename) tuples
    - dict {row_idx: filename}
    - list indexed by row (e.g. [None, 'a.jpg', ...]) 或 list of filenames
    若找不到對應檔案則回傳 None。
    """
    try:
        if mapping is None:
            return None
        if isinstance(mapping, dict):
            return mapping.get(selected_row)
        if isinstance(mapping, list):
            # list of tuples
            if len(mapping) > 0 and isinstance(mapping[0], (list, tuple)) and len(mapping[0]) >= 2:
                for tup in mapping:
                    try:
                        if tup[0] == selected_row:
                            return tup[1]
                    except Exception:
                        continue
            # list indexed by row or list of filenames
            if 0 <= selected_row < len(mapping):
                entry = mapping[selected_row]
                if isinstance(entry, str):
                    return entry
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    return entry[1]
    except Exception:
        pass
    return None


# ========== Streamlit UI ==========
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🗂️", layout="wide")
    st.title(APP_TITLE)
    st.caption("上傳出勤表（JPG 或 PDF），由模型解析並輸出 CSV。")

    # Sidebar: API key and concurrency settings
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

        # 模型選擇
        model = st.selectbox(
            "Gemini 模型",
            options=["gemini-2.0-flash-lite", "gemini-2.0-flash"],
            index=0,
            help="建議大量解析時使用 gemini-2.0-flash-lite 以節省成本；若結果品質不足再改用 gemini-2.0-flash。",
        )

        max_pages = st.number_input("PDF 解析頁數上限", min_value=1, max_value=30, value=5, step=1)
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
                m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", fname)
                if m:
                    day = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    month = f"{m.group(1)}-{m.group(2)}"
                else:
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
                matched = [(p, fn, day) for (p, fn, day) in scanned_files if day.startswith(sel_month)]
                if not matched:
                    st.warning(f"{sel_month} 無可匯出的 CSV 檔。")
                else:
                    buf = io.BytesIO()
                    try:
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

                        def _sanitize_sheet_name(name: str) -> str:
                            name = "".join(ch for ch in name if ord(ch) >= 32)
                            name = re.sub(r'[\:\?\\\/\*\[\]]', "_", name)
                            name = name[:31] if name else "sheet"
                            return name

                        used_sheets = set()
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
                                    sheet_name = (original[: max(0, 31 - len(suffix))] + suffix)[:31]
                                    i += 1
                                sheet_name = _sanitize_sheet_name(sheet_name)
                                try:
                                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                                except Exception:
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

    # 只變更「解析成 CSV」按鈕顏色（以 aria-label 精準選擇）
    st.markdown(
        """
        <style>
        button[aria-label="解析成 CSV"] {
            background-color: #cfefff !important;
            color: #000 !important;
            border: 1px solid #bce0ff !important;
            box-shadow: none !important;
        }
        /* 使左右兩側容器內的按鈕樣式不被誤改（保守） */
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        parse_clicked = st.button("解析成 CSV", type="primary", width="stretch")
    with col2:
        st.button("全部清空", on_click=clear_uploads, width="stretch")

    if parse_clicked:
        if not uploaded or len(uploaded) == 0:
            st.warning("請先上傳檔案。")
            return

        files = uploaded if isinstance(uploaded, list) else [uploaded]

        api_key = get_gemini_api_key()
        if not api_key:
            st.error("缺少有效的 GEMINI API Key，請於側邊欄設定。")
            return

        run_api_concurrency = min(int(local_api_concurrency), GLOBAL_API_CONCURRENCY)
        run_api_sem = GLOBAL_API_SEMAPHORE if run_api_concurrency == GLOBAL_API_CONCURRENCY else threading.Semaphore(run_api_concurrency)

        try:
            all_rows = []
            header = ",".join(CSV_HEADERS)

            file_previews = {}
            row_to_file_mapping = []
            current_row_index = 0

            futures = []
            file_futures_map = {}  # {future: (filename, file_bytes, file_type)}

            for file_idx, uploaded_file in enumerate(files, start=1):
                st.divider()
                st.caption(f"📄 處理檔案 {file_idx}/{len(files)}: {uploaded_file.name}")

                if uploaded_file.type in ("image/jpeg", "image/jpg") or uploaded_file.name.lower().endswith((".jpg", ".jpeg")):
                    image_bytes = uploaded_file.read()
                    file_previews[uploaded_file.name] = image_bytes
                    fut = submit_task(
                        call_gemini_for_image_csv_worker,
                        image_bytes,
                        model,
                        uploaded_file.name,
                        api_key,
                        GENAI_LOCK,
                        run_api_sem,
                    )
                    file_futures_map[fut] = (uploaded_file.name, image_bytes, "image")
                    futures.append(fut)

                elif uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                    pdf_bytes = uploaded_file.read()
                    file_previews[uploaded_file.name] = pdf_bytes

                    def _process_pdf(pdf_b: bytes, fname: str, model_name: str, max_p: int, key: str, lock: threading.Lock, sema: Optional[threading.Semaphore]):
                        parts = []
                        try:
                            imgs = pdf_to_jpegs(pdf_b, dpi=220, max_pages=max_p)
                        except Exception:
                            return ""
                        for idx, img in enumerate(imgs, start=1):
                            try:
                                text = call_gemini_for_image_csv_worker(img, model_name, f"{fname}:p{idx}", key, lock, sema)
                                if text:
                                    parts.append(text)
                            except Exception:
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
                    file_futures_map[fut] = (uploaded_file.name, pdf_bytes, "pdf")
                    futures.append(fut)

                else:
                    st.warning(f"檔案 {uploaded_file.name} 格式不支援，已跳過。")
                    continue

            if futures:
                with st.spinner("模型解析中（背景執行）..."):
                    for fut in concurrent.futures.as_completed(futures):
                        filename, file_bytes, file_type = file_futures_map.get(fut, ("unknown", None, "unknown"))
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
                            for line in lines:
                                if line.strip().replace(" ", "") != header.replace(" ", ""):
                                    row_to_file_mapping.append((current_row_index, filename))
                                    current_row_index += 1
                            all_rows.extend(lines)
                        except Exception:
                            all_rows.append(text)

            merged_lines = []
            for i, line in enumerate(all_rows):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if line_stripped.replace(" ", "") == header.replace(" ", ""):
                    if not merged_lines:
                        merged_lines.append(header)
                else:
                    merged_lines.append(line_stripped)

            if not merged_lines or merged_lines[0].replace(" ", "") != header.replace(" ", ""):
                merged_lines.insert(0, header)

            csv_text = "\n".join(merged_lines)
            csv_text = fix_csv_column_count_and_shift(csv_text, headers=CSV_HEADERS)

            st.session_state["file_previews"] = file_previews
            st.session_state["row_to_file_mapping"] = row_to_file_mapping

            df = None
            try:
                if csv_text:
                    df = csv_to_dataframe(csv_text)
                    st.session_state["parsed_dataframe"] = df.copy()
            except Exception:
                st.warning("CSV 預覽失敗，但仍可下載原始文字。請檢查欄位與分隔符號（建議確保逗號分隔與首列表頭）。")
                st.session_state["parsed_dataframe"] = None

        except Exception as e:
            st.error(f"解析失敗：{e}")

    # 顯示解析結果（基於 session_state）
    parsed_df = st.session_state.get("parsed_dataframe")
    file_previews = st.session_state.get("file_previews", {})
    row_to_file_mapping = st.session_state.get("row_to_file_mapping", [])

    if parsed_df is not None and len(parsed_df) > 0:
        st.subheader("解析結果")

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown("**表格資料（可編輯）**")
            # 調高左側 DataEditor 高度為 80vh，接近下方水平線
            st.markdown(
                """
                <style>
                div[data-testid="stDataEditor"] > div {
                    height: 80vh !important;
                    min-height: 720px !important;
                }
                div[data-testid="stDataEditor"] {
                    height: 80vh !important;
                    min-height: 720px !important;
                }
                .stDataEditor {
                    height: 80vh !important;
                    min-height: 720px !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            edited_df = st.data_editor(
                parsed_df,
                width="stretch",
                num_rows="fixed",
                key="data_editor"
            )
            st.session_state["parsed_dataframe"] = edited_df.copy()

            csv_buffer = io.StringIO()
            edited_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_text_edited = csv_buffer.getvalue()
            safe_csv = sanitize_csv_for_excel(csv_text_edited)
            st.session_state["edited_csv"] = safe_csv.encode("utf-8-sig")

        with right_col:
            st.markdown("**檔案預覽**")
            # 顯示文件預覽，預覽區高度已由 render_image_viewer 設定為 80vh
            if row_to_file_mapping and len(edited_df) > 0:
                selected_row = st.selectbox(
                    "選擇要預覽的資料列",
                    options=list(range(len(edited_df))),
                    format_func=lambda x: f"第 {x} 列",
                    key="preview_row_selector"
                )

                # 使用通用 helper 取得對應檔名
                filename = filename_for_row(selected_row, row_to_file_mapping)

                if filename:
                    if filename in file_previews:
                        file_bytes = file_previews[filename]
                        st.caption(f"**檔案：** {filename}")
                        if filename.lower().endswith((".jpg", ".jpeg")):
                            try:
                                render_image_viewer(file_bytes, filename)
                            except Exception as e:
                                st.warning(f"無法顯示圖片預覽：{e}")
                        elif filename.lower().endswith(".pdf"):
                            try:
                                if fitz is not None:
                                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                                    if len(doc) > 0:
                                        page = doc.load_page(0)
                                        pix = page.get_pixmap(dpi=150, alpha=False)
                                        img_bytes = pix.tobytes("jpeg")
                                        render_image_viewer(img_bytes, f"{filename} (第1頁)")
                                    doc.close()
                                else:
                                    st.info(f"PDF 檔案：{filename}\n（需要 PyMuPDF 套件以顯示預覽）")
                            except Exception as e:
                                st.warning(f"無法預覽 PDF：{e}")
                        else:
                            st.info(f"檔案：{filename}")
                    else:
                        st.info("找不到對應的檔案")
                else:
                    st.info("此列無對應的檔案可預覽")
            else:
                st.info("無可預覽的檔案")

        st.divider()
        edited_csv_bytes = st.session_state.get("edited_csv", b"")
        st.download_button(
            label="下載 CSV",
            data=edited_csv_bytes,
            file_name="attendance.csv",
            mime="text/csv",
            width="content"
        )


if __name__ == "__main__":
    main()
