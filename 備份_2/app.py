# python
import io
import os
import re
import time
from typing import Optional
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


# --- 新增：互動式影像檢視器（放大/縮小/拖動） ---
def render_image_viewer(image_bytes: bytes, caption: str = ""):
    """
    使用一段輕量的 HTML/JS 在 Streamlit 中嵌入可縮放/拖曳的影像檢視器。
    支援滑鼠滾輪縮放、按住拖曳平移。僅用於預覽（不改變原始 bytes）。
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    img_src = f"data:image/jpeg;base64,{b64}"
    html = f"""
    <div style="width:100%; height:70vh; border:1px solid rgba(0,0,0,0.08); position:relative; overflow:hidden; touch-action:none;">
      <div id="viewer" style="width:100%; height:100%; position:relative; background:#f6f6f6; display:flex; align-items:center; justify-content:center;">
        <img id="img" src="{img_src}" style="transform-origin:0 0; cursor:grab; position:absolute; left:0; top:0; will-change:transform; user-select:none; -webkit-user-drag:none;"/>
      </div>
      <div style="position:absolute; right:8px; top:8px; background:rgba(255,255,255,0.8); padding:4px 8px; border-radius:6px; font-size:12px;">
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

      // 初始化：將圖片放到左上並顯示原始大小
      img.onload = function() {{
        img.style.transform = `scale(${{scale}}) translate(${{originX}}px, ${{originY}}px)`;
      }};

      // 滾輪縮放（以游標位置為中心）
      viewer.onwheel = function(e) {{
        e.preventDefault();
        const rect = img.getBoundingClientRect();
        // 滑鼠在圖片內的相對座標
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const beforeScale = scale;
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        scale = Math.max(0.1, Math.min(10, scale * delta));
        // 計算以滑鼠為中心的位移補償（簡化版）
        originX -= (mx / beforeScale) - (mx / scale);
        originY -= (my / beforeScale) - (my / scale);
        img.style.transform = `scale(${{scale}}) translate(${{originX}}px, ${{originY}}px)`;
      }};

      // 指標事件處理（支援觸控筆與滑鼠）
      viewer.addEventListener('pointerdown', function(e) {{
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        viewer.setPointerCapture(e.pointerId);
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

      // 雙擊回復原始值
      viewer.addEventListener('dblclick', function(e) {{
        scale = 1;
        originX = 0;
        originY = 0;
        img.style.transform = `scale(${{scale}}) translate(${{originX}}px, ${{originY}}px)`;
      }});
    }})();
    </script>
    """
    # 高度使用 70vh 與右側 CSS 一致
    st.components.v1.html(html, height=600, scrolling=False)


# ========== Streamlit UI ==========
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🗂️", layout="wide")
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

        # 🔹 模型選擇：加入 gemini-2.0-flash-lite，預設用 lite 省成本
        model = st.selectbox(
            "Gemini 模型",
            options=["gemini-2.0-flash-lite", "gemini-2.0-flash"],
            index=0,
            help="建議大量解析時使用 gemini-2.0-flash-lite 以節省成本；若結果品質不足再改用 gemini-2.0-flash。",
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

            # 存储文件预览和建立映射关系
            file_previews = {}
            row_to_file_mapping = []
            current_row_index = 0

            futures = []
            file_futures_map = {}  # {future: (filename, file_bytes, file_type)}

            # submit tasks to the global executor using submit_task (bounded)
            for file_idx, uploaded_file in enumerate(files, start=1):
                st.divider()
                st.caption(f"📄 處理檔案 {file_idx}/{len(files)}: {uploaded_file.name}")

                if uploaded_file.type in ("image/jpeg", "image/jpg") or uploaded_file.name.lower().endswith((".jpg", ".jpeg")):
                    image_bytes = uploaded_file.read()
                    # 存储文件预览
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
                    # 存储文件预览
                    file_previews[uploaded_file.name] = pdf_bytes

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
                    file_futures_map[fut] = (uploaded_file.name, pdf_bytes, "pdf")
                    futures.append(fut)

                else:
                    st.warning(f"檔案 {uploaded_file.name} 格式不支援，已跳過。")
                    continue

            # show spinner while futures complete; collect results as they finish
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
                            # 建立行与文件的映射關係（跳過表頭）
                            for line in lines:
                                if line.strip().replace(" ", "") != header.replace(" ", ""):
                                    row_to_file_mapping.append((current_row_index, filename))
                                    current_row_index += 1
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

            # 存储到session_state
            # 注意：row_to_file_mapping的索引对应最终DataFrame的行索引（从0开始，不包括表头）
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

    # 显示解析结果（独立于parse_clicked，基于session_state）
    parsed_df = st.session_state.get("parsed_dataframe")
    file_previews = st.session_state.get("file_previews", {})
    row_to_file_mapping = st.session_state.get("row_to_file_mapping", [])

    if parsed_df is not None and len(parsed_df) > 0:
        st.subheader("解析結果")

        # 左右分栏：左侧表格，右侧文件预覽
        # 左側50%，右側50%
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown("**表格資料（可編輯）**")
            # 使用CSS确保表格高度与右侧預覽區匹配
            # 使用更具体的选择器来设置表格容器高度
            st.markdown(
                """
                <style>
                div[data-testid="stDataEditor"] > div {
                    height: 70vh !important;
                    min-height: 600px !important;
                }
                div[data-testid="stDataEditor"] {
                    height: 70vh !important;
                    min-height: 600px !important;
                }
                .stDataEditor {
                    height: 70vh !important;
                    min-height: 600px !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            # 使用可编辑的表格，从session_state获取最新数据
            # use_container_width=True 确保表格使用容器全宽，表格会自动支持横向和纵向滚动
            edited_df = st.data_editor(
                parsed_df,
                use_container_width=True,
                num_rows="fixed",
                key="data_editor"
            )
            # 更新session_state中的dataframe
            st.session_state["parsed_dataframe"] = edited_df.copy()

            # 将编辑后的DataFrame转换为CSV并存储
            csv_buffer = io.StringIO()
            edited_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_text_edited = csv_buffer.getvalue()
            safe_csv = sanitize_csv_for_excel(csv_text_edited)
            st.session_state["edited_csv"] = safe_csv.encode("utf-8-sig")

        with right_col:
            st.markdown("**檔案預覽**")
            # 顯示文件預覽
            if row_to_file_mapping and len(edited_df) > 0:
                # 獲取當前選中行對應的文件
                selected_row = st.selectbox(
                    "選擇要預覽的資料列",
                    options=list(range(len(edited_df))),
                    format_func=lambda x: f"第 {x} 列",
                    key="preview_row_selector"
                )

                if selected_row < len(row_to_file_mapping):
                    _, filename = row_to_file_mapping[selected_row]
                    if filename in file_previews:
                        file_bytes = file_previews[filename]
                        st.caption(f"**檔案：** {filename}")
                        if filename.lower().endswith((".jpg", ".jpeg")):
                            # 使用互動式檢視器顯示 JPG
                            try:
                                render_image_viewer(file_bytes, filename)
                            except Exception as e:
                                st.warning(f"無法顯示圖片預覽：{e}")
                        elif filename.lower().endswith(".pdf"):
                            # PDF預覽：顯示第一頁，並使用互動式檢視器
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
                    st.info("請選擇有效的資料列")
            else:
                st.info("無可預覽的檔案")

        # 下載按鈕：使用編輯後的資料
        st.divider()
        edited_csv_bytes = st.session_state.get("edited_csv", b"")
        st.download_button(
            label="下載 CSV",
            data=edited_csv_bytes,
            file_name="attendance.csv",
            mime="text/csv",
            use_container_width=False
        )


if __name__ == "__main__":
    main()
