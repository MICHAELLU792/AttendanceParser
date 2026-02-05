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
# 新增：編輯版 DataFrame 快取與對應下載 bytes
if "edited_dataframe" not in st.session_state:
    st.session_state["edited_dataframe"] = None
if "edited_csv" not in st.session_state:
    st.session_state["edited_csv"] = b""
# 按文件分组的数据
if "data_by_file" not in st.session_state:
    st.session_state["data_by_file"] = {}  # {filename: DataFrame}
if "edited_data_by_file" not in st.session_state:
    st.session_state["edited_data_by_file"] = {}  # {filename: DataFrame}

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

def calculate_hours(start_time: str, end_time: str) -> str:
    """计算两个时间之间的总时数（小时），格式：HH:MM -> 小时数（浮点数）
    如果时间格式不正确（比如字段错位导致的值），返回空字符串"""
    if not start_time or not end_time or pd.isna(start_time) or pd.isna(end_time):
        return ""
    
    try:
        start_time = str(start_time).strip()
        end_time = str(end_time).strip()
        if not start_time or not end_time:
            return ""
        
        # 验证时间格式，防止字段错位导致错误计算
        if not is_valid_time_format(start_time) or not is_valid_time_format(end_time):
            return ""
        
        # 解析时间格式 HH:MM
        def parse_time(t: str) -> Optional[float]:
            parts = t.split(":")
            if len(parts) == 2:
                try:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    return hours + minutes / 60.0
                except ValueError:
                    return None
            return None
        
        start = parse_time(start_time)
        end = parse_time(end_time)
        
        if start is None or end is None:
            return ""
        
        # 处理跨日情况（如果结束时间小于开始时间，假设是第二天）
        if end < start:
            end += 24
        
        hours = end - start
        # 格式化为保留2位小数的字符串
        return f"{hours:.2f}"
    except Exception:
        return ""

def is_valid_time_format(time_str: str) -> bool:
    """验证时间字符串是否符合HH:MM格式"""
    if not time_str or pd.isna(time_str):
        return False
    try:
        time_str = str(time_str).strip()
        if not time_str:
            return False
        parts = time_str.split(":")
        if len(parts) != 2:
            return False
        hours = int(parts[0])
        minutes = int(parts[1])
        # 验证小时和分钟的范围
        if hours < 0 or hours >= 24 or minutes < 0 or minutes >= 60:
            return False
        return True
    except (ValueError, AttributeError):
        return False

def calculate_minutes(start_time: str, end_time: str) -> str:
    """计算两个时间之间的总分钟数，格式：HH:MM -> 分钟数（整数，纯数字字符串，如"56"，不是"00:56"这种时间格式）
    如果时间格式不正确（比如字段错位导致的值），返回空字符串"""
    if not start_time or not end_time or pd.isna(start_time) or pd.isna(end_time):
        return ""
    
    try:
        start_time = str(start_time).strip()
        end_time = str(end_time).strip()
        if not start_time or not end_time:
            return ""
        
        # 验证时间格式，防止字段错位导致错误计算
        if not is_valid_time_format(start_time) or not is_valid_time_format(end_time):
            return ""
        
        # 解析时间格式 HH:MM
        def parse_time_to_minutes(t: str) -> Optional[int]:
            parts = t.split(":")
            if len(parts) == 2:
                try:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    return hours * 60 + minutes
                except ValueError:
                    return None
            return None
        
        start_minutes = parse_time_to_minutes(start_time)
        end_minutes = parse_time_to_minutes(end_time)
        
        if start_minutes is None or end_minutes is None:
            return ""
        
        # 处理跨日情况（如果结束时间小于开始时间，假设是第二天）
        if end_minutes < start_minutes:
            end_minutes += 24 * 60
        
        total_minutes = end_minutes - start_minutes
        return str(total_minutes)
    except Exception:
        return ""

def normalize_leave_type(leave_type: str) -> str:
    """标准化假别，确保假别精准"""
    if not leave_type or pd.isna(leave_type):
        return ""
    
    leave_type = str(leave_type).strip()
    if not leave_type:
        return ""
    
    # 标准假别列表（根据gemini_client.py中的定义）
    valid_leave_types = [
        "事假", "病假", "特休", "公假", "喪假", "婚假", 
        "產假", "陪產假", "育嬰假", "家庭照顧假", "補休", 
        "半薪病假", "其他"
    ]
    
    # 精确匹配
    if leave_type in valid_leave_types:
        return leave_type
    
    # 模糊匹配（处理常见变体）
    leave_type_lower = leave_type.lower()
    type_mapping = {
        "事": "事假",
        "病": "病假",
        "特": "特休",
        "公": "公假",
        "喪": "喪假",
        "婚": "婚假",
        "產": "產假",
        "陪產": "陪產假",
        "育嬰": "育嬰假",
        "家庭照顧": "家庭照顧假",
        "補": "補休",
        "半薪": "半薪病假",
    }
    
    for key, value in type_mapping.items():
        if key in leave_type_lower or leave_type_lower in key:
            return value
    
    # 如果无法匹配，返回原值（可能是"其他"或特殊假别）
    return leave_type

def generate_monthly_dates(year: int, month: int) -> list[str]:
    """生成指定年月的所有日期列表（YYYY-MM-DD格式）"""
    import calendar
    days_in_month = calendar.monthrange(year, month)[1]
    dates = []
    for day in range(1, days_in_month + 1):
        dates.append(f"{year}-{month:02d}-{day:02d}")
    return dates

def expand_dataframe_to_monthly(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """将DataFrame扩展为包含该月所有日期的完整列表，并处理跨月数据"""
    from app_config import CSV_HEADERS
    from datetime import datetime
    
    # 确定主要月份：优先从出勤记录的日期推断，如果没有出勤记录，则从第一个记录的日期推断
    # 支持当年和去年的日期识别
    main_year, main_month = None, None
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    # 收集所有有效日期
    all_dates = []
    if not df.empty and "日期" in df.columns:
        for _, row in df.iterrows():
            date_val = row.get("日期", "")
            if pd.notna(date_val) and str(date_val).strip():
                try:
                    date_obj = pd.to_datetime(str(date_val).strip())
                    all_dates.append((date_obj.year, date_obj.month, date_obj.day))
                except Exception:
                    continue
    
    # 确定主要年份和月份
    if all_dates:
        # 统计所有年份的出现频率（不限制范围，从数据中推断）
        year_counts = {}
        for year, month, day in all_dates:
            # 只考虑合理的年份范围（2000-2100），避免异常数据
            if 2000 <= year <= 2100:
                year_counts[year] = year_counts.get(year, 0) + 1
        
        # 选择出现频率最高的年份（优先当年，其次是去年，最后是其他年份）
        if year_counts:
            # 优先考虑当年和去年
            if current_year in year_counts:
                selected_year = current_year
            elif current_year - 1 in year_counts:
                selected_year = current_year - 1
            else:
                # 如果当年和去年都不在数据中，选择出现频率最高的年份
                selected_year = max(year_counts.keys(), key=lambda y: year_counts[y])
            
            # 找出该年份中最常见的月份
            month_counts = {}
            for year, month, day in all_dates:
                if year == selected_year:
                    month_counts[month] = month_counts.get(month, 0) + 1
            
            if month_counts:
                main_year = selected_year
                main_month = max(month_counts.keys(), key=lambda m: month_counts[m])
    
    # 如果没有找到有效日期，优先从出勤记录推断
    if main_year is None and not df.empty and "日期" in df.columns and "記錄類型" in df.columns:
        # 优先找出勤记录的日期（接受合理范围内的年份：2000-2100）
        for _, row in df.iterrows():
            record_type_val = row.get("記錄類型", "")
            record_type = str(record_type_val).strip() if pd.notna(record_type_val) else ""
            if record_type == "出勤":
                date_val = row.get("日期", "")
                if pd.notna(date_val) and str(date_val).strip():
                    try:
                        date_obj = pd.to_datetime(str(date_val).strip())
                        # 接受合理范围内的年份（2000-2100）
                        if 2000 <= date_obj.year <= 2100:
                            main_year, main_month = date_obj.year, date_obj.month
                            break
                    except Exception:
                        continue
        
        # 如果没有出勤记录，找第一个有效日期（合理范围内的年份）
        if main_year is None:
            for _, row in df.iterrows():
                date_val = row.get("日期", "")
                if pd.notna(date_val) and str(date_val).strip():
                    try:
                        date_obj = pd.to_datetime(str(date_val).strip())
                        # 接受合理范围内的年份（2000-2100）
                        if 2000 <= date_obj.year <= 2100:
                            main_year, main_month = date_obj.year, date_obj.month
                            break
                    except Exception:
                        continue
    
    # 如果还是无法确定，使用当前年月
    if main_year is None:
        main_year, main_month = current_year, current_month
    
    # 生成主要月份的所有日期
    main_dates = set(generate_monthly_dates(main_year, main_month))
    
    # 创建完整的日期DataFrame
    full_df = pd.DataFrame(columns=CSV_HEADERS)
    
    # 将现有数据分类：主要月份内的数据和主要月份外的数据
    data_in_main_month = {}  # {date_str: [row_dict, ...]}
    data_outside_main_month = []  # [row_dict, ...]
    
    if not df.empty and "日期" in df.columns:
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            date_val = row.get("日期", "")
            
            if pd.notna(date_val) and str(date_val).strip():
                try:
                    date_obj = pd.to_datetime(str(date_val).strip())
                    date_str = date_obj.strftime("%Y-%m-%d")
                    
                    # 判断是否在主要月份内
                    if date_str in main_dates:
                        if date_str not in data_in_main_month:
                            data_in_main_month[date_str] = []
                        data_in_main_month[date_str].append(row_dict)
                    else:
                        # 不在主要月份内，append到外部数据列表
                        data_outside_main_month.append(row_dict)
                except Exception:
                    # 日期解析失败，也加入到外部数据列表
                    data_outside_main_month.append(row_dict)
            else:
                # 没有日期，也加入到外部数据列表（可能是跨月请假/加班的记录）
                data_outside_main_month.append(row_dict)
    
    # 为主要月份的每个日期创建行
    main_dates_sorted = sorted(main_dates)
    for date_str in main_dates_sorted:
        if date_str in data_in_main_month:
            # 如果该日期有数据，使用现有数据
            for row_dict in data_in_main_month[date_str]:
                new_row = {col: row_dict.get(col, "") for col in CSV_HEADERS}
                new_row["日期"] = date_str
                # 标准化假别
                leave_type_val = new_row.get("假別", "")
                if pd.notna(leave_type_val) and str(leave_type_val).strip():
                    new_row["假別"] = normalize_leave_type(str(leave_type_val).strip())
                # 计算总时数（只要有上班時間和下班時間就计算，不限于出勤记录）
                start_time_val = new_row.get("上班時間", "")
                end_time_val = new_row.get("下班時間", "")
                start_time = str(start_time_val).strip() if pd.notna(start_time_val) and str(start_time_val).strip() else ""
                end_time = str(end_time_val).strip() if pd.notna(end_time_val) and str(end_time_val).strip() else ""
                if start_time and end_time:
                    new_row["總時數"] = calculate_hours(start_time, end_time)
                
                # 计算加班時數(分鐘)（只要有加班時間(起)和加班時間(迄)就计算）
                overtime_start_val = new_row.get("加班時間(起)", "")
                overtime_end_val = new_row.get("加班時間(迄)", "")
                overtime_start = str(overtime_start_val).strip() if pd.notna(overtime_start_val) and str(overtime_start_val).strip() else ""
                overtime_end = str(overtime_end_val).strip() if pd.notna(overtime_end_val) and str(overtime_end_val).strip() else ""
                if overtime_start and overtime_end:
                    minutes = calculate_minutes(overtime_start, overtime_end)
                    if minutes:
                        new_row["加班時數(分鐘)"] = minutes
                full_df = pd.concat([full_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # 如果该日期没有数据，创建空行（只填日期）
            new_row = {col: "" for col in CSV_HEADERS}
            new_row["日期"] = date_str
            full_df = pd.concat([full_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # 将主要月份外的数据 append 在最后面
    for row_dict in data_outside_main_month:
        new_row = {col: row_dict.get(col, "") for col in CSV_HEADERS}
        # 保持原有日期（如果有的话）
        if "日期" not in new_row or not new_row["日期"]:
            date_val = row_dict.get("日期", "")
            if pd.notna(date_val) and str(date_val).strip():
                try:
                    date_obj = pd.to_datetime(str(date_val).strip())
                    new_row["日期"] = date_obj.strftime("%Y-%m-%d")
                except Exception:
                    new_row["日期"] = ""
        # 标准化假别
        leave_type_val = new_row.get("假別", "")
        if pd.notna(leave_type_val) and str(leave_type_val).strip():
            new_row["假別"] = normalize_leave_type(str(leave_type_val).strip())
        # 计算总时数（只要有上班時間和下班時間就计算，不限于出勤记录）
        start_time_val = new_row.get("上班時間", "")
        end_time_val = new_row.get("下班時間", "")
        start_time = str(start_time_val).strip() if pd.notna(start_time_val) and str(start_time_val).strip() else ""
        end_time = str(end_time_val).strip() if pd.notna(end_time_val) and str(end_time_val).strip() else ""
        if start_time and end_time:
            new_row["總時數"] = calculate_hours(start_time, end_time)
        
        # 计算加班時數(分鐘)（只要有加班時間(起)和加班時間(迄)就计算）
        overtime_start_val = new_row.get("加班時間(起)", "")
        overtime_end_val = new_row.get("加班時間(迄)", "")
        overtime_start = str(overtime_start_val).strip() if pd.notna(overtime_start_val) and str(overtime_start_val).strip() else ""
        overtime_end = str(overtime_end_val).strip() if pd.notna(overtime_end_val) and str(overtime_end_val).strip() else ""
        if overtime_start and overtime_end:
            minutes = calculate_minutes(overtime_start, overtime_end)
            if minutes:
                new_row["加班時數(分鐘)"] = minutes
        full_df = pd.concat([full_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return full_df

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


# --- 互動式影像檢視器（放大/縮小/拖動/旋轉） ---
def render_image_viewer(image_bytes: bytes, caption: str = ""):
    """
    在 Streamlit 中嵌入可縮放/拖曳/旋轉的影像檢視器，視覺高度以 80vh 為基準。
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    img_src = f"data:image/jpeg;base64,{b64}"
    # 使用唯一ID避免多个实例冲突
    viewer_id = f"viewer_{hash(caption) % 100000}"
    img_id = f"img_{hash(caption) % 100000}"
    # 調整為 80vh 視覺高度
    html = f"""
    <div style="width:100%; height:80vh; border:1px solid rgba(0,0,0,0.08); position:relative; overflow:hidden; touch-action:none;">
      <div id="{viewer_id}" style="width:100%; height:100%; position:relative; background:#f6f6f6; display:flex; align-items:center; justify-content:center;">
        <img id="{img_id}" src="{img_src}" style="transform-origin:center center; cursor:grab; position:absolute; will-change:transform; user-select:none; -webkit-user-drag:none; max-width:none; max-height:none;"/>
      </div>
      <div style="position:absolute; right:8px; top:8px; background:rgba(255,255,255,0.85); padding:4px 8px; border-radius:6px; font-size:12px;">
        {caption}
      </div>
      <div style="position:absolute; left:8px; top:8px; display:flex; gap:4px; flex-direction:column;">
        <button id="rotate-left-{viewer_id}" style="background:rgba(255,255,255,0.9); border:1px solid rgba(0,0,0,0.2); border-radius:4px; padding:6px 10px; cursor:pointer; font-size:12px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">↺ 左轉90°</button>
        <button id="rotate-right-{viewer_id}" style="background:rgba(255,255,255,0.9); border:1px solid rgba(0,0,0,0.2); border-radius:4px; padding:6px 10px; cursor:pointer; font-size:12px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">↻ 右轉90°</button>
        <button id="rotate-180-{viewer_id}" style="background:rgba(255,255,255,0.9); border:1px solid rgba(0,0,0,0.2); border-radius:4px; padding:6px 10px; cursor:pointer; font-size:12px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">↻ 旋轉180°</button>
        <button id="flip-h-{viewer_id}" style="background:rgba(255,255,255,0.9); border:1px solid rgba(0,0,0,0.2); border-radius:4px; padding:6px 10px; cursor:pointer; font-size:12px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">⇄ 水平翻轉</button>
        <button id="flip-v-{viewer_id}" style="background:rgba(255,255,255,0.9); border:1px solid rgba(0,0,0,0.2); border-radius:4px; padding:6px 10px; cursor:pointer; font-size:12px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">⇅ 垂直翻轉</button>
        <button id="reset-{viewer_id}" style="background:rgba(255,255,255,0.9); border:1px solid rgba(0,0,0,0.2); border-radius:4px; padding:6px 10px; cursor:pointer; font-size:12px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">重置</button>
      </div>
    </div>
    <script>
    (function(){{
      const viewer = document.getElementById('{viewer_id}');
      const img = document.getElementById('{img_id}');
      let scale = 1;
      let originX = 0;
      let originY = 0;
      let rotation = 0;
      let flipH = 1;
      let flipV = 1;
      let dragging = false;
      let lastX = 0, lastY = 0;

      function updateTransform() {{
        // CSS transform顺序：先平移，再旋转，再翻转，最后缩放
        // 这样可以确保旋转中心在图片中心
        const translate = `translate(${{originX}}px, ${{originY}}px)`;
        const rotate = `rotate(${{rotation}}deg)`;
        const flipX = flipH === -1 ? 'scaleX(-1)' : '';
        const flipY = flipV === -1 ? 'scaleY(-1)' : '';
        const scaleStr = `scale(${{scale}})`;
        // 组合变换：translate -> rotate -> flip -> scale
        let transforms = [translate, rotate];
        if (flipH === -1) transforms.push(flipX);
        if (flipV === -1) transforms.push(flipY);
        transforms.push(scaleStr);
        img.style.transform = transforms.join(' ');
      }}

      function centerImage() {{
        // 初始化图片位置到中心
        const viewerRect = viewer.getBoundingClientRect();
        const imgRect = img.getBoundingClientRect();
        originX = (viewerRect.width - imgRect.width * scale) / 2;
        originY = (viewerRect.height - imgRect.height * scale) / 2;
        updateTransform();
      }}

      img.onload = function() {{
        // 等待一帧确保图片尺寸已计算
        setTimeout(centerImage, 10);
      }};

      viewer.onwheel = function(e) {{
        e.preventDefault();
        const rect = viewer.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const beforeScale = scale;
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        scale = Math.max(0.1, Math.min(10, scale * delta));
        originX -= (mx - originX) * (delta - 1);
        originY -= (my - originY) * (delta - 1);
        updateTransform();
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
        updateTransform();
      }});

      function endDrag(e) {{
        dragging = false;
        img.style.cursor = 'grab';
      }}
      viewer.addEventListener('pointerup', endDrag);
      viewer.addEventListener('pointercancel', endDrag);
      viewer.addEventListener('pointerleave', endDrag);

      document.getElementById('rotate-left-{viewer_id}').addEventListener('click', function() {{
        rotation -= 90;
        updateTransform();
      }});

      document.getElementById('rotate-right-{viewer_id}').addEventListener('click', function() {{
        rotation += 90;
        updateTransform();
      }});

      document.getElementById('rotate-180-{viewer_id}').addEventListener('click', function() {{
        rotation += 180;
        updateTransform();
      }});

      document.getElementById('flip-h-{viewer_id}').addEventListener('click', function() {{
        flipH *= -1;
        updateTransform();
      }});

      document.getElementById('flip-v-{viewer_id}').addEventListener('click', function() {{
        flipV *= -1;
        updateTransform();
      }});

      document.getElementById('reset-{viewer_id}').addEventListener('click', function() {{
        scale = 1;
        rotation = 0;
        flipH = 1;
        flipV = 1;
        centerImage();
      }});

      viewer.addEventListener('dblclick', function(e) {{
        scale = 1;
        rotation = 0;
        flipH = 1;
        flipV = 1;
        centerImage();
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


# 回呼：當 data_editor 的 widget 值改變時執行，立即保存編輯結果
def _on_data_editor_change_legacy():
    val = st.session_state.get("data_editor")
    if isinstance(val, pd.DataFrame):
        # 保存使用者的編輯快取
        st.session_state["edited_dataframe"] = val.copy()
        # 同步更新可下載的 CSV bytes
        try:
            buf = io.StringIO()
            val.to_csv(buf, index=False, encoding="utf-8-sig")
            safe = sanitize_csv_for_excel(buf.getvalue())
            st.session_state["edited_csv"] = safe.encode("utf-8-sig")
        except Exception:
            # 忽略轉換失敗但不要清掉 edited_dataframe
            pass


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
            options=["gemini-2.5-flash", "gemini-3-flash-preview"],
            index=0,
            help="建議大量解析時使用 gemini-2.5-flash 以節省成本；若結果品質不足再改用 gemini-3-flash-preview。",
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
        div.stButton > button[kind="primary"] {
            background-color: #cfefff !important; /* 正常狀態 */
            color: #000 !important;
            border: 1px solid #bce0ff !important;
            box-shadow: none !important;
            transition: background-color 0.2s ease, box-shadow 0.2s ease;
        }

        /* hover 滑過效果 */
        div.stButton > button[kind="primary"]:hover {
            background-color: #bde6ff !important; /* 稍微深一點藍 */
            border-color: #a9d7ff !important;
            box-shadow: 0 0 4px rgba(188, 224, 255, 0.9) !important; /* 微光 */
        }

        /* active 按下去效果 */
        div.stButton > button[kind="primary"]:active {
            background-color: #aadaff !important;
            border-color: #98cfff !important;
            box-shadow: 0 0 2px rgba(160, 205, 255, 0.7) inset !important;
        }

        /* focus 鍵盤切換時的外框效果 */
        div.stButton > button[kind="primary"]:focus {
            outline: none !important;
            box-shadow: 0 0 4px rgba(140, 200, 255, 0.9) !important;
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

            # 先批量提交所有任务，避免在循环中频繁更新UI
            for file_idx, uploaded_file in enumerate(files, start=1):
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

            # 使用进度条显示处理进度
            if futures:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_files = len(futures)
                completed = 0
                
                for fut in concurrent.futures.as_completed(futures):
                    completed += 1
                    progress = completed / total_files
                    progress_bar.progress(progress)
                    
                    filename, file_bytes, file_type = file_futures_map.get(fut, ("unknown", None, "unknown"))
                    status_text.text(f"處理中：{completed}/{total_files} - {filename}")
                    
                    try:
                        text = fut.result()
                    except Exception as e:
                        error_msg = str(e)
                        # 如果是429错误，提供更友好的错误信息
                        if "429" in error_msg or "資源耗盡" in error_msg or "Resource exhausted" in error_msg:
                            st.error(f"檔案 {filename} 處理失敗：API資源耗盡（429錯誤）。請稍後再試或減少同時處理的檔案數量。")
                        else:
                            st.warning(f"檔案 {filename} 處理失敗：{e}")
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

                progress_bar.progress(1.0)
                status_text.text(f"完成！已處理 {completed}/{total_files} 個檔案")
                progress_bar.empty()
                status_text.empty()

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
                    # 解析新檔案時清除舊的編輯快取，開始新的編輯階段
                    st.session_state["edited_dataframe"] = None
                    st.session_state["edited_csv"] = b""
                    
                    # 按文件分组数据
                    data_by_file = {}
                    for row_idx in range(len(df)):
                        if row_idx < len(row_to_file_mapping):
                            _, filename = row_to_file_mapping[row_idx]
                            if filename not in data_by_file:
                                data_by_file[filename] = []
                            data_by_file[filename].append(row_idx)
                    
                    # 为每个文件创建DataFrame，并扩展为完整的月份日期列表
                    df_by_file = {}
                    for filename, row_indices in data_by_file.items():
                        file_df = df.iloc[row_indices].copy().reset_index(drop=True)
                        # 扩展为完整的月份日期列表
                        expanded_df = expand_dataframe_to_monthly(file_df, filename)
                        # 计算总时数和加班時數(分鐘)（只要有对应时间字段就计算）
                        for idx, row in expanded_df.iterrows():
                            # 计算总时数
                            start_time_val = row.get("上班時間", "")
                            end_time_val = row.get("下班時間", "")
                            start_time = str(start_time_val).strip() if pd.notna(start_time_val) and str(start_time_val).strip() else ""
                            end_time = str(end_time_val).strip() if pd.notna(end_time_val) and str(end_time_val).strip() else ""
                            total_hours_val = row.get("總時數", "")
                            total_hours = str(total_hours_val).strip() if pd.notna(total_hours_val) and str(total_hours_val).strip() else ""
                            if start_time and end_time and not total_hours:
                                expanded_df.at[idx, "總時數"] = calculate_hours(start_time, end_time)
                            
                            # 计算加班時數(分鐘)
                            overtime_start_val = row.get("加班時間(起)", "")
                            overtime_end_val = row.get("加班時間(迄)", "")
                            overtime_start = str(overtime_start_val).strip() if pd.notna(overtime_start_val) and str(overtime_start_val).strip() else ""
                            overtime_end = str(overtime_end_val).strip() if pd.notna(overtime_end_val) and str(overtime_end_val).strip() else ""
                            overtime_minutes_val = row.get("加班時數(分鐘)", "")
                            overtime_minutes = str(overtime_minutes_val).strip() if pd.notna(overtime_minutes_val) and str(overtime_minutes_val).strip() else ""
                            if overtime_start and overtime_end and not overtime_minutes:
                                minutes = calculate_minutes(overtime_start, overtime_end)
                                if minutes:
                                    expanded_df.at[idx, "加班時數(分鐘)"] = minutes
                        df_by_file[filename] = expanded_df
                    
                    st.session_state["data_by_file"] = df_by_file
                    st.session_state["edited_data_by_file"] = {}
            except Exception:
                st.warning("CSV 預覽失敗，但仍可下載原始文字。請檢查欄位與分隔符號（建議確保逗號分隔與首列表頭）。")
                st.session_state["parsed_dataframe"] = None
                st.session_state["data_by_file"] = {}
                st.session_state["edited_data_by_file"] = {}

        except Exception as e:
            st.error(f"解析失敗：{e}")

    # 顯示解析結果（基於 session_state）
    data_by_file = st.session_state.get("data_by_file", {})
    file_previews = st.session_state.get("file_previews", {})
    edited_data_by_file = st.session_state.get("edited_data_by_file", {})

    if data_by_file and len(data_by_file) > 0:
        st.subheader("解析結果（按檔案分頁）")
        
        # 添加CSS样式：设置表格编辑器高度，支持Excel式粘贴
        st.markdown(
            """
            <style>
            div[data-testid="stDataEditor"] > div {
                height: 80vh !important;
                min-height: 600px !important;
            }
            div[data-testid="stDataEditor"] {
                height: 80vh !important;
                min-height: 600px !important;
            }
            /* 支持Excel式大范围粘贴：确保单元格可以编辑，支持多行多列粘贴 */
            .stDataEditor input, .stDataEditor textarea {
                font-family: monospace;
                white-space: pre-wrap;
            }
            /* 优化粘贴体验 */
            .stDataEditor [contenteditable="true"] {
                white-space: pre-wrap;
            }
            /* 出勤日期时间group的背景色（浅蓝色）- 索引0-6 */
            .stDataEditor th[data-column-index="0"],
            .stDataEditor th[data-column-index="1"],
            .stDataEditor th[data-column-index="2"],
            .stDataEditor th[data-column-index="3"],
            .stDataEditor th[data-column-index="4"],
            .stDataEditor th[data-column-index="5"],
            .stDataEditor th[data-column-index="6"],
            .stDataEditor td[data-column-index="0"],
            .stDataEditor td[data-column-index="1"],
            .stDataEditor td[data-column-index="2"],
            .stDataEditor td[data-column-index="3"],
            .stDataEditor td[data-column-index="4"],
            .stDataEditor td[data-column-index="5"],
            .stDataEditor td[data-column-index="6"] {
                background-color: #e3f2fd !important;
            }
            /* 请假group的背景色（浅绿色）- 索引7-13 */
            .stDataEditor th[data-column-index="7"],
            .stDataEditor th[data-column-index="8"],
            .stDataEditor th[data-column-index="9"],
            .stDataEditor th[data-column-index="10"],
            .stDataEditor th[data-column-index="11"],
            .stDataEditor th[data-column-index="12"],
            .stDataEditor th[data-column-index="13"],
            .stDataEditor td[data-column-index="7"],
            .stDataEditor td[data-column-index="8"],
            .stDataEditor td[data-column-index="9"],
            .stDataEditor td[data-column-index="10"],
            .stDataEditor td[data-column-index="11"],
            .stDataEditor td[data-column-index="12"],
            .stDataEditor td[data-column-index="13"] {
                background-color: #e8f5e9 !important;
            }
            /* 加班group的背景色（浅橙色）- 索引14-18 */
            .stDataEditor th[data-column-index="14"],
            .stDataEditor th[data-column-index="15"],
            .stDataEditor th[data-column-index="16"],
            .stDataEditor th[data-column-index="17"],
            .stDataEditor th[data-column-index="18"],
            .stDataEditor td[data-column-index="14"],
            .stDataEditor td[data-column-index="15"],
            .stDataEditor td[data-column-index="16"],
            .stDataEditor td[data-column-index="17"],
            .stDataEditor td[data-column-index="18"] {
                background-color: #fff3e0 !important;
            }
            /* 备注列的背景色（浅灰色）- 索引19 */
            .stDataEditor th[data-column-index="19"],
            .stDataEditor td[data-column-index="19"] {
                background-color: #f5f5f5 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # 使用tabs按文件分页
        file_names = list(data_by_file.keys())
        tabs = st.tabs([f"📄 {name}" for name in file_names])
        
        for tab_idx, (filename, tab) in enumerate(zip(file_names, tabs)):
            with tab:
                # 创建左右两列布局：左侧图片预览，右侧表格编辑
                left_col, right_col = st.columns([1, 1])
                
                # 左侧：图片预览（带旋转功能）
                with left_col:
                    # 使用columns将标题和文件名放在同一行
                    title_col, filename_col = st.columns([1, 2])
                    with title_col:
                        st.markdown("**檔案預覽**")
                    with filename_col:
                        st.caption(f"**檔案：** {filename}")
                    
                    if filename in file_previews:
                        file_bytes = file_previews[filename]
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
                        st.info(f"找不到檔案：{filename}")
                
                # 右侧：表格编辑器（支持Excel式大范围粘贴）
                with right_col:
                    st.markdown("**表格資料（可編輯，支援Excel式大範圍貼上）**")
                    
                    # Replace功能：在表格上方添加查找替换功能
                    with st.expander("🔍 查找與替換", expanded=False):
                        replace_col1, replace_col2, replace_col3 = st.columns([2, 2, 1])
                        with replace_col1:
                            find_what = st.text_input("Find what:", key=f"find_what_{filename}", value="")
                        with replace_col2:
                            replace_with = st.text_input("Replace with:", key=f"replace_with_{filename}", value="")
                        with replace_col3:
                            st.write("")  # 占位
                            st.write("")  # 占位
                            replace_button = st.button("替換", key=f"replace_btn_{filename}", use_container_width=True)
                    
                    # 获取该文件对应的DataFrame
                    df_for_file = data_by_file[filename]
                    
                    # 定义editor_key（用于检查最新的编辑数据）
                    editor_key = f"data_editor_{filename}"
                    
                    # 检查是否有编辑过的版本
                    # 优先级：1. st.data_editor中的最新数据（最准确） 2. edited_data_by_file 3. 原始数据
                    # 注意：在替换功能执行后，editor_key 会被清除，所以 base_df 会从 edited_data_by_file 获取替换后的数据
                    base_df = None
                    
                    # 首先检查st.data_editor中是否有最新数据（用户可能刚编辑但还没触发on_change）
                    if editor_key in st.session_state:
                        editor_data = st.session_state[editor_key]
                        if isinstance(editor_data, pd.DataFrame) and len(editor_data) > 0:
                            base_df = editor_data.copy()
                    
                    # 如果没有从editor获取到，则从edited_data_by_file读取
                    # 这是关键：替换功能执行后，editor_key 会被清除，所以会从这里获取替换后的数据
                    if base_df is None and filename in edited_data_by_file:
                        edited_df = edited_data_by_file[filename]
                        if isinstance(edited_df, pd.DataFrame) and len(edited_df) > 0:
                            base_df = edited_df.copy()
                    
                    # 如果还是没有，使用原始数据
                    if base_df is None:
                        base_df = df_for_file.copy()
                    
                    # 注意：替换功能将在 st.data_editor 之后执行，以便直接使用 edited_df（最可靠的数据源）
                    # 处理替换功能的标志（在 st.data_editor 之后执行）
                    should_replace = replace_button and find_what
                    
                    # 准备编辑用的DataFrame：将所有列转换为字符串类型，以支持TextColumn和Excel式粘贴
                    df_for_editor = base_df.copy()
                    for col in df_for_editor.columns:
                        try:
                            df_for_editor[col] = df_for_editor[col].fillna("").astype(str)
                        except Exception:
                            df_for_editor[col] = df_for_editor[col].apply(lambda v: "" if pd.isna(v) else str(v))
                    
                    # 配置列：所有列都设为可编辑的TextColumn，支持Excel式粘贴
                    # 由于我们已经将所有列转换为字符串，所以可以使用TextColumn
                    column_config = {}
                    try:
                        for col in df_for_editor.columns:
                            column_config[col] = st.column_config.TextColumn(col, required=False)
                    except Exception:
                        column_config = {}
                    
                    col_cfg = column_config if column_config else None
                    
                    # 在显示前，自动计算总时数和加班時數(分鐘)（只要有对应时间字段就计算）
                    if "上班時間" in df_for_editor.columns and "下班時間" in df_for_editor.columns and "總時數" in df_for_editor.columns:
                        for idx in df_for_editor.index:
                            # 计算总时数
                            start_time = str(df_for_editor.at[idx, "上班時間"]).strip() if pd.notna(df_for_editor.at[idx, "上班時間"]) else ""
                            end_time = str(df_for_editor.at[idx, "下班時間"]).strip() if pd.notna(df_for_editor.at[idx, "下班時間"]) else ""
                            current_hours = str(df_for_editor.at[idx, "總時數"]).strip() if pd.notna(df_for_editor.at[idx, "總時數"]) else ""
                            if start_time and end_time and (not current_hours or current_hours == "" or current_hours == "nan"):
                                hours = calculate_hours(start_time, end_time)
                                df_for_editor.at[idx, "總時數"] = hours
                    
                    # 计算加班時數(分鐘)
                    if "加班時間(起)" in df_for_editor.columns and "加班時間(迄)" in df_for_editor.columns and "加班時數(分鐘)" in df_for_editor.columns:
                        for idx in df_for_editor.index:
                            overtime_start = str(df_for_editor.at[idx, "加班時間(起)"]).strip() if pd.notna(df_for_editor.at[idx, "加班時間(起)"]) else ""
                            overtime_end = str(df_for_editor.at[idx, "加班時間(迄)"]).strip() if pd.notna(df_for_editor.at[idx, "加班時間(迄)"]) else ""
                            current_overtime_minutes = str(df_for_editor.at[idx, "加班時數(分鐘)"]).strip() if pd.notna(df_for_editor.at[idx, "加班時數(分鐘)"]) else ""
                            if overtime_start and overtime_end and (not current_overtime_minutes or current_overtime_minutes == "" or current_overtime_minutes == "nan"):
                                minutes = calculate_minutes(overtime_start, overtime_end)
                                if minutes:
                                    df_for_editor.at[idx, "加班時數(分鐘)"] = minutes
                    
                    # 定义editor_key（用于st.data_editor）
                    editor_key = f"data_editor_{filename}"
                    
                    # 创建回调函数：确保每次编辑都保存，并自动计算总时数和加班時數(分鐘)
                    def make_on_change(fname):
                        def on_change():
                            try:
                                val = st.session_state.get(f"data_editor_{fname}")
                                if isinstance(val, pd.DataFrame):
                                    # 自动计算总时数和加班時數(分鐘)（只要有对应时间字段就计算）
                                    df_copy = val.copy()
                                    
                                    # 计算总时数
                                    if "上班時間" in df_copy.columns and "下班時間" in df_copy.columns and "總時數" in df_copy.columns:
                                        for idx in df_copy.index:
                                            start_time = str(df_copy.at[idx, "上班時間"]).strip() if pd.notna(df_copy.at[idx, "上班時間"]) else ""
                                            end_time = str(df_copy.at[idx, "下班時間"]).strip() if pd.notna(df_copy.at[idx, "下班時間"]) else ""
                                            if start_time and end_time:
                                                hours = calculate_hours(start_time, end_time)
                                                df_copy.at[idx, "總時數"] = hours
                                    
                                    # 计算加班時數(分鐘)
                                    if "加班時間(起)" in df_copy.columns and "加班時間(迄)" in df_copy.columns and "加班時數(分鐘)" in df_copy.columns:
                                        for idx in df_copy.index:
                                            overtime_start = str(df_copy.at[idx, "加班時間(起)"]).strip() if pd.notna(df_copy.at[idx, "加班時間(起)"]) else ""
                                            overtime_end = str(df_copy.at[idx, "加班時間(迄)"]).strip() if pd.notna(df_copy.at[idx, "加班時間(迄)"]) else ""
                                            if overtime_start and overtime_end:
                                                minutes = calculate_minutes(overtime_start, overtime_end)
                                                if minutes:
                                                    df_copy.at[idx, "加班時數(分鐘)"] = minutes
                                    
                                    # 立即保存到edited_data_by_file
                                    edited_data_by_file = st.session_state.get("edited_data_by_file", {})
                                    if not isinstance(edited_data_by_file, dict):
                                        edited_data_by_file = {}
                                    edited_data_by_file[fname] = df_copy
                                    st.session_state["edited_data_by_file"] = edited_data_by_file
                            except Exception as e:
                                # 记录错误但不中断用户体验
                                import traceback
                                print(f"Error in on_change callback for {fname}: {e}")
                                print(traceback.format_exc())
                        return on_change
                    
                    # st.data_editor原生支持Excel式大范围粘贴（多行多列）
                    # 用户可以直接从Excel复制多行多列数据，然后粘贴到表格中
                    edited_df = st.data_editor(
                        df_for_editor,
                        column_config=col_cfg,
                        width='stretch',
                        num_rows="dynamic",  # 改为dynamic以支持添加行，方便大范围粘贴
                        key=editor_key,
                        on_change=make_on_change(filename)
                    )
                    
                    # 在 st.data_editor 之后执行替换功能（关键：直接使用 edited_df，这是最可靠的数据源）
                    if should_replace:
                        # 直接使用 edited_df（st.data_editor 的返回值），这是用户当前在表格中看到的所有编辑
                        replace_base_df = edited_df.copy()
                        
                        # 自动计算总时数和加班時數(分鐘)（如果有对应字段）
                        if "上班時間" in replace_base_df.columns and "下班時間" in replace_base_df.columns and "總時數" in replace_base_df.columns:
                            for idx in replace_base_df.index:
                                start_time = str(replace_base_df.at[idx, "上班時間"]).strip() if pd.notna(replace_base_df.at[idx, "上班時間"]) else ""
                                end_time = str(replace_base_df.at[idx, "下班時間"]).strip() if pd.notna(replace_base_df.at[idx, "下班時間"]) else ""
                                if start_time and end_time:
                                    hours = calculate_hours(start_time, end_time)
                                    replace_base_df.at[idx, "總時數"] = hours
                        
                        if "加班時間(起)" in replace_base_df.columns and "加班時間(迄)" in replace_base_df.columns and "加班時數(分鐘)" in replace_base_df.columns:
                            for idx in replace_base_df.index:
                                overtime_start = str(replace_base_df.at[idx, "加班時間(起)"]).strip() if pd.notna(replace_base_df.at[idx, "加班時間(起)"]) else ""
                                overtime_end = str(replace_base_df.at[idx, "加班時間(迄)"]).strip() if pd.notna(replace_base_df.at[idx, "加班時間(迄)"]) else ""
                                if overtime_start and overtime_end:
                                    minutes = calculate_minutes(overtime_start, overtime_end)
                                    if minutes:
                                        replace_base_df.at[idx, "加班時數(分鐘)"] = minutes
                        
                        # 执行替换操作（基于用户当前编辑后的数据状态）
                        base_df = replace_base_df.copy()
                        replace_count = 0
                        # 在所有列中查找并替换
                        for col in base_df.columns:
                            # 转换为字符串类型，但先处理NaN值，避免变成"nan"字符串
                            col_data = base_df[col].fillna("").astype(str)
                            # 将"nan"字符串替换回空字符串
                            col_data = col_data.replace("nan", "")
                            # 计算替换前的匹配数量
                            matches = col_data.str.contains(str(find_what), regex=False, na=False).sum()
                            replace_count += matches
                            # 执行替换
                            col_data = col_data.str.replace(str(find_what), str(replace_with), regex=False)
                            # 保存替换后的数据，确保空字符串保持为空字符串，而不是"nan"
                            base_df[col] = col_data.replace("nan", "")
                        
                        # 保存替换后的数据到 edited_data_by_file
                        edited_data_by_file = st.session_state.get("edited_data_by_file", {})
                        if not isinstance(edited_data_by_file, dict):
                            edited_data_by_file = {}
                        edited_data_by_file[filename] = base_df.copy()
                        st.session_state["edited_data_by_file"] = edited_data_by_file
                        
                        # 清除editor_key的session_state，强制重新加载替换后的数据
                        if editor_key in st.session_state:
                            del st.session_state[editor_key]
                        
                        if replace_count > 0:
                            st.success(f"已將 \"{find_what}\" 替換為 \"{replace_with}\"（共 {replace_count} 處）")
                        else:
                            st.info(f"未找到 \"{find_what}\"")
                        
                        # 重要：在 rerun 前，确保 edited_data_by_file 中的数据已被保存
                        st.rerun()
                    
                    # 只在首次初始化时保存，避免每次渲染都更新session_state导致重新渲染
                    # 后续的编辑由on_change回调处理
                    # 从session_state获取最新值进行检查
                    current_edited_data = st.session_state.get("edited_data_by_file", {})
                    if filename not in current_edited_data:
                        try:
                            current_editor_value = st.session_state.get(editor_key)
                            if isinstance(current_editor_value, pd.DataFrame):
                                current_edited_data[filename] = current_editor_value.copy()
                                st.session_state["edited_data_by_file"] = current_edited_data
                        except Exception:
                            # 如果同步失败，至少保存edited_df（这是编辑器返回的值）
                            current_edited_data[filename] = edited_df.copy()
                            st.session_state["edited_data_by_file"] = current_edited_data
        
        st.divider()
        
        # 合并所有文件的编辑数据，生成最终的CSV
        # 优先从编辑器当前状态读取，确保获取最新数据（即使页面闲置后恢复）
        try:
            all_edited_dfs = []
            data_by_file = st.session_state.get("data_by_file", {})
            edited_data_by_file = st.session_state.get("edited_data_by_file", {})
            
            for filename in file_names:
                editor_key = f"data_editor_{filename}"
                # 优先从编辑器当前状态读取（最可靠）
                if editor_key in st.session_state:
                    editor_df = st.session_state[editor_key]
                    if isinstance(editor_df, pd.DataFrame) and len(editor_df) > 0:
                        all_edited_dfs.append(editor_df.copy())
                        continue
                
                # 如果编辑器状态不存在，从edited_data_by_file读取
                if filename in edited_data_by_file:
                    edited_df = edited_data_by_file[filename]
                    if isinstance(edited_df, pd.DataFrame) and len(edited_df) > 0:
                        all_edited_dfs.append(edited_df)
                        continue
                
                # 最后回退到原始数据（只添加有数据的文件）
                if filename in data_by_file:
                    df = data_by_file[filename]
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        all_edited_dfs.append(df)
            
            if all_edited_dfs:
                merged_df = pd.concat(all_edited_dfs, ignore_index=True)
                buf = io.StringIO()
                merged_df.to_csv(buf, index=False, encoding="utf-8-sig")
                safe_csv = sanitize_csv_for_excel(buf.getvalue())
                st.session_state["edited_csv"] = safe_csv.encode("utf-8-sig")
        except Exception as e:
            st.warning(f"合併資料時發生錯誤：{e}")
        
        # 下载按钮：下载所有编辑后的数据
        edited_csv_bytes = st.session_state.get("edited_csv", b"")
        st.download_button(
            label="下載 CSV（原始資料）",
            data=edited_csv_bytes,
            file_name="attendance.csv",
            mime="text/csv",
            width='content'
        )


if __name__ == "__main__":
    main()
