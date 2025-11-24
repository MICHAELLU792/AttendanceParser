# python
import os
import threading
import concurrent.futures

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

# Configure Gemini API if ENV_API_KEY is available
if ENV_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=ENV_API_KEY)
    except Exception:
        # 忽略配置錯誤，工作時會再檢查
        pass


