# python
# config/logging_gemini.py
from datetime import datetime
from pathlib import Path
import csv
import logging
import os
import io
import time
import re
import threading
from typing import Any, Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# 每 1,000,000 tokens 的價格（USD）
# 參考 Gemini 官方定價：
# gemini-2.0-flash      input: $0.15 / 1M, output: $0.60 / 1M
# gemini-2.0-flash-lite input: $0.075 / 1M, output: $0.30 / 1M
MODEL_PRICING = {
    "gemini-2.0-flash": {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    "gemini-2.0-flash-lite": {
        "input_per_million": 0.075,
        "output_per_million": 0.30,
    },
}

# 用於避免同時執行多個 flush 的鎖
_flush_lock = threading.Lock()


def ensure_log_dir() -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "pending").mkdir(parents=True, exist_ok=True)
        logger.debug("ensure_log_dir: %s", str(LOG_DIR))
    except Exception:
        logger.exception("建立 log 目錄失敗: %s", str(LOG_DIR))


def calc_cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> Dict[str, float]:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    in_rate = pricing["input_per_million"]
    out_rate = pricing["output_per_million"]
    input_cost = prompt_tokens / 1_000_000 * in_rate
    output_cost = output_tokens / 1_000_000 * out_rate
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
    }


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return 0


def _get_from_obj(obj: Any, key: str) -> Optional[Any]:
    if obj is None:
        return None
    try:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            return getattr(obj, key)
    except Exception:
        pass
    return None


def _find_token_values(obj: Any, depth: int = 0, max_depth: int = 6) -> Dict[str, int]:
    if depth > max_depth or obj is None:
        return {"prompt": 0, "output": 0, "total": 0}
    found = {"prompt": 0, "output": 0, "total": 0}

    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if isinstance(v, (int, float, str)) and str(v).strip().lstrip("+-").replace(".", "", 1).isdigit():
                    iv = _safe_int(v)
                    if "prompt" in lk or "input" in lk:
                        found["prompt"] = max(found["prompt"], iv)
                    elif "completion" in lk or "output" in lk:
                        found["output"] = max(found["output"], iv)
                    elif "total" in lk or "token_count" in lk or lk.endswith("tokens"):
                        found["total"] = max(found["total"], iv)
                    elif "token" in lk and found["total"] == 0:
                        found["total"] = max(found["total"], iv)
                else:
                    sub = _find_token_values(v, depth + 1, max_depth)
                    found["prompt"] = max(found["prompt"], sub["prompt"])
                    found["output"] = max(found["output"], sub["output"])
                    found["total"] = max(found["total"], sub["total"])

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                sub = _find_token_values(item, depth + 1, max_depth)
                found["prompt"] = max(found["prompt"], sub["prompt"])
                found["output"] = max(found["output"], sub["output"])
                found["total"] = max(found["total"], sub["total"])

        else:
            for attr in dir(obj):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(obj, attr)
                except Exception:
                    continue
                sub = _find_token_values(val, depth + 1, max_depth)
                found["prompt"] = max(found["prompt"], sub["prompt"])
                found["output"] = max(found["output"], sub["output"])
                found["total"] = max(found["total"], sub["total"])
    except Exception:
        logger.exception("遞迴掃描 token 欄位失敗")

    return found


def _extract_tokens(metadata: Any) -> Tuple[int, int, int]:
    try:
        candidates = [
            ("prompt_token_count", "candidates_token_count", "total_token_count"),
            ("prompt_tokens", "output_tokens", "total_tokens"),
            ("input_tokens", "output_tokens", "total_tokens"),
            ("prompt_tokens", "completion_tokens", "total_tokens"),
        ]
        for p_key, o_key, t_key in candidates:
            p = _get_from_obj(metadata, p_key)
            o = _get_from_obj(metadata, o_key)
            t = _get_from_obj(metadata, t_key)
            if p is not None or o is not None or t is not None:
                p_i = _safe_int(p)
                o_i = _safe_int(o)
                t_i = _safe_int(t) if t is not None else (p_i + o_i)
                logger.debug("tokens from direct keys: %s %s %s", p_i, o_i, t_i)
                return p_i, o_i, t_i

        usage = _get_from_obj(metadata, "usage") or _get_from_obj(metadata, "token_usage")
        if usage:
            p = _get_from_obj(usage, "prompt_tokens") or _get_from_obj(usage, "input_tokens")
            o = _get_from_obj(usage, "completion_tokens") or _get_from_obj(usage, "output_tokens")
            t = _get_from_obj(usage, "total_tokens")
            p_i = _safe_int(p)
            o_i = _safe_int(o)
            t_i = _safe_int(t) if t is not None else (p_i + o_i)
            logger.debug("tokens from usage dict: %s %s %s", p_i, o_i, t_i)
            return p_i, o_i, t_i

        if isinstance(metadata, dict) and "candidates" in metadata and isinstance(metadata["candidates"], (list, tuple)):
            prompt_acc = 0
            output_acc = 0
            total_acc = 0
            for c in metadata["candidates"]:
                if isinstance(c, dict):
                    prompt_acc = max(
                        prompt_acc,
                        _safe_int(c.get("prompt_token_count") or c.get("input_tokens") or 0),
                    )
                    output_acc = max(
                        output_acc,
                        _safe_int(c.get("candidates_token_count") or c.get("output_tokens") or 0),
                    )
                    total_acc = max(
                        total_acc,
                        _safe_int(c.get("total_token_count") or c.get("token_count") or 0),
                    )
                    sub = _find_token_values(c)
                    prompt_acc = max(prompt_acc, sub["prompt"])
                    output_acc = max(output_acc, sub["output"])
                    total_acc = max(total_acc, sub["total"])
            if prompt_acc or output_acc or total_acc:
                logger.debug("tokens from candidates list: %s %s %s", prompt_acc, output_acc, total_acc)
                return prompt_acc, output_acc, total_acc

        scanned = _find_token_values(metadata)
        p_i, o_i, t_i = scanned["prompt"], scanned["output"], scanned["total"]
        if t_i == 0 and (p_i or o_i):
            t_i = p_i + o_i
        logger.debug("tokens from recursive scan: %s %s %s", p_i, o_i, t_i)
        return p_i, o_i, t_i

    except Exception:
        logger.exception("解析 token 欄位時發生例外")
        return 0, 0, 0


# --- 簡易鎖機制（透過建立 lock 檔案） ---
def _acquire_lock(lock_path: Path, timeout: float = 5.0, poll: float = 0.1, stale_seconds: float = 30.0) -> bool:
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{os.getpid()}\n{time.time()}".encode("utf-8"))
            finally:
                os.close(fd)
            logger.debug("acquired lock: %s", lock_path)
            return True
        except FileExistsError:
            try:
                stat = lock_path.stat()
                age = time.time() - stat.st_mtime
                if age > stale_seconds:
                    try:
                        lock_path.unlink()
                        logger.warning("stale lock removed: %s", lock_path)
                        continue
                    except Exception:
                        logger.exception("移除 stale lock 失敗: %s", lock_path)
                if time.time() - start > timeout:
                    logger.warning("取得 lock 超時: %s", lock_path)
                    return False
                time.sleep(poll)
            except FileNotFoundError:
                continue
        except Exception:
            logger.exception("取得 lock 時發生例外: %s", lock_path)
            return False


def _release_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            lock_path.unlink()
            logger.debug("released lock: %s", lock_path)
    except Exception:
        logger.exception("釋放 lock 失敗: %s", lock_path)


# --- CSV helpers & robust write with retries and pending fallback ---
def _csv_line_from_row(fieldnames, row) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writerow(row)
    line = buf.getvalue()
    if not line.endswith("\n"):
        line += "\n"
    return line


def _csv_header_line(fieldnames) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(fieldnames)
    header = buf.getvalue()
    if not header.endswith("\n"):
        header += "\n"
    return header


def _write_row_with_retries(
    log_path: Path,
    fieldnames,
    row: dict,
    is_empty: bool,
    max_retries: int = 3,
    retry_delay: float = 0.1,
) -> bool:
    header_line = _csv_header_line(fieldnames)
    line = _csv_line_from_row(fieldnames, row)

    for attempt in range(1, max_retries + 1):
        try:
            # append as single write to reduce partial-line risk
            with log_path.open("a", newline="", encoding="utf-8-sig") as f:
                # double-check emptiness to avoid race
                try:
                    current_size = log_path.stat().st_size if log_path.exists() else 0
                except Exception:
                    current_size = 0
                if is_empty and current_size == 0:
                    f.write(header_line)
                    is_empty = False
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    logger.exception("fsync 失敗（但已嘗試 flush）: %s", str(log_path))
            logger.info("已寫入 Gemini log: %s (attempt %s)", str(log_path), attempt)
            return True
        except Exception:
            logger.exception("寫入 log 嘗試 %s 失敗: %s", attempt, str(log_path))
            time.sleep(retry_delay * attempt)

    # fallback: 寫入 pending 檔（避免遺失）
    try:
        pending_dir = LOG_DIR / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_path = pending_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}-{os.getpid()}.csv"
        with pending_path.open("w", encoding="utf-8-sig", newline="") as pf:
            pf.write(header_line)
            pf.write(line)
            pf.flush()
            try:
                os.fsync(pf.fileno())
            except Exception:
                logger.exception("pending fsync 失敗: %s", str(pending_path))
        logger.warning("寫入主檔失敗，已寫入 pending：%s", str(pending_path))
        return False
    except Exception:
        logger.exception("寫入 pending 檔失敗，最後遺失該筆 log")
        return False


def log_gemini_usage(
    model: str,
    usage_metadata: Any,
    *,
    uploaded_filename: str = "",
    extra_info: str = "",
) -> None:
    """
    寫入當天 log（安全、容錯、支援多次 append、重試與 pending 回退）。
    每次呼叫時會嘗試在背景處理 pending（非阻塞，並避免重入）。
    """
    try:
        ensure_log_dir()

        # 嘗試啟動非阻塞的背景 flush（若已有 flush 在執行則跳過）
        try:
            if _flush_lock.acquire(blocking=False):
                def _bg_worker():
                    try:
                        flush_pending_to_main()
                    except Exception:
                        logger.exception("background flush_pending_to_main() 失敗")
                    finally:
                        try:
                            _flush_lock.release()
                        except Exception:
                            pass

                threading.Thread(target=_bg_worker, daemon=True).start()
            else:
                logger.debug("flush_pending_to_main already running, skip starting another")
        except Exception:
            logger.exception("啟動背景 flush 失敗")

        today_str = datetime.now().strftime("%Y-%m-%d")
        log_path = LOG_DIR / f"{today_str}.csv"
        lock_path = LOG_DIR / f"{today_str}.lock"
        ts = datetime.now().isoformat(timespec="seconds")

        prompt_tokens, output_tokens, total_tokens = _extract_tokens(usage_metadata)
        if total_tokens == 0 and (prompt_tokens or output_tokens):
            total_tokens = prompt_tokens + output_tokens

        cost = calc_cost_usd(model, prompt_tokens, output_tokens)

        row = {
            "timestamp": ts,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost_usd": cost["input_cost"],
            "output_cost_usd": cost["output_cost"],
            "total_cost_usd": cost["total_cost"],
            "filename": (uploaded_filename or "")[:255],
        }

        fieldnames = [
            "timestamp",
            "model",
            "prompt_tokens",
            "output_tokens",
            "total_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "total_cost_usd",
            "filename",
        ]

        # 取得鎖，若失敗則仍嘗試寫入一次（避免完全不紀錄）
        locked = _acquire_lock(lock_path, timeout=3.0)
        try:
            file_exists = log_path.exists()
            is_empty = (not file_exists) or (file_exists and log_path.stat().st_size == 0)
            logger.debug("log_path=%s exists=%s is_empty=%s", str(log_path), file_exists, is_empty)

            success = _write_row_with_retries(log_path, fieldnames, row, is_empty, max_retries=3, retry_delay=0.1)
            if not success:
                logger.warning("寫入主日誌失敗，已 fallback 到 pending")
        finally:
            if locked:
                _release_lock(lock_path)

    except Exception:
        logger.exception("寫入 Gemini 使用日誌時發生例外，已忽略以免影響主流程")


def flush_pending_to_main() -> None:
    """
    掃描 `logs/pending`，將 pending CSV 合併回對應的 `logs/YYYY-MM-DD.csv`。
    - 會嘗試解析每列的 `timestamp` 取得目標日期（失敗則退回到檔名或今天）。
    - 對每個目標主檔一次取得 lock、append 多列（包含 header 若檔案為空），寫入後 fsync。
    - 若所有目標檔都寫入成功，刪除 pending 檔；失敗時保留。
    """
    ensure_log_dir()
    pending_dir = LOG_DIR / "pending"
    if not pending_dir.exists():
        return

    pending_files = list(pending_dir.glob("*.csv"))
    if not pending_files:
        logger.debug("no pending files to flush")
        return

    # 讀取 pending 檔，並按要寫入的主檔群組資料
    targets: Dict[Path, Dict[str, List[Tuple[List[str], Dict[str, str]]]]] = {}
    file_map: Dict[Path, List[Path]] = {}

    for p in pending_files:
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as pf:
                reader = csv.DictReader(pf)
                rows = list(reader)
                fieldnames = reader.fieldnames or [
                    "timestamp",
                    "model",
                    "prompt_tokens",
                    "output_tokens",
                    "total_tokens",
                    "input_cost_usd",
                    "output_cost_usd",
                    "total_cost_usd",
                    "filename",
                ]
            if not rows:
                try:
                    p.unlink()
                    logger.info("刪除空 pending 檔: %s", str(p))
                except Exception:
                    logger.exception("刪除空 pending 檔失敗: %s", str(p))
                continue

            file_map[p] = []
            for row in rows:
                ts = (row.get("timestamp") or "").strip()
                date_str = ""
                if ts:
                    try:
                        date_str = datetime.fromisoformat(ts).date().isoformat()
                    except Exception:
                        date_str = ""
                if not date_str:
                    m = re.match(r"^(\d{8})", p.name)
                    if m:
                        try:
                            date_str = datetime.strptime(m.group(1), "%Y%m%d").date().isoformat()
                        except Exception:
                            date_str = ""
                if not date_str:
                    date_str = datetime.now().date().isoformat()

                log_path = LOG_DIR / f"{date_str}.csv"
                file_map[p].append(log_path)

                key = ",".join(fieldnames)
                if log_path not in targets:
                    targets[log_path] = {}
                if key not in targets[log_path]:
                    targets[log_path][key] = []
                targets[log_path][key].append((fieldnames, row))
        except Exception:
            logger.exception("讀取 pending 檔失敗，保留檔案以便下次再試: %s", str(p))
            continue

    # 寫入每個目標主檔（依 fieldnames 群組以確保 header 一致）
    write_success_for_pending: Dict[Path, bool] = {p: True for p in file_map.keys()}
    for log_path, group in targets.items():
        lock_path = LOG_DIR / f"{log_path.stem}.lock"
        locked = _acquire_lock(lock_path, timeout=5.0)
        try:
            try:
                file_exists = log_path.exists()
                is_empty = (not file_exists) or (file_exists and log_path.stat().st_size == 0)
            except Exception:
                is_empty = True

            for key, entries in group.items():
                if not entries:
                    continue
                fieldnames = entries[0][0]
                rows = [e[1] for e in entries]
                header_line = _csv_header_line(fieldnames)
                lines = "".join(_csv_line_from_row(fieldnames, r) for r in rows)

                try:
                    with log_path.open("a", newline="", encoding="utf-8-sig") as f:
                        try:
                            current_size = log_path.stat().st_size if log_path.exists() else 0
                        except Exception:
                            current_size = 0
                        if is_empty and current_size == 0:
                            f.write(header_line)
                            is_empty = False
                        f.write(lines)
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            logger.exception("fsync 失敗（但已嘗試 flush）: %s", str(log_path))
                    logger.info("已把 %d pending 列寫入 %s", len(rows), str(log_path))
                except Exception:
                    logger.exception("寫入主檔失敗: %s", str(log_path))
                    for p2, lst in file_map.items():
                        if log_path in lst:
                            write_success_for_pending[p2] = False
        finally:
            if locked:
                _release_lock(lock_path)

    # 刪除那些全部目標寫入都成功的 pending 檔
    for p, ok in write_success_for_pending.items():
        if ok:
            try:
                p.unlink()
                logger.info("已刪除已 flush 的 pending 檔: %s", str(p))
            except Exception:
                logger.exception("刪除 pending 檔失敗: %s", str(p))


# 模組載入時確保 logs 與 pending 目錄存在，並在背景執行一次 pending flush
try:
    ensure_log_dir()
    logger.debug("logs 與 pending 已初始化: %s", str(LOG_DIR))

    def _background_flush_once():
        try:
            flush_pending_to_main()
        except Exception:
            logger.exception("background flush_pending_to_main() 失敗")

    threading.Thread(target=_background_flush_once, daemon=True).start()

except Exception:
    logger.exception("初始化 logs 資料夾失敗")
