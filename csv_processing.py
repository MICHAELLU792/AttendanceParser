# python
import io
import csv
import re
import pandas as pd

from app_config import CSV_HEADERS


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

