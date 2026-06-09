"""土耳其语 Mizan PDF 表格提取 — 仅 PDF，无 mock JSON"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from row_filter import filter_records
from meta_extractor import extract_doc_meta

COLUMNS = [
    "hesap_kodu",
    "hesap_adi",
    "borc",
    "alacak",
    "borc_bakiye",
    "alacak_bakiye",
]


def extract_mizan_from_pdf(pdf_path: str | Path) -> pd.DataFrame:
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber") from e

    rows: list[list] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table({
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "min_words_vertical": 3,
            })
            if table:
                rows.extend(table)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows)
    if len(df.columns) >= len(COLUMNS):
        df = df.iloc[:, : len(COLUMNS)]
        df.columns = COLUMNS
    else:
        df.columns = COLUMNS[: len(df.columns)]
    return df.dropna(how="all")


def clean_turkish_number(value: Any) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    s = str(value).strip()
    if not s or s in ("-", "—"):
        return 0.0
    s = s.replace(" ", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def get_account_level(code: str) -> int:
    if not code:
        return 0
    code = str(code).strip()
    dot_parts = len(code.split("."))
    if dot_parts > 1:
        return min(3 + dot_parts - 1, 4)
    return min(len(code.split(".")[0]), 4)


def pdf_to_records(pdf_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """PDF → 清洗后的 records + doc_meta"""
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"仅支持 PDF 文件，收到: {path}")

    df = extract_mizan_from_pdf(path)
    raw_records = []
    for _, row in df.iterrows():
        code = str(row.get("hesap_kodu", "") or "").strip()
        name = str(row.get("hesap_adi", "") or "").strip()
        if not name and not code:
            continue
        raw_records.append({
            "hesap_kodu": code,
            "hesap_adi": name,
            "level": get_account_level(code),
            "borc": clean_turkish_number(row.get("borc")),
            "alacak": clean_turkish_number(row.get("alacak")),
            "borc_bakiye": clean_turkish_number(row.get("borc_bakiye")),
            "alacak_bakiye": clean_turkish_number(row.get("alacak_bakiye")),
        })

    records, dropped = filter_records(raw_records)
    meta = extract_doc_meta(path)
    meta.update({
        "source_file": str(path.resolve()),
        "raw_rows": len(raw_records),
        "valid_rows": len(records),
        "dropped_rows": len(dropped),
    })
    return records, meta
