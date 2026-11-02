import os
import io
import re
import csv
import logging
import zipfile
import subprocess
import shutil
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("smart_review_analyzer.parser")

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit
MAX_REVIEWS_LIMIT = 500  # Configured limit to protect server

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".pdf", ".docx", ".doc", ".xlsx", ".xls"}

# Keywords to detect review text columns in tabular data (CSV / Excel)
PRIMARY_REVIEW_KEYWORDS = [
    "review_text", "review text", "customer_review", "customer review",
    "reviews", "review", "comment", "comments", "feedback",
    "text", "content", "description", "body", "message",
    "opinion", "critique", "testimonial", "statement", "notes"
]

METADATA_COLUMN_KEYWORDS = [
    "id", "review_id", "review id", "no", "num", "number", "index",
    "code", "date", "time", "timestamp", "rating", "score", "stars",
    "author", "user", "username", "customer_id", "customer id",
    "status", "category", "verified", "helpful", "likes", "dislikes"
]

REVIEW_COLUMN_KEYWORDS = PRIMARY_REVIEW_KEYWORDS

TABLE_HEADER_KEYWORDS = {
    "id", "no", "#", "review", "feedback", "comment", "rating",
    "text", "customer", "sentiment", "date", "author", "user", "score"
}

# Regex patterns for detecting review boundaries
NUMBERED_PATTERN = re.compile(
    r'^\s*(?:(?:review|item|feedback|customer|#)?\s*(\d+)[\.\:\)\-\]]|\[\s*(\d+)\s*\])\s+',
    re.IGNORECASE
)
BULLET_PATTERN = re.compile(r'^\s*[\u2022\u2023\u25E6\u2043\u2219\*\-\+▪▫►■]\s+')
SECTION_DIVIDER_PATTERN = re.compile(r'^\s*(?:[-=_*~]{3,})\s*$')


class DocumentParsingError(Exception):
    """Custom exception raised when document parsing fails."""
    pass


def extract_docx_content(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts text from DOCX documents preserving sensible reading order across
    paragraphs, headings, bullet lists, and tables.
    """
    try:
        import docx
        from docx.text.paragraph import Paragraph
        from docx.table import Table
    except ImportError:
        raise DocumentParsingError("python-docx is not installed in the server environment.")

    try:
        doc_stream = io.BytesIO(file_bytes)
        doc = docx.Document(doc_stream)
    except (zipfile.BadZipFile, Exception) as e:
        raise DocumentParsingError(f"Corrupted or invalid DOCX document: {str(e)}")

    extracted_elements: List[Dict[str, Any]] = []
    paragraphs_raw: List[str] = []

    try:
        for child in doc.element.body:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "p":
                p = Paragraph(child, doc)
                text = p.text.strip()
                if text:
                    paragraphs_raw.append(text)
                    extracted_elements.append({
                        "type": "paragraph",
                        "text": text,
                        "source_id": f"P{len(paragraphs_raw)}"
                    })
            elif tag == "tbl":
                tbl = Table(child, doc)
                for r_idx, row in enumerate(tbl.rows):
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    unique_cells = []
                    for c in row_cells:
                        if not unique_cells or c != unique_cells[-1]:
                            unique_cells.append(c)
                    
                    if not unique_cells:
                        continue
                    
                    is_header_row = (r_idx == 0 and len(unique_cells) > 1 and 
                                     all(any(k in c.lower() for k in TABLE_HEADER_KEYWORDS) for c in unique_cells))
                    if is_header_row:
                        continue
                    
                    if len(unique_cells) == 1:
                        cell_text = unique_cells[0]
                    else:
                        content_cells = [c for c in unique_cells if not re.match(r'^\s*#?\d+\s*$', c)]
                        cell_text = " - ".join(content_cells) if content_cells else " - ".join(unique_cells)
                    
                    if cell_text.strip():
                        extracted_elements.append({
                            "type": "table_row",
                            "text": cell_text.strip(),
                            "source_id": f"Table-Row {r_idx + 1}"
                        })
    except Exception as e:
        raise DocumentParsingError(f"Failed while extracting DOCX content: {str(e)}")

    return {
        "elements": extracted_elements,
        "paragraphs": paragraphs_raw,
        "text": "\n\n".join([el["text"] for el in extracted_elements]),
        "page_count": 1,
        "warnings": []
    }


def extract_pdf_content(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts text from PDF documents using pypdf across all pages.
    Detects scanned or image-only PDFs where OCR is required.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocumentParsingError("pypdf is not installed in the server environment.")

    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)
        total_pages = len(reader.pages)
    except Exception as e:
        raise DocumentParsingError(f"Corrupted or invalid PDF document: {str(e)}")

    if total_pages == 0:
        raise DocumentParsingError("The uploaded PDF document contains 0 pages.")

    extracted_elements: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    warnings: List[str] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                full_text_parts.append(page_text)
                # Split page text into non-empty blocks / paragraphs
                blocks = [b.strip() for b in re.split(r'\n\s*\n', page_text) if b.strip()]
                for b_idx, block in enumerate(blocks, start=1):
                    # Filter out lonely page numbers (e.g., '1', 'Page 1')
                    if re.match(r'^(?:page\s*)?\d+\s*$', block, re.IGNORECASE):
                        continue
                    extracted_elements.append({
                        "type": "paragraph",
                        "text": block,
                        "page": page_idx,
                        "source_id": f"Page {page_idx}, Block {b_idx}"
                    })
        except Exception as page_err:
            warnings.append(f"Warning on page {page_idx}: {str(page_err)}")

    full_text = "\n\n".join(full_text_parts).strip()
    alpha_chars = len(re.findall(r'[a-zA-Z0-9]', full_text))

    # Detect scanned or image-only PDF
    if alpha_chars < 15 and total_pages > 0:
        raise DocumentParsingError(
            f"This PDF ({total_pages} page{'s' if total_pages > 1 else ''}) appears to be a scanned or image-only document containing no extractable text. "
            "OCR (Optical Character Recognition) is required to process scanned images. "
            "Please convert this document into a searchable PDF or upload a text-based document."
        )

    return {
        "elements": extracted_elements,
        "paragraphs": full_text_parts,
        "text": full_text,
        "page_count": total_pages,
        "warnings": warnings
    }


def _detect_csv_delimiter(text: str) -> str:
    """
    Detects the most plausible CSV delimiter (',', ';', '\t', '|')
    by checking delimiter frequency consistency across lines.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()][:15]
    if not lines:
        return ','

    candidates = [',', ';', '\t', '|']
    best_delim = ','
    best_score = -1

    for d in candidates:
        counts = []
        for line in lines:
            try:
                parsed = list(csv.reader([line], delimiter=d))
                counts.append(len(parsed[0]) if parsed else 0)
            except Exception:
                pass
        
        if counts and len(set(counts)) == 1 and counts[0] > 1:
            score = counts[0] * 100
            if score > best_score:
                best_score = score
                best_delim = d
        elif counts and min(counts) > 1:
            score = min(counts) * 10
            if score > best_score:
                best_score = score
                best_delim = d

    if best_score > 0:
        return best_delim

    try:
        sample = text[:4096]
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=',;\t|')
        return dialect.delimiter
    except Exception:
        return ','


def _find_best_text_column(
    headers: List[str],
    rows_sample: List[List[Any]],
    target_column_name: Optional[str] = None
) -> Tuple[int, str, List[str]]:
    """
    Identifies the most suitable review text column in tabular data.
    - If target_column_name is specified and matches a header, selects it.
    - If target_column_name does not match, logs a warning and falls back to auto-detection.
    - Ignores ID, rating, date, and metadata columns.
    - Uses text heuristics (average string length, word count, punctuation) to score candidate columns.
    - If multiple plausible columns exist, returns all of them in candidates list while selecting the top one.
    - Returns (column_index, column_name, candidates_list).
    """
    num_cols = len(headers) if headers else (len(rows_sample[0]) if rows_sample else 0)
    if num_cols == 0:
        return 0, "Column 1", ["Column 1"]

    # 1. Target column name explicitly requested
    if target_column_name and str(target_column_name).strip():
        t_raw = str(target_column_name).strip().lower()
        t_norm = re.sub(r'[\s_\-]+', ' ', t_raw)
        
        # Exact match
        for idx, col in enumerate(headers):
            c_raw = str(col or "").strip().lower()
            if t_raw == c_raw:
                return idx, str(headers[idx]), [str(headers[idx])]

        # Normalized match
        for idx, col in enumerate(headers):
            c_norm = re.sub(r'[\s_\-]+', ' ', str(col or "").strip().lower())
            if t_norm == c_norm or t_norm in c_norm or c_norm in t_norm:
                return idx, str(headers[idx]), [str(headers[idx])]

    # 2. Score every column
    col_scores = []
    plausible_candidates = []

    for c_idx in range(num_cols):
        col_header = str(headers[c_idx] if c_idx < len(headers) else f"Column {c_idx+1}").strip()
        col_norm = re.sub(r'[\s_\-]+', ' ', col_header.lower())

        # Check if header explicitly matches metadata keywords
        is_metadata = any(
            meta == col_norm or col_norm.endswith(" " + meta) or col_norm.startswith(meta + " ") or (" " + meta + " ") in (" " + col_norm + " ")
            for meta in METADATA_COLUMN_KEYWORDS
        )

        # Sample row data for this column
        texts = [str(r[c_idx]).strip() for r in rows_sample if c_idx < len(r) and r[c_idx] is not None]
        texts = [t for t in texts if t and t.lower() not in {"nan", "null", "none"}]
        
        avg_len = sum(len(t) for t in texts) / max(len(texts), 1) if texts else 0.0
        avg_words = sum(len(t.split()) for t in texts) / max(len(texts), 1) if texts else 0.0
        
        all_chars = "".join(texts)
        alpha_count = sum(1 for ch in all_chars if ch.isalpha())
        alpha_ratio = alpha_count / max(len(all_chars), 1)

        score = 0.0

        # Header keyword scoring
        header_matched_primary = False
        for kw in PRIMARY_REVIEW_KEYWORDS:
            if kw == col_norm:
                score += 160.0
                header_matched_primary = True
                break
            elif (" " + kw + " ") in (" " + col_norm + " ") or col_norm.startswith(kw + " ") or col_norm.endswith(" " + kw):
                score += 100.0
                header_matched_primary = True
                break
            elif kw in col_norm and not is_metadata:
                score += 60.0
                header_matched_primary = True
                break

        if is_metadata:
            score -= 150.0

        # Data content scoring
        if avg_words >= 4 and avg_len >= 20 and alpha_ratio >= 0.5:
            score += avg_len * 1.5 + avg_words * 5.0
            plausible_candidates.append((score, col_header, c_idx))
        elif avg_words >= 2 and avg_len >= 8 and alpha_ratio >= 0.4:
            score += avg_len * 0.8 + avg_words * 2.0
            if header_matched_primary:
                plausible_candidates.append((score, col_header, c_idx))
        elif alpha_ratio < 0.3:
            score -= 100.0

        col_scores.append((score, avg_len, c_idx, col_header))

    # Sort columns by total score descending
    col_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_c_idx = col_scores[0][2]
    best_name = col_scores[0][3]

    # Plausible candidates list for user
    plausible_candidates.sort(key=lambda x: x[0], reverse=True)
    candidate_names = []
    for _, name, _ in plausible_candidates:
        if name not in candidate_names:
            candidate_names.append(name)

    if not candidate_names:
        candidate_names = [best_name]

    return best_c_idx, best_name, candidate_names


def extract_csv_content(file_bytes: bytes, target_column_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts review records from CSV file bytes, detecting review text column.
    Properly ignores empty rows and excludes column headers from review records.
    """
    text = None
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DocumentParsingError("Unable to decode CSV file with supported encodings (UTF-8, Latin-1).")

    lines = []
    extracted_elements = []
    warnings = []

    try:
        delimiter = _detect_csv_delimiter(text)
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        
        # 1. Filter out completely blank or whitespace-only rows
        raw_rows = []
        for r in reader:
            cleaned_row = [str(c or "").strip() for c in r]
            if any(cleaned_row):
                raw_rows.append(cleaned_row)

        if not raw_rows:
            return {"elements": [], "paragraphs": [], "text": "", "row_count": 0, "warnings": []}

        # 2. Determine if the first row is a header
        first_row = raw_rows[0]
        has_header = False
        
        try:
            sample = text[:4096]
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(sample)
        except Exception:
            has_header = False

        # Keyword match on first row cells
        first_row_normalized = [re.sub(r'[\s_\-]+', ' ', c.lower().strip()) for c in first_row]
        if any(
            any(k in c for k in PRIMARY_REVIEW_KEYWORDS) or c in TABLE_HEADER_KEYWORDS
            for c in first_row_normalized
        ):
            has_header = True

        # Heuristic: if first row cells are short identifier names (< 35 chars, no sentence punctuation)
        # while data rows in the same column are longer sentences with spaces and punctuation
        if not has_header and len(raw_rows) > 1:
            for c_idx in range(len(first_row)):
                cell_val = first_row[c_idx]
                cell_len = len(cell_val)
                sub_lens = [len(r[c_idx]) for r in raw_rows[1:10] if c_idx < len(r)]
                avg_sub_len = sum(sub_lens) / max(len(sub_lens), 1)
                if cell_len < 35 and avg_sub_len > 35 and not any(p in cell_val for p in [".", "!", "?"]):
                    has_header = True
                    break

        headers = [h.strip() for h in first_row] if has_header else [f"Column {i+1}" for i in range(len(first_row))]
        data_rows = raw_rows[1:] if has_header else raw_rows

        review_col_idx, review_col_name, candidates = _find_best_text_column(
            headers, data_rows[:30], target_column_name=target_column_name
        )

        if target_column_name and review_col_name != target_column_name and target_column_name not in headers:
            warnings.append(
                f"Specified column '{target_column_name}' was not found in headers {headers}. "
                f"Automatically selected '{review_col_name}' instead."
            )

        header_names_lower = {h.strip().lower() for h in headers if h}
        header_names_normalized = {re.sub(r'[\s_\-]+', ' ', h.strip().lower()) for h in headers if h}
        known_header_words = {
            "review", "reviews", "review text", "reviews text", "feedback",
            "comment", "comments", "customer review", "customer reviews",
            "text", "content", "rating", "sentiment", "id"
        }

        for r_idx, row in enumerate(data_rows, start=(2 if has_header else 1)):
            if not row:
                continue
            if len(row) > review_col_idx:
                raw_val = row[review_col_idx]
                if raw_val is None:
                    continue
                val = str(raw_val).strip()
                if not val:
                    continue
                val_norm = re.sub(r'[\s_\-]+', ' ', val.lower())
                
                # Exclude any cell that is actually a column header name
                if val_norm in known_header_words or val.strip().lower() in header_names_lower or val_norm in header_names_normalized:
                    continue
                
                lines.append(val)
                extracted_elements.append({
                    "type": "csv_row",
                    "text": val,
                    "source_id": f"Row {r_idx}",
                    "row_number": r_idx
                })

    except Exception as e:
        raise DocumentParsingError(f"CSV format error: {str(e)}")

    if not extracted_elements:
        raise DocumentParsingError(
            f"No readable review records could be identified in the CSV file using column '{review_col_name}'. "
            f"Available columns: {headers}. Please verify review text column."
        )

    return {
        "elements": extracted_elements,
        "paragraphs": lines,
        "text": "\n".join(lines),
        "row_count": len(extracted_elements),
        "review_column_used": review_col_name,
        "column_candidates": candidates,
        "available_columns": headers,
        "warnings": warnings
    }


def extract_xlsx_content(file_bytes: bytes, target_column_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts review records from Excel (.xlsx) workbooks using openpyxl.
    Inspects worksheets and auto-detects review text columns.
    """
    try:
        import openpyxl
    except ImportError:
        raise DocumentParsingError("openpyxl is not installed in the server environment.")

    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    except Exception as e:
        raise DocumentParsingError(f"Corrupted or invalid XLSX workbook: {str(e)}")

    sheet_names = wb.sheetnames
    if not sheet_names:
        raise DocumentParsingError("The XLSX workbook contains no worksheets.")

    ws = wb.active or wb[sheet_names[0]]
    raw_rows = list(ws.iter_rows(values_only=True))

    # Filter out blank rows
    non_empty_rows = []
    for r in raw_rows:
        cleaned = [str(c or "").strip() for c in r]
        if any(cleaned):
            non_empty_rows.append(cleaned)

    if not non_empty_rows:
        raise DocumentParsingError(f"Worksheet '{ws.title}' in the XLSX workbook is empty.")

    first_row = non_empty_rows[0]
    first_row_normalized = [c.lower().replace("_", " ").replace("-", " ").strip() for c in first_row]
    has_header = any(
        any(k in c for k in REVIEW_COLUMN_KEYWORDS) or c in TABLE_HEADER_KEYWORDS
        for c in first_row_normalized
    )

    headers = [h.strip() for h in first_row] if has_header else [f"Column {i+1}" for i in range(len(first_row))]
    data_rows = non_empty_rows[1:] if has_header else non_empty_rows

    review_col_idx, review_col_name, candidates = _find_best_text_column(
        headers, data_rows[:20], target_column_name=target_column_name
    )

    header_names_lower = {h.strip().lower() for h in headers if h}
    header_names_normalized = {h.strip().lower().replace("_", " ").replace("-", " ") for h in headers if h}
    known_header_words = {
        "review", "reviews", "review text", "reviews text", "feedback",
        "comment", "comments", "customer review", "customer reviews",
        "text", "content", "rating", "sentiment", "id"
    }

    lines = []
    extracted_elements = []

    for r_idx, row in enumerate(data_rows, start=(2 if has_header else 1)):
        if not row:
            continue
        if len(row) > review_col_idx and row[review_col_idx] is not None:
            val = str(row[review_col_idx]).strip()
            val_norm = val.lower().replace("_", " ").replace("-", " ").strip()
            if not val or val_norm in known_header_words or val.strip().lower() in header_names_lower or val_norm in header_names_normalized:
                continue

            lines.append(val)
            extracted_elements.append({
                "type": "xlsx_row",
                "text": val,
                "source_id": f"Row {r_idx}",
                "row_number": r_idx,
                "sheet": ws.title
            })

    if not extracted_elements:
        raise DocumentParsingError(f"No readable review records found in worksheet '{ws.title}'.")

    return {
        "elements": extracted_elements,
        "paragraphs": lines,
        "text": "\n".join(lines),
        "worksheet_count": len(sheet_names),
        "worksheet_names": sheet_names,
        "active_worksheet": ws.title,
        "row_count": len(extracted_elements),
        "review_column_used": review_col_name,
        "column_candidates": candidates,
        "warnings": []
    }


def extract_xls_content(file_bytes: bytes, target_column_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts review records from legacy Excel (.xls) workbooks using xlrd.
    """
    try:
        import xlrd
    except ImportError:
        raise DocumentParsingError("xlrd is not installed in the server environment.")

    try:
        wb = xlrd.open_workbook(file_contents=file_bytes)
    except Exception as e:
        raise DocumentParsingError(f"Corrupted or invalid legacy XLS workbook: {str(e)}")

    sheet_names = wb.sheet_names()
    if not sheet_names:
        raise DocumentParsingError("The XLS workbook contains no worksheets.")

    sheet = wb.sheet_by_index(0)
    if sheet.nrows == 0:
        raise DocumentParsingError(f"Worksheet '{sheet.name}' in the XLS workbook is empty.")

    first_row = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]
    first_row_normalized = [c.lower().replace("_", " ").replace("-", " ").strip() for c in first_row]
    has_header = any(
        any(k in c for k in REVIEW_COLUMN_KEYWORDS) or c in TABLE_HEADER_KEYWORDS
        for c in first_row_normalized
    )

    headers = [h.strip() for h in first_row] if has_header else [f"Column {i+1}" for i in range(sheet.ncols)]
    start_row = 1 if has_header else 0

    sample_rows = []
    for r in range(start_row, min(start_row + 20, sheet.nrows)):
        sample_rows.append([sheet.cell_value(r, c) for c in range(sheet.ncols)])

    review_col_idx, review_col_name, candidates = _find_best_text_column(
        headers, sample_rows, target_column_name=target_column_name
    )

    header_names_lower = {h.strip().lower() for h in headers if h}
    header_names_normalized = {h.strip().lower().replace("_", " ").replace("-", " ") for h in headers if h}
    known_header_words = {
        "review", "reviews", "review text", "reviews text", "feedback",
        "comment", "comments", "customer review", "customer reviews",
        "text", "content", "rating", "sentiment", "id"
    }

    lines = []
    extracted_elements = []

    for r_idx in range(start_row, sheet.nrows):
        cell_val = sheet.cell_value(r_idx, review_col_idx) if review_col_idx < sheet.ncols else ""
        val = str(cell_val).strip()
        val_norm = val.lower().replace("_", " ").replace("-", " ").strip()
        if not val or val_norm in known_header_words or val.strip().lower() in header_names_lower or val_norm in header_names_normalized:
            continue

        lines.append(val)
        extracted_elements.append({
            "type": "xls_row",
            "text": val,
            "source_id": f"Row {r_idx + 1}",
            "row_number": r_idx + 1,
            "sheet": sheet.name
        })

    if not extracted_elements:
        raise DocumentParsingError(f"No readable review records found in worksheet '{sheet.name}'.")

    return {
        "elements": extracted_elements,
        "paragraphs": lines,
        "text": "\n".join(lines),
        "worksheet_count": len(sheet_names),
        "worksheet_names": sheet_names,
        "active_worksheet": sheet.name,
        "row_count": len(extracted_elements),
        "review_column_used": review_col_name,
        "column_candidates": candidates,
        "warnings": []
    }


def extract_doc_legacy_content(file_bytes: bytes) -> Dict[str, Any]:
    """
    Handles legacy Word (.doc) binary files.
    If LibreOffice is installed, converts to .docx.
    Otherwise attempts binary stream extraction or guides the user to modern .docx.
    """
    # 1. Check if LibreOffice is available on the host system
    soffice_path = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice_path:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "legacy_input.doc")
            with open(input_path, "wb") as f:
                f.write(file_bytes)
            try:
                cmd = [soffice_path, "--headless", "--convert-to", "docx", "--outdir", tmpdir, input_path]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                converted_path = os.path.join(tmpdir, "legacy_input.docx")
                if os.path.exists(converted_path):
                    with open(converted_path, "rb") as cf:
                        return extract_docx_content(cf.read())
            except Exception as conv_err:
                logger.warning(f"LibreOffice conversion failed: {conv_err}")

    # 2. Pure Python text stream fallback for OLE2 binary .doc files
    # Word 97-2004 files contain UTF-16LE or ASCII text runs in their stream
    text_chunks = []
    # Search for printable ASCII runs >= 15 chars
    ascii_runs = re.findall(b'[\x20-\x7E\r\n]{15,}', file_bytes)
    for r in ascii_runs:
        decoded = r.decode('latin-1', errors='ignore').strip()
        # Filter out common binary OLE metadata
        if not any(meta in decoded for meta in ["Root Entry", "WordDocument", "SummaryInformation", "CompObj", "Microsoft Word"]):
            if len(decoded.split()) >= 3:
                text_chunks.append(decoded)

    # Search for UTF-16LE runs >= 15 chars
    utf16_runs = re.findall(b'(?:[\x20-\x7E]\x00){15,}', file_bytes)
    for r in utf16_runs:
        try:
            decoded = r.decode('utf-16le', errors='ignore').strip()
            if len(decoded.split()) >= 3 and not any(meta in decoded for meta in ["Normal.dotm", "Heading", "Default Paragraph"]):
                text_chunks.append(decoded)
        except Exception:
            pass

    # Deduplicate while preserving order
    unique_chunks = list(dict.fromkeys(text_chunks))
    combined_text = "\n\n".join(unique_chunks).strip()

    if len(unique_chunks) >= 1 and len(combined_text.split()) >= 8:
        elements = [{
            "type": "paragraph",
            "text": chunk,
            "source_id": f"P{i+1}"
        } for i, chunk in enumerate(unique_chunks)]
        return {
            "elements": elements,
            "paragraphs": unique_chunks,
            "text": combined_text,
            "warnings": [
                "Extracted using legacy fallback binary parser. For full structural fidelity (tables and formatting), save or export your document as modern Word (.docx)."
            ]
        }

    # If extraction is empty or insufficient, provide clear conversion instructions
    raise DocumentParsingError(
        "Legacy Word (.doc) format conversion software is unavailable on the server and the binary file contains no extractable plain text. "
        "Please open this document in Microsoft Word, LibreOffice, or Google Docs and save it as modern Word (.docx) or plain text (.txt)."
    )


def extract_txt_content(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts text from plain text files with robust encoding fallback.
    """
    text = None
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
            
    if text is None:
        raise DocumentParsingError("Unable to decode text file. Please ensure it is saved in UTF-8 format.")

    raw_blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if not raw_blocks:
        raw_blocks = [line.strip() for line in text.splitlines() if line.strip()]

    elements = [{
        "type": "text_block",
        "text": b,
        "source_id": f"Block {i+1}"
    } for i, b in enumerate(raw_blocks)]

    return {
        "elements": elements,
        "paragraphs": raw_blocks,
        "text": text.strip(),
        "warnings": []
    }


def validate_and_extract_document(
    filename: str,
    file_bytes: bytes,
    max_file_size: int = MAX_FILE_SIZE_BYTES,
    column_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates uploaded document, extracts text content, and returns extraction diagnostics
    for all supported formats: .csv, .txt, .pdf, .docx, .doc, .xlsx, .xls.
    """
    file_size = len(file_bytes)
    if file_size == 0:
        raise DocumentParsingError("Uploaded file is completely empty (0 bytes).")

    if file_size > max_file_size:
        max_mb = max_file_size / (1024 * 1024)
        actual_mb = round(file_size / (1024 * 1024), 2)
        raise DocumentParsingError(
            f"File size ({actual_mb} MB) exceeds maximum allowed size limit ({max_mb} MB)."
        )

    _, ext = os.path.splitext(filename or "")
    ext_lower = ext.lower()

    if not ext_lower:
        raise DocumentParsingError("Uploaded file has no file extension. Supported formats: .csv, .txt, .pdf, .docx, .doc, .xlsx, .xls")

    if ext_lower not in SUPPORTED_EXTENSIONS:
        raise DocumentParsingError(
            f"Unsupported file format '{ext_lower}'. Supported formats are: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    # Route extraction by extension
    if ext_lower == ".docx":
        extracted = extract_docx_content(file_bytes)
    elif ext_lower == ".doc":
        extracted = extract_doc_legacy_content(file_bytes)
    elif ext_lower == ".pdf":
        extracted = extract_pdf_content(file_bytes)
    elif ext_lower == ".csv":
        extracted = extract_csv_content(file_bytes, target_column_name=column_name)
    elif ext_lower == ".xlsx":
        extracted = extract_xlsx_content(file_bytes, target_column_name=column_name)
    elif ext_lower == ".xls":
        extracted = extract_xls_content(file_bytes, target_column_name=column_name)
    elif ext_lower == ".txt":
        extracted = extract_txt_content(file_bytes)
    else:
        raise DocumentParsingError(f"Unsupported file format '{ext_lower}'.")

    full_text = extracted.get("text", "").strip()
    words = full_text.split()
    char_count = len(full_text)
    word_count = len(words)

    # Verify extracted text is sufficient
    if char_count < 3 or word_count == 0 or not re.search(r'[a-zA-Z0-9]', full_text):
        raise DocumentParsingError(
            "The document contains empty or insufficient text for sentiment analysis. "
            "Please ensure the document contains readable review content."
        )

    preview = full_text[:160].replace("\n", " ").strip()
    if len(full_text) > 160:
        preview += "..."

    # Build format-specific metrics
    pages = extracted.get("page_count")
    sheets = extracted.get("worksheet_count")
    rows = extracted.get("row_count")
    col_used = extracted.get("review_column_used")
    warnings = extracted.get("warnings", [])

    diagnostics = {
        "filename": filename,
        "file_extension": ext_lower,
        "file_size_bytes": file_size,
        "file_size_formatted": f"{round(file_size / 1024, 1)} KB" if file_size < 1024*1024 else f"{round(file_size / (1024*1024), 2)} MB",
        "character_count": char_count,
        "word_count": word_count,
        "preview": preview,
        "raw_elements_count": len(extracted.get("elements", [])),
        "page_count": pages,
        "worksheet_count": sheets,
        "row_count": rows,
        "review_column_used": col_used,
        "column_candidates": extracted.get("column_candidates", []),
        "available_columns": extracted.get("available_columns", []),
        "warnings": warnings
    }

    # Generate user-facing summary string (e.g. "File processed successfully: reviews.pdf | 4 pages | 3,120 chars | 12 reviews identified.")
    unit_info = []
    if pages:
        unit_info.append(f"{pages} page{'s' if pages > 1 else ''}")
    elif sheets:
        unit_info.append(f"{sheets} sheet{'s' if sheets > 1 else ''}")
    if rows:
        unit_info.append(f"{rows} data row{'s' if rows > 1 else ''}")
    elif len(extracted.get("paragraphs", [])):
        unit_info.append(f"{len(extracted['paragraphs'])} paragraph{'s' if len(extracted['paragraphs']) > 1 else ''}")

    unit_str = " | ".join(unit_info) + " | " if unit_info else ""
    diagnostics["summary_text"] = f"File processed successfully: {filename} | {unit_str}{char_count:,} characters extracted"

    logger.info(
        f"Document Extracted: file='{filename}' ({ext_lower}), size={file_size}B, "
        f"chars={char_count}, words={word_count}, elements={len(extracted.get('elements', []))}"
    )

    return {
        "extracted_text": full_text,
        "elements": extracted.get("elements", []),
        "diagnostics": diagnostics,
        "metadata": {
            "total_pages": pages,
            "page_count": pages,
            "worksheet_count": sheets,
            "row_count": rows
        }
    }


def split_into_reviews(
    extracted_data: Dict[str, Any],
    mode: str = "auto",
    max_reviews: int = MAX_REVIEWS_LIMIT,
    column_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Intelligently splits extracted document content into individual review items.
    Returns a list of dictionaries: [{"id": "Row 2", "text": "review content..."}, ...]
    
    Modes:
      - 'auto': Automatically decides boundary based on tabular rows, numbered items, or paragraphs.
      - 'single': Treats the entire document as a single review.
      - 'multiple': Forces splitting into separate reviews based on delimiters/blocks.
      - 'paragraph': Forces each non-empty paragraph to be an individual review.
    """
    elements = extracted_data.get("elements", [])
    full_text = (extracted_data.get("extracted_text") or extracted_data.get("text") or "").strip()

    if not full_text:
        return []

    known_header_words = {
        "review", "reviews", "review text", "reviews text", "feedback",
        "comment", "comments", "customer review", "customer reviews",
        "text", "content", "rating", "sentiment", "id"
    }

    # Mode: Explicit Single Review
    if mode == "single":
        clean_t = _clean_review_prefix(full_text)
        return [{"id": "Document 1", "text": clean_t}] if len(clean_t) > 2 else []

    # Mode: Explicit Each Paragraph = One Review
    if mode == "paragraph":
        reviews = []
        for i, el in enumerate(elements, start=1):
            t = _clean_review_prefix(el.get("text", ""))
            t_norm = t.lower().replace("_", " ").replace("-", " ").strip()
            if t_norm in known_header_words:
                continue
            if len(t) > 2:
                reviews.append({
                    "id": el.get("source_id", f"Paragraph {len(reviews) + 1}"),
                    "text": t
                })
        return reviews[:max_reviews]

    # Check tabular data (CSV / XLSX / XLS / DOCX Table rows)
    is_tabular = any(el.get("type") in ("csv_row", "xlsx_row", "xls_row", "table_row") for el in elements)

    # If tabular data or mode == "multiple":
    if mode in ["multiple", "auto"] and is_tabular:
        reviews = []
        for i, el in enumerate(elements, start=1):
            t = _clean_review_prefix(el.get("text", ""))
            t_norm = re.sub(r'[\s_\-]+', ' ', t.lower().strip())
            if t_norm in known_header_words:
                continue
            if len(t) >= 2:
                reviews.append({
                    "id": el.get("source_id", f"Record {len(reviews) + 1}"),
                    "text": t
                })
        if len(reviews) >= 1 or mode == "multiple":
            return reviews[:max_reviews]

    # Check for numbered review items (e.g. '1. ', '2. ', 'Review 1:')
    numbered_matches = 0
    bullet_matches = 0
    for el in elements:
        t = el.get("text", "").strip()
        if NUMBERED_PATTERN.match(t):
            numbered_matches += 1
        elif BULLET_PATTERN.match(t):
            bullet_matches += 1

    if numbered_matches >= 2 or bullet_matches >= 2:
        raw_res = _split_elements_by_markers_or_boundaries(elements, full_text)[:max_reviews]
        return [r for r in raw_res if r.get("text", "").lower().replace("_", " ").strip() not in known_header_words]

    # Multi-paragraph documents
    if len(elements) >= 2:
        avg_words_per_para = sum(len(el.get("text", "").split()) for el in elements) / len(elements)
        if len(elements) >= 3 or avg_words_per_para < 60:
            reviews = []
            for i, el in enumerate(elements, start=1):
                clean_t = _clean_review_prefix(el.get("text", ""))
                t_norm = clean_t.lower().replace("_", " ").replace("-", " ").strip()
                if t_norm in known_header_words:
                    continue
                if len(clean_t) > 3:
                    reviews.append({
                        "id": el.get("source_id", f"Review {len(reviews) + 1}"),
                        "text": clean_t
                    })
            return reviews[:max_reviews]

    # Fallback to single review if text is continuous narrative
    clean_single = _clean_review_prefix(full_text)
    if clean_single.lower().replace("_", " ").strip() in known_header_words:
        return []
    return [{"id": "Review 1", "text": clean_single}] if len(clean_single) > 2 else []


def _split_elements_by_markers_or_boundaries(
    elements: List[Dict[str, Any]],
    full_text: str
) -> List[Dict[str, Any]]:
    """
    Helper to split document elements containing numbered reviews, bullets,
    table rows, or paragraph boundaries.
    """
    reviews: List[Dict[str, Any]] = []
    current_text_parts: List[str] = []
    current_id: Optional[str] = None
    item_counter = 1

    for el in elements:
        it = el.get("text", "").strip()
        el_type = el.get("type", "paragraph")
        source_id = el.get("source_id", f"Review {item_counter}")
        if not it:
            continue

        if SECTION_DIVIDER_PATTERN.match(it):
            if current_text_parts:
                reviews.append({
                    "id": current_id or f"Review {item_counter}",
                    "text": " ".join(current_text_parts).strip()
                })
                item_counter += 1
                current_text_parts = []
                current_id = None
            continue

        is_tabular = el_type in ("csv_row", "xlsx_row", "xls_row", "table_row")
        is_numbered_or_bullet = bool(NUMBERED_PATTERN.match(it) or BULLET_PATTERN.match(it))

        if is_tabular or is_numbered_or_bullet:
            if current_text_parts:
                reviews.append({
                    "id": current_id or f"Review {item_counter}",
                    "text": " ".join(current_text_parts).strip()
                })
                item_counter += 1
                current_text_parts = []
            current_id = source_id
            current_text_parts.append(_clean_review_prefix(it))
        else:
            if current_text_parts:
                current_text_parts.append(it)
            else:
                current_id = source_id
                current_text_parts.append(_clean_review_prefix(it))

    if current_text_parts:
        reviews.append({
            "id": current_id or f"Review {item_counter}",
            "text": " ".join(current_text_parts).strip()
        })

    # Regex fallback if traversal returned <= 1
    if len(reviews) <= 1:
        splits = re.split(r'(?m)(?=^\s*(?:(?:review|item|customer|#)?\s*\d+[\.\:\)\-\]]|\[\s*\d+\s*\])\s+)', full_text)
        splits = [s.strip() for s in splits if s.strip()]
        if len(splits) > 1:
            return [{"id": f"Review {i+1}", "text": _clean_review_prefix(s)} for i, s in enumerate(splits) if len(_clean_review_prefix(s)) > 2]

        splits = re.split(r'(?m)(?=^\s*[\u2022\u2023\u25E6\u2043\u2219\*\-\+▪▫►■]\s+)', full_text)
        splits = [s.strip() for s in splits if s.strip()]
        if len(splits) > 1:
            return [{"id": f"Review {i+1}", "text": _clean_review_prefix(s)} for i, s in enumerate(splits) if len(_clean_review_prefix(s)) > 2]

        para_splits = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
        if len(para_splits) > 1:
            return [{"id": f"Review {i+1}", "text": _clean_review_prefix(p)} for i, p in enumerate(para_splits) if len(_clean_review_prefix(p)) > 2]

    return [r for r in reviews if len(r["text"].strip()) > 2]


def _clean_review_prefix(text: str) -> str:
    """
    Cleans leading numbers, bullet markers, or label prefixes from review text.
    E.g. '1. The product quality is excellent.' -> 'The product quality is excellent.'
    """
    s = text.strip()
    s = NUMBERED_PATTERN.sub('', s)
    s = BULLET_PATTERN.sub('', s)
    s = re.sub(r'^(?:Review|Customer Feedback|Feedback|Comment)\s*:\s*', '', s, flags=re.IGNORECASE)
    return s.strip()
