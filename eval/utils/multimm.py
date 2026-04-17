import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile


SPREADSHEET_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CSV_ENCODING = "utf-8-sig"


def _column_letters_to_index(reference: str) -> int:
    letters = []
    for char in reference:
        if char.isalpha():
            letters.append(char.upper())
        else:
            break

    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _load_shared_strings(zip_file: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []

    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    shared_strings: list[str] = []
    for string_item in root.findall("a:si", SPREADSHEET_NS):
        shared_strings.append("".join(node.text or "" for node in string_item.findall(".//a:t", SPREADSHEET_NS)))
    return shared_strings


def _extract_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", SPREADSHEET_NS))

    value_node = cell.find("a:v", SPREADSHEET_NS)
    if value_node is None or value_node.text is None:
        return ""

    if cell_type == "s":
        return shared_strings[int(value_node.text)]
    return value_node.text


def convert_xlsx_to_csv(xlsx_path: Path) -> Path:
    xlsx_path = Path(xlsx_path)
    csv_path = xlsx_path.with_suffix(".csv")
    if csv_path.exists() and csv_path.stat().st_mtime >= xlsx_path.stat().st_mtime:
        return csv_path

    with ZipFile(xlsx_path) as zip_file:
        shared_strings = _load_shared_strings(zip_file)
        sheet_root = ET.fromstring(zip_file.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet_root.findall(".//a:sheetData/a:row", SPREADSHEET_NS):
        values_by_column: dict[int, str] = {}
        max_column_index = -1
        for cell in row.findall("a:c", SPREADSHEET_NS):
            column_index = _column_letters_to_index(cell.attrib["r"])
            values_by_column[column_index] = _extract_cell_value(cell, shared_strings)
            max_column_index = max(max_column_index, column_index)

        if max_column_index < 0:
            rows.append([])
            continue

        rows.append([values_by_column.get(index, "") for index in range(max_column_index + 1)])

    with csv_path.open("w", newline="", encoding=CSV_ENCODING) as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    return csv_path


def ensure_multimm_csvs(data_root: Path) -> dict[str, Path]:
    data_root = Path(data_root)
    paths: dict[str, Path] = {}
    for language in ("CN", "EN"):
        candidates = [
            data_root / f"{language}.xlsx",
            data_root / "all" / f"{language}.xlsx",
            data_root / "data" / f"{language}.xlsx",
            data_root / "data" / "all" / f"{language}.xlsx",
        ]
        xlsx_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if xlsx_path is None:
            raise FileNotFoundError(f"Missing MultiMM excel file under {data_root} for language {language}")
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Missing MultiMM excel file: {xlsx_path}")
        paths[language] = convert_xlsx_to_csv(xlsx_path)
    return paths


def read_multimm_csv(csv_path: Path) -> List[Dict[str, str]]:
    with Path(csv_path).open("r", encoding=CSV_ENCODING, newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return []

    header = list(rows[0])
    max_width = max(len(row) for row in rows)
    if len(header) < max_width:
        header.extend([f"extra_{index}" for index in range(len(header), max_width)])

    normalized_rows: List[Dict[str, str]] = []
    for row in rows[1:]:
        padded_row = row + [""] * (len(header) - len(row))
        normalized_rows.append(dict(zip(header, padded_row)))
    return normalized_rows


def resolve_multimm_columns(rows: List[Dict[str, str]]) -> Dict[str, str]:
    if not rows:
        raise ValueError("MultiMM csv is empty")

    available_columns = list(rows[0].keys())

    def pick(*candidates: str) -> str:
        for candidate in candidates:
            if candidate in available_columns:
                return candidate
        raise KeyError(f"Missing expected MultiMM column. Available columns: {available_columns}")

    metaphor_column = "MetaphorOccurrence"
    if metaphor_column not in available_columns:
        unnamed_candidates = [column for column in available_columns if column == "" or column.startswith("extra_")]
        if unnamed_candidates:
            metaphor_column = unnamed_candidates[0]
        else:
            raise KeyError(f"Missing metaphor column. Available columns: {available_columns}")

    return {
        "image": pick("Pic_id"),
        "text": pick("Text"),
        "metaphor": metaphor_column,
        "target": pick("Target"),
        "source": pick("Source"),
        "emotion": pick("SentimentCategory"),
    }


def get_image_dir(data_root: Path, language: str, image_root: Optional[Path] = None) -> Path:
    if image_root is not None:
        image_dir = Path(image_root) / f"imgs_{language}"
    elif (Path(data_root) / f"imgs_{language}").exists():
        image_dir = Path(data_root) / f"imgs_{language}"
    else:
        image_dir = Path(data_root) / "data" / f"imgs_{language}"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing MultiMM image directory: {image_dir}")
    return image_dir
