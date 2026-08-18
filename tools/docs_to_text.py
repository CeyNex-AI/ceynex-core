"""Convert the governing documents under docs/ to plain text for reading.

The binary documents (SRS, SAD, Feasibility Report, proposals, schedule) are the
project's normative sources; every module docstring in this repo names the SRS
section it implements, so they need to be greppable.

Rule, deliberate: when a .docx and a .pdf of the same document sit side by side,
the .docx wins. PDF extraction reflows columns, mangles table cells and drops
list structure; python-docx walks the real document tree, so paragraphs and
tables come out in document order with cell boundaries intact.

    python tools/docs_to_text.py                 # docs/ -> docs/_text/
    python tools/docs_to_text.py --docs-dir X --out-dir Y
    python tools/docs_to_text.py --stdout SRS/Software_Requirements_Specification.docx

Extractors are chosen per suffix and each degrades to the next available backend
rather than failing the whole run:

    .docx  python-docx
    .pdf   pdfplumber -> pypdf -> pdftotext(1)
    .xlsx  openpyxl
    .html  stdlib regex strip
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Documents whose text we never want: images, archives, anything binary that is
# not a document. Everything else with a known suffix is converted.
TEXT_SUFFIXES = {".docx", ".pdf", ".xlsx", ".html", ".htm", ".md", ".txt"}

# A .pdf is skipped when a sibling with one of these suffixes exists.
PREFERRED_OVER_PDF = (".docx", ".doc")


class ExtractionError(RuntimeError):
    """No backend could read the file."""


def extract_docx(path: Path) -> str:
    """Paragraphs and tables in document order.

    python-docx exposes `.paragraphs` and `.tables` as two flat lists, which
    loses their interleaving — a table between sections 3.1 and 3.2 would land
    at the end of the file. Walking `body` children keeps the reading order.
    """
    import docx  # python-docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    out: list[str] = []

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            text = Paragraph(child, document).text.strip()
            if text:
                out.append(text)
        elif child.tag == qn("w:tbl"):
            out.append("[TABLE]")
            for row in Table(child, document).rows:
                cells = (c.text.strip().replace("\n", " ") for c in row.cells)
                out.append(" | ".join(cells))
            out.append("[/TABLE]")

    return "\n".join(out)


def extract_pdf(path: Path) -> str:
    """pdfplumber, then pypdf, then the pdftotext binary."""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                pages.append(f"[PAGE {i}]\n{page.extract_text() or ''}")
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [
            f"[PAGE {i}]\n{page.extract_text() or ''}"
            for i, page in enumerate(reader.pages, start=1)
        ]
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        done = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ExtractionError(
            f"no PDF backend for {path.name} — pip install pdfplumber, or apt install poppler-utils"
        ) from exc


def extract_xlsx(path: Path) -> str:
    """One block per sheet, tab-separated, blank trailing cells trimmed."""
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    out: list[str] = []

    for sheet in workbook.worksheets:
        out.append(f"[SHEET {sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if v is None else str(v).strip() for v in row]
            while cells and not cells[-1]:
                cells.pop()
            if cells:
                out.append("\t".join(cells))
        out.append("")

    workbook.close()
    return "\n".join(out)


def extract_html(path: Path) -> str:
    """Strip script/style, then tags. Enough for a Gantt export; not a parser."""
    import html as html_module

    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"<(script|style)\b.*?</\1>", "", raw, flags=re.S | re.I)
    text = html_module.unescape(re.sub(r"<[^>]+>", "\n", raw))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def extract_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


EXTRACTORS = {
    ".docx": extract_docx,
    ".pdf": extract_pdf,
    ".xlsx": extract_xlsx,
    ".html": extract_html,
    ".htm": extract_html,
    ".md": extract_plain,
    ".txt": extract_plain,
}


def extract(path: Path) -> str:
    extractor = EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        raise ExtractionError(f"no extractor for {path.suffix}")
    return extractor(path)


def superseded_by_word(pdf: Path) -> Path | None:
    """The .docx/.doc sibling that should be converted instead of this PDF."""
    for suffix in PREFERRED_OVER_PDF:
        sibling = pdf.with_suffix(suffix)
        if sibling.exists():
            return sibling
    return None


def discover(docs_dir: Path, out_dir: Path) -> list[Path]:
    """Convertible files under docs_dir, PDFs with a Word sibling removed."""
    found = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if out_dir in path.parents:  # never re-convert our own output
            continue
        if path.suffix.lower() == ".pdf" and superseded_by_word(path) is not None:
            continue
        found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs-dir", type=Path, default=repo_root.parent / "docs")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--stdout",
        type=Path,
        default=None,
        metavar="FILE",
        help="convert one file (relative to --docs-dir, or absolute) and print it",
    )
    args = parser.parse_args(argv)

    docs_dir = args.docs_dir.resolve()
    if not docs_dir.is_dir():
        print(f"docs dir not found: {docs_dir}", file=sys.stderr)
        return 2

    if args.stdout is not None:
        target = args.stdout if args.stdout.is_absolute() else docs_dir / args.stdout
        if not target.exists():
            print(f"not found: {target}", file=sys.stderr)
            return 2
        word = superseded_by_word(target) if target.suffix.lower() == ".pdf" else None
        if word is not None:
            print(f"# reading {word.name} instead of the PDF", file=sys.stderr)
            target = word
        print(extract(target))
        return 0

    out_dir = (args.out_dir or docs_dir / "_text").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = discover(docs_dir, out_dir)
    if not targets:
        print(f"nothing to convert under {docs_dir}", file=sys.stderr)
        return 1

    failures = 0
    for path in targets:
        relative = path.relative_to(docs_dir)
        destination = out_dir / relative.with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = extract(path)
        except (ExtractionError, ImportError) as exc:
            print(f"FAIL {relative}: {exc}", file=sys.stderr)
            failures += 1
            continue
        destination.write_text(text, encoding="utf-8")
        print(f"  {relative}  ->  {destination.relative_to(out_dir.parent)}  ({len(text):,} chars)")

    skipped = [p for p in sorted(docs_dir.rglob("*.pdf")) if superseded_by_word(p)]
    for pdf in skipped:
        print(f"  skipped {pdf.relative_to(docs_dir)} (Word original converted instead)")

    print(f"\n{len(targets) - failures} converted, {failures} failed -> {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
