"""
Section-aware PDF parsing for SEC filings and general financial documents.

Extracts per-page text via pymupdf (fitz), detects 10-K/10-Q section
boundaries (Item headers), strips boilerplate, and assembles selected
sections into a context string for LLM generation.

For non-structured documents (earnings transcripts, broker notes),
falls back to page-range selection.
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import fitz  # pymupdf


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A detected section (Item / Part) in a structured SEC filing."""
    title: str              # e.g. "ITEM 7. Management's Discussion and Analysis..."
    item_number: str        # e.g. "7", "1A", "II"
    start_page: int         # 0-indexed
    end_page: int           # 0-indexed, inclusive
    start_char: int         # char offset in concatenated full text
    end_char: int           # char offset in concatenated full text

    @property
    def char_count(self) -> int:
        return self.end_char - self.start_char

    @property
    def page_range_display(self) -> str:
        """Human-readable page range (1-indexed)."""
        if self.start_page == self.end_page:
            return str(self.start_page + 1)
        return f"{self.start_page + 1}\u2013{self.end_page + 1}"


@dataclass
class ParsedDocument:
    """Result of parsing a PDF file."""
    filename: str
    total_pages: int
    total_chars: int
    sections: list[Section] = field(default_factory=list)
    page_texts: list[str] = field(default_factory=list)
    boilerplate_chars_removed: int = 0
    whitespace_chars_removed: int = 0

    @property
    def has_sections(self) -> bool:
        return len(self.sections) > 0

    @property
    def total_chars_removed(self) -> int:
        return self.boilerplate_chars_removed + self.whitespace_chars_removed


# ---------------------------------------------------------------------------
# Item / Part regex patterns
# ---------------------------------------------------------------------------

# Matches "ITEM 7." or "ITEM 1A." etc. at the start of a line
_ITEM_RE = re.compile(
    r"^\s*ITEM\s+(\d+[A-Z]?)\.?\s+(.+)",
    re.IGNORECASE | re.MULTILINE,
)

# Matches "PART I", "PART II", etc.
_PART_RE = re.compile(
    r"^\s*PART\s+(I{1,4}|IV)\b\s*(.*)",
    re.IGNORECASE | re.MULTILINE,
)

# Maximum character offset on a page where a header must appear
# to be considered a real section header (not a body-text reference).
_HEADER_POSITION_LIMIT = 500


# ---------------------------------------------------------------------------
# Section detection — TOC-based
# ---------------------------------------------------------------------------

def _detect_sections_from_toc(doc: fitz.Document, page_texts: list[str]) -> list[Section]:
    """
    Try to extract sections from the PDF's embedded table of contents.
    Many 10-Ks filed via EDGAR have TOC bookmarks.
    """
    toc = doc.get_toc()  # [[level, title, page_num], ...]
    if not toc:
        return []

    # Filter TOC entries that look like Item or Part headers
    relevant = []
    for level, title, page_num in toc:
        page_idx = page_num - 1  # fitz TOC uses 1-indexed pages
        if page_idx < 0 or page_idx >= len(page_texts):
            continue

        item_match = re.match(r"ITEM\s+(\d+[A-Z]?)", title, re.IGNORECASE)
        part_match = re.match(r"PART\s+(I{1,4}|IV)", title, re.IGNORECASE)

        if item_match:
            relevant.append((item_match.group(1).upper(), title.strip(), page_idx))
        elif part_match:
            relevant.append((part_match.group(1).upper(), title.strip(), page_idx))

    if len(relevant) < 2:
        return []

    return _build_sections(relevant, page_texts)


# ---------------------------------------------------------------------------
# Section detection — text-based (regex fallback)
# ---------------------------------------------------------------------------

def _detect_sections_from_text(page_texts: list[str]) -> list[Section]:
    """
    Scan extracted page text for Item/Part headers using regex.
    Only matches headers appearing in the first N chars of a page
    to avoid false positives from body-text references.

    When Item-level sections are detected, PART headers are dropped
    since they are structural dividers that overlap with Items.
    """
    seen_items: set[str] = set()
    item_entries: list[tuple[str, str, int]] = []   # Item-level
    part_entries: list[tuple[str, str, int]] = []    # Part-level

    for page_idx, page_text in enumerate(page_texts):
        # Only search the top portion of the page
        search_region = page_text[:_HEADER_POSITION_LIMIT]

        for match in _ITEM_RE.finditer(search_region):
            item_num = match.group(1).upper()
            if item_num in seen_items:
                continue  # Deduplicate pagination headers
            seen_items.add(item_num)
            full_title = match.group(0).strip()
            item_entries.append((item_num, full_title, page_idx))

        for match in _PART_RE.finditer(search_region):
            part_num = match.group(1).upper()
            part_key = f"PART_{part_num}"
            if part_key in seen_items:
                continue
            seen_items.add(part_key)
            full_title = match.group(0).strip()
            part_entries.append((part_num, full_title, page_idx))

    # Prefer Item-level sections; only fall back to Part headers
    # when no Items are detected (e.g., a proxy statement).
    if len(item_entries) >= 2:
        relevant = item_entries
    elif item_entries or part_entries:
        relevant = sorted(item_entries + part_entries, key=lambda x: x[2])
    else:
        return []

    if len(relevant) < 2:
        return []

    return _build_sections(relevant, page_texts)


# ---------------------------------------------------------------------------
# Shared: build Section objects from detected headers
# ---------------------------------------------------------------------------

def _build_sections(
    relevant: list[tuple[str, str, int]],
    page_texts: list[str],
) -> list[Section]:
    """
    Convert a list of (item_number, title, start_page) into Section objects
    with computed end_page and char offsets.
    """
    # Pre-compute cumulative char offsets per page
    cum_chars = [0]
    for pt in page_texts:
        cum_chars.append(cum_chars[-1] + len(pt))

    total_pages = len(page_texts)
    sections: list[Section] = []

    for i, (item_num, title, start_page) in enumerate(relevant):
        # End page = start of next section - 1, or last page
        if i + 1 < len(relevant):
            end_page = relevant[i + 1][2] - 1
            if end_page < start_page:
                end_page = start_page
        else:
            end_page = total_pages - 1

        start_char = cum_chars[start_page]
        end_char = cum_chars[end_page + 1]

        sections.append(Section(
            title=title,
            item_number=item_num,
            start_page=start_page,
            end_page=end_page,
            start_char=start_char,
            end_char=end_char,
        ))

    return sections


# ---------------------------------------------------------------------------
# Boilerplate filtering
# ---------------------------------------------------------------------------

# --- Line-level patterns: entire lines that are noise ---
_LINE_DISCARD_RE = [
    # Bare page numbers
    re.compile(r"^\s*\d{1,3}\s*$"),
    # "Page N" footers
    re.compile(r"^\s*(?:Page|p\.?)\s+\d{1,3}\s*$", re.IGNORECASE),
    # Repeated "Table of Contents" breadcrumbs
    re.compile(r"^\s*Table of Contents\s*$", re.IGNORECASE),
    # TOC dotted-leader lines: "Item 7 ........... 42"
    re.compile(r"^.{10,80}\.{4,}\s*\d{1,3}\s*$"),
    # Signature lines "/s/ John Smith"
    re.compile(r"^\s*/s/\s+.+$", re.IGNORECASE),
    # Financial-statement cross-reference footers
    re.compile(
        r"^\s*(?:See |The )?(?:accompanying )?notes?\s+"
        r"(?:to |are an integral part of )"
        r"(?:the )?(?:consolidated )?financial statements",
        re.IGNORECASE,
    ),
]

# --- Paragraph-level patterns: whole paragraphs that are boilerplate ---
_PARA_DISCARD_RE = [
    # Forward-looking statements disclaimer
    re.compile(
        r"forward[- ]looking statements?.*?"
        r"(?:Private Securities Litigation Reform Act|"
        r"actual\s+results?.+?differ\s+materially|"
        r"subject to risks.+?uncertainties|"
        r"undertakes?\s+no\s+obligation\s+to\s+(?:publicly\s+)?update)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Signature block language
    re.compile(
        r"[Pp]ursuant to the requirements of (?:Section 13|the Securities Exchange Act)",
        re.IGNORECASE,
    ),
    # Cover-page checkbox boilerplate
    re.compile(
        r"Indicate by check mark whether the registrant",
        re.IGNORECASE,
    ),
    # SEC filing header
    re.compile(
        r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION",
        re.IGNORECASE,
    ),
    # "Read in conjunction with" preamble
    re.compile(
        r"(?:the following|this)\s+discussion.+?"
        r"should be read in conjunction with",
        re.IGNORECASE,
    ),
    # Certifications (SOX 302/906)
    re.compile(
        r"CERTIFICATION.+?PURSUANT TO.+?"
        r"(?:RULE\s+13a-14|SECTION\s+(?:302|906)|18\s*U\.?S\.?C)",
        re.IGNORECASE,
    ),
    # Power of attorney
    re.compile(
        r"KNOW ALL (?:MEN|PERSONS) BY THESE PRESENTS.+?"
        r"attorney-in-fact",
        re.IGNORECASE | re.DOTALL,
    ),
    # Exhibit index entries
    re.compile(
        r"(?:Filed|Furnished)\s+herewith|"
        r"[Ii]ncorporated (?:herein )?by reference",
        re.IGNORECASE,
    ),
    # Incorporation by reference to proxy
    re.compile(
        r"incorporated (?:herein )?by reference.+?"
        r"(?:Proxy Statement|Definitive Proxy)",
        re.IGNORECASE,
    ),
    # Auditor consent boilerplate
    re.compile(
        r"[Cc]onsent of [Ii]ndependent [Rr]egistered [Pp]ublic [Aa]ccounting [Ff]irm",
    ),
    # XBRL / iXBRL tag artifacts
    re.compile(r"</?(?:ix|xbrli?):[^>]+>"),
    re.compile(r"xmlns:(?:ix|xbrli?|dei|us-gaap)\s*="),
    # EDGAR SGML envelope
    re.compile(
        r"<(?:SEC-HEADER|DOCUMENT|TYPE|SEQUENCE|FILENAME|DESCRIPTION)>",
    ),
    re.compile(r"ACCESSION NUMBER:\s*\d{10}-\d{2}-\d{6}"),
]


def _detect_page_headers(page_texts: list[str], min_freq: float = 0.3) -> set[str]:
    """
    Detect repeated page header/footer strings.

    Many 10-K PDFs repeat the company name, form type, or page-number line
    on every page.  We find short lines (≤120 chars) that appear on ≥30%
    of pages (minimum 5 pages) and treat them as headers to strip.
    """
    line_counts: Counter = Counter()
    total_pages = len(page_texts)
    if total_pages < 10:
        return set()  # Not enough pages for reliable frequency detection

    for page_text in page_texts:
        # Deduplicate within a page to avoid double-counting
        seen_on_page: set[str] = set()
        for line in page_text.splitlines():
            stripped = line.strip()
            if 3 <= len(stripped) <= 120 and stripped not in seen_on_page:
                seen_on_page.add(stripped)
                line_counts[stripped] += 1

    threshold = max(int(total_pages * min_freq), 5)
    return {line for line, count in line_counts.items() if count >= threshold}


def _strip_boilerplate_page(
    page_text: str,
    repeated_headers: set[str],
) -> tuple[str, int]:
    """
    Filter a single page's text, removing boilerplate lines and paragraphs.

    Returns (cleaned_text, chars_removed).
    """
    original_len = len(page_text)

    # --- Phase 1: Line-level filtering ---
    lines = page_text.splitlines(keepends=True)
    kept_lines: list[str] = []
    for line in lines:
        stripped = line.strip()

        # Skip repeated page headers/footers
        if stripped in repeated_headers:
            continue

        # Skip lines matching line-discard patterns
        if any(pat.match(stripped) for pat in _LINE_DISCARD_RE):
            continue

        kept_lines.append(line)

    text = "".join(kept_lines)

    # --- Phase 2: Paragraph-level filtering ---
    paragraphs = text.split("\n\n")
    kept_paragraphs: list[str] = []
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue
        # Skip paragraphs matching discard patterns
        if any(pat.search(para_stripped) for pat in _PARA_DISCARD_RE):
            continue
        kept_paragraphs.append(para)

    cleaned = "\n\n".join(kept_paragraphs)
    chars_removed = original_len - len(cleaned)
    return cleaned, max(chars_removed, 0)


def strip_boilerplate(page_texts: list[str]) -> tuple[list[str], int]:
    """
    Strip boilerplate from all pages.

    Returns (cleaned_page_texts, total_chars_removed).
    """
    repeated_headers = _detect_page_headers(page_texts)
    cleaned: list[str] = []
    total_removed = 0

    for page_text in page_texts:
        clean_text, removed = _strip_boilerplate_page(page_text, repeated_headers)
        cleaned.append(clean_text)
        total_removed += removed

    return cleaned, total_removed


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

# Horizontal rule: a line consisting entirely of 5+ repeated dashes,
# equals signs, or underscores (with optional surrounding whitespace).
_HORIZONTAL_RULE_RE = re.compile(r"^[ \t]*[-=_]{5,}[ \t]*$", re.MULTILINE)

# Runs of 2+ inline spaces (not at start of line — preserve indentation).
_INLINE_MULTI_SPACE_RE = re.compile(r"(?<=\S)[ \t]{2,}")

# Three or more consecutive newlines → collapse to double newline.
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in extracted PDF text without losing content.

    Four safe transformations (ordered for correctness):
      1. Strip trailing whitespace from every line.
      2. Collapse horizontal-rule lines (-----, =====, _____) to a
         single short marker so table/section boundaries remain visible.
      3. Collapse runs of 2+ inline spaces (after non-space chars) to
         a single space.  Preserves leading indentation.
      4. Collapse 3+ consecutive blank lines to a single blank line.

    These operations are content-preserving: no words, numbers, or
    punctuation are removed.  Only redundant whitespace inserted by
    pymupdf's spatial-layout reconstruction is compressed.
    """
    # 1. Trailing whitespace per line
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)

    # 2. Horizontal rules → short marker (keeps table boundaries visible)
    text = _HORIZONTAL_RULE_RE.sub("---", text)

    # 3. Collapse inline multi-space runs (preserve leading indent)
    text = _INLINE_MULTI_SPACE_RE.sub(" ", text)

    # 4. Collapse excess blank lines
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)

    return text


def normalize_pages(page_texts: list[str]) -> tuple[list[str], int]:
    """
    Apply whitespace normalization to all pages.

    Returns (normalized_page_texts, total_chars_saved).
    """
    normalized: list[str] = []
    total_saved = 0
    for page_text in page_texts:
        before = len(page_text)
        clean = normalize_whitespace(page_text)
        normalized.append(clean)
        total_saved += before - len(clean)
    return normalized, total_saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_document(file_obj, filter_boilerplate: bool = True) -> ParsedDocument:
    """
    Parse a PDF file, detect sections, and optionally strip boilerplate.

    Parameters
    ----------
    file_obj : file-like
        Streamlit UploadedFile or any file-like with .read() and .name.
    filter_boilerplate : bool
        If True (default), strip SEC filing boilerplate (cover pages,
        forward-looking disclaimers, signature blocks, page headers, etc.)
        before returning.  This reduces token waste and improves signal
        density for LLM generation.

    Returns
    -------
    ParsedDocument with page_texts always populated.
    sections populated if 10-K/10-Q structure detected, else empty.
    """
    raw_bytes = file_obj.read()
    filename = getattr(file_obj, "name", "document.pdf")

    doc = fitz.open(stream=raw_bytes, filetype="pdf")

    raw_page_texts: list[str] = []
    for page in doc:
        raw_page_texts.append(page.get_text("text") or "")

    # Detect sections on RAW text (before stripping) so headers are intact
    sections = _detect_sections_from_toc(doc, raw_page_texts)
    if not sections:
        sections = _detect_sections_from_text(raw_page_texts)

    doc.close()

    # Strip boilerplate after section detection
    boilerplate_removed = 0
    if filter_boilerplate:
        page_texts, boilerplate_removed = strip_boilerplate(raw_page_texts)
    else:
        page_texts = raw_page_texts

    # Normalize whitespace (runs after boilerplate so we don't waste
    # cycles normalizing text that was going to be discarded anyway)
    whitespace_removed = 0
    page_texts, whitespace_removed = normalize_pages(page_texts)

    total_chars = sum(len(t) for t in page_texts)

    # Recompute section char offsets on cleaned text
    if sections:
        cum_chars = [0]
        for pt in page_texts:
            cum_chars.append(cum_chars[-1] + len(pt))
        for sec in sections:
            sec.start_char = cum_chars[sec.start_page]
            end_idx = min(sec.end_page + 1, len(cum_chars) - 1)
            sec.end_char = cum_chars[end_idx]

    return ParsedDocument(
        filename=filename,
        total_pages=len(page_texts),
        total_chars=total_chars,
        sections=sections,
        page_texts=page_texts,
        boilerplate_chars_removed=boilerplate_removed,
        whitespace_chars_removed=whitespace_removed,
    )


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def assemble_context(
    doc: ParsedDocument,
    selected_sections: Optional[list[Section]] = None,
    selected_pages: Optional[tuple[int, int]] = None,
    char_limit: int = 15_000,
) -> str:
    """
    Assemble context string from selected sections or page range.

    Parameters
    ----------
    doc : ParsedDocument
    selected_sections : list[Section], optional
        Sections to include (section mode).
    selected_pages : tuple[int, int], optional
        (start_page, end_page) 0-indexed inclusive (page-range mode).
    char_limit : int
        Maximum character count for the returned context.

    Returns
    -------
    str — assembled context, possibly truncated.
    """
    if selected_sections is not None and selected_pages is not None:
        raise ValueError("Provide selected_sections or selected_pages, not both.")
    if selected_sections is None and selected_pages is None:
        raise ValueError("Provide either selected_sections or selected_pages.")

    if selected_sections is not None:
        parts = []
        for sec in selected_sections:
            header = f"--- {sec.title} ---"
            body = "".join(doc.page_texts[sec.start_page : sec.end_page + 1])
            parts.append(f"{header}\n\n{body}")
        full_text = "\n\n".join(parts)
    else:
        start, end = selected_pages
        full_text = "".join(doc.page_texts[start : end + 1])

    if len(full_text) <= char_limit:
        return full_text

    truncated = full_text[:char_limit]
    remaining = len(full_text) - char_limit
    truncated += f"\n\n[... truncated, content continues for {remaining:,} more chars]"
    return truncated


# ---------------------------------------------------------------------------
# Section chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A chunk of a section that fits within a character budget."""
    index: int              # 0-based chunk number
    total: int              # total number of chunks
    text: str               # the chunk content
    char_count: int         # len(text)
    section_title: str      # parent section title
    start_char: int         # char offset within the section
    end_char: int           # char offset within the section


def chunk_section(
    doc: ParsedDocument,
    section: Section,
    chunk_budget: int = 60_000,
    overlap: int = 500,
) -> list[Chunk]:
    """
    Split a section into chunks that each fit within ``chunk_budget`` chars.

    Uses paragraph-aware splitting (same pattern as EquityResearchChunker):
    splits on double-newlines so chunks break at paragraph boundaries rather
    than mid-sentence.  Consecutive paragraphs are accumulated until adding
    the next paragraph would exceed the budget.

    Parameters
    ----------
    doc : ParsedDocument
    section : Section
    chunk_budget : int
        Max chars per chunk.
    overlap : int
        Chars of overlap between consecutive chunks for context continuity.
        The last ``overlap`` chars of chunk N are prepended to chunk N+1.

    Returns
    -------
    list[Chunk] — always at least one chunk even if section is empty.
    """
    # Extract full section text
    body = "".join(doc.page_texts[section.start_page : section.end_page + 1])

    # If it fits, return a single chunk
    if len(body) <= chunk_budget:
        return [Chunk(
            index=0, total=1, text=body, char_count=len(body),
            section_title=section.title, start_char=0, end_char=len(body),
        )]

    # Split into paragraphs (same approach as EquityResearchChunker)
    paragraphs = [p for p in body.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_chars = 0
    section_offset = 0  # running char offset within the section

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the "\n\n" join separator

        # If a single paragraph exceeds the budget, hard-split it
        if para_len > chunk_budget and not current_parts:
            for start in range(0, len(para), chunk_budget - overlap):
                piece = para[start : start + chunk_budget]
                chunks.append(Chunk(
                    index=len(chunks), total=0,  # total filled in at end
                    text=piece, char_count=len(piece),
                    section_title=section.title,
                    start_char=section_offset + start,
                    end_char=section_offset + start + len(piece),
                ))
            section_offset += len(para) + 2
            continue

        # Would adding this paragraph exceed the budget?
        if current_chars + para_len > chunk_budget and current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(Chunk(
                index=len(chunks), total=0,
                text=chunk_text, char_count=len(chunk_text),
                section_title=section.title,
                start_char=section_offset,
                end_char=section_offset + len(chunk_text),
            ))
            section_offset += len(chunk_text) + 2

            # Overlap: carry the last paragraph into the next chunk
            if overlap > 0 and current_parts:
                last_para = current_parts[-1]
                carry = last_para[-overlap:] if len(last_para) > overlap else last_para
                current_parts = [carry]
                current_chars = len(carry)
            else:
                current_parts = []
                current_chars = 0

        current_parts.append(para)
        current_chars += para_len

    # Flush remaining
    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        chunks.append(Chunk(
            index=len(chunks), total=0,
            text=chunk_text, char_count=len(chunk_text),
            section_title=section.title,
            start_char=section_offset,
            end_char=section_offset + len(chunk_text),
        ))

    # Fill in totals
    for c in chunks:
        c.total = len(chunks)

    return chunks


def chunk_context(
    doc: ParsedDocument,
    selected_sections: list[Section],
    chunk_budget: int = 60_000,
    overlap: int = 500,
) -> list[Chunk]:
    """
    Chunk multiple selected sections. Sections that fit within the budget
    become a single chunk; oversized sections are split.

    Returns a flat list of chunks across all selected sections.
    """
    all_chunks: list[Chunk] = []
    for sec in selected_sections:
        sec_chunks = chunk_section(doc, sec, chunk_budget, overlap)
        # Re-index across the combined list
        for c in sec_chunks:
            c.index = len(all_chunks)
            all_chunks.append(c)
    # Update totals
    for c in all_chunks:
        c.total = len(all_chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def get_section_summary(sections: list[Section], char_budget: int = 15_000) -> list[dict]:
    """
    Build a summary table for the section-selector UI.

    Returns list of dicts suitable for st.dataframe().
    """
    rows = []
    for sec in sections:
        chars = sec.char_count
        rows.append({
            "Section": sec.title[:80],
            "Item": sec.item_number,
            "Pages": sec.page_range_display,
            "Chars": f"{chars:,}",
            "~Words": f"{chars // 5:,}",
            "Fits Budget": "\u2705" if chars <= char_budget else "\u26a0\ufe0f",
        })
    return rows
