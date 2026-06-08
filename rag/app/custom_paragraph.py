import copy
import logging
import re

from rag.app.custom_parser import (
    _build_media_doc,
    _build_text_doc,
    _chunk_units,
    _clean_text,
    _detect_repeated_boilerplate,
    _detect_repeated_title_boilerplate,
    _detect_toc_pages,
    _doc_meta,
    _format_table_content,
    _get_positions,
    _is_boilerplate_text,
    _is_page_number,
    _load_payload,
    _merge_positions,
    _merge_short_page_continuations,
    _normalized_repeat_key,
    _page_number_from_positions,
    _should_merge_with_previous,
    tokenize_table,
)


_PARAGRAPH_END_RE = re.compile(r"[。！？；;:]\s*$")


def _build_paragraph_title(base_title, active_title):
    active_title = _clean_text(active_title)
    if active_title:
        return active_title
    return _clean_text(base_title)


def _append_unit(units, text, positions, force_new=False):
    text = _clean_text(text)
    if not text:
        return

    if (
        not force_new
        and units
        and not units[-1].get("force_new")
        and _should_merge_with_previous(units[-1]["text"], text)
    ):
        units[-1]["text"] = f"{units[-1]['text']}  {text}".strip()
        units[-1]["positions"] = _merge_positions([units[-1]["positions"], positions])
        return

    units.append(
        {
            "text": text,
            "positions": positions or [],
            "force_new": bool(force_new),
        }
    )


def _flush_units(results, doc, units, eng, max_tokens, overlap_tokens):
    if not units:
        return []

    merged_units = []
    current_text = ""
    current_positions = []

    def flush_current_unit():
        nonlocal current_text, current_positions
        current_text = _clean_text(current_text)
        if not current_text:
            current_text = ""
            current_positions = []
            return
        merged_units.append({"text": current_text, "positions": current_positions})
        current_text = ""
        current_positions = []

    for item in units:
        text = _clean_text(item.get("text"))
        if not text:
            continue
        positions = item.get("positions") or []

        if not current_text:
            current_text = text
            current_positions = positions
            continue

        if _PARAGRAPH_END_RE.search(current_text):
            flush_current_unit()
            current_text = text
            current_positions = positions
            continue

        current_text = f"{current_text}  {text}".strip()
        current_positions = _merge_positions([current_positions, positions])

    flush_current_unit()

    for unit in merged_units:
        for chunk in _chunk_units([unit], max_tokens, overlap_tokens):
            results.append(
                _build_text_doc(
                    doc,
                    chunk["text"],
                    eng,
                    chunk["positions"],
                )
            )
    return []


def _split_text_blocks(text):
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    paragraphs = []
    for block in re.split(r"\n\s*\n", cleaned):
        block = re.sub(r"\s*\n\s*", " ", block)
        block = _clean_text(block)
        if block:
            paragraphs.append(block)
    return paragraphs


def custom_parse(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, **kwargs):
    """
    Parse structured article JSON and chunk it by paragraphs.

    Compared with custom_parser, this parser does not maintain hierarchical
    heading paths in chunk titles. Titles are only used as paragraph
    boundaries and local context for tables/images.
    """

    logging.info("Custom paragraph parsing %s", filename)
    if callback:
        callback(0.1, "Loading structured article JSON.")

    parser_config = kwargs.get("parser_config", {}) or {}
    payload = _load_payload(filename, binary, parser_config=parser_config, callback=callback)
    partitions = payload.get("partitions") or []
    if not partitions:
        logging.warning("No partitions found in custom paragraph parser input.")
        return []

    max_tokens = max(64, int(parser_config.get("chunk_token_num", 1024) or 1024))
    overlap_percent = max(0, min(50, int(parser_config.get("overlapped_percent", 10) or 10)))
    overlap_tokens = int(max_tokens * overlap_percent / 100)
    strip_toc = str(parser_config.get("strip_toc", "true")).lower() != "false"
    strip_repeated_boilerplate = str(parser_config.get("strip_repeated_boilerplate", "true")).lower() != "false"
    keep_cover_page_boilerplate = str(parser_config.get("keep_cover_page_boilerplate", "false")).lower() != "false"
    include_title_paragraphs = str(parser_config.get("include_title_paragraphs", "true")).lower() != "false"
    eng = lang.lower() == "english"

    doc = _doc_meta(filename)
    base_title = doc.get("title_tks") or ""
    toc_pages = _detect_toc_pages(partitions) if strip_toc else set()
    repeated_boilerplate_keys = _detect_repeated_boilerplate(partitions) if strip_repeated_boilerplate else set()
    repeated_title_boilerplate_keys = _detect_repeated_title_boilerplate(partitions) if strip_repeated_boilerplate else set()
    has_major_titles = any(
        (part.get("type") or "").strip().lower() == "title" and part.get("title_type") == 2
        for part in partitions
    )

    results = []
    table_items = []
    paragraph_units = []
    active_title = ""
    started_major_content = not has_major_titles

    if callback:
        callback(0.2, "Cleaning page artifacts and collecting paragraphs.")

    for part in partitions:
        positions = _get_positions(part)
        part_pages = [pos[0] for pos in positions] if positions else []
        if part_pages and (max(part_pages) < from_page or min(part_pages) > to_page):
            continue

        page_no = _page_number_from_positions(positions)
        if strip_toc and page_no in toc_pages:
            continue
        if strip_repeated_boilerplate and _is_boilerplate_text(
            part,
            repeated_boilerplate_keys,
            keep_cover_page=keep_cover_page_boilerplate,
        ):
            continue

        part_type = (part.get("type") or "").strip().lower()
        title_type = part.get("title_type")
        text = _clean_text(part.get("text"))
        caption = _clean_text(part.get("caption"))
        data = _clean_text(part.get("data"))

        if not started_major_content:
            if part_type == "title" and title_type == 2 and text and text != "目录":
                started_major_content = True
            else:
                continue

        if part_type == "title":
            title_key = _normalized_repeat_key(text)
            if (
                strip_repeated_boilerplate
                and title_key
                and title_key in repeated_title_boilerplate_keys
                and not (keep_cover_page_boilerplate and page_no in {1, 2})
            ):
                continue
            if not text or text == "目录":
                continue

            if title_type in {1, 2}:
                paragraph_units = _flush_units(
                    results,
                    doc,
                    paragraph_units,
                    eng,
                    max_tokens,
                    overlap_tokens,
                )
                active_title = text

            if include_title_paragraphs:
                _append_unit(paragraph_units, text, positions, force_new=True)
            continue

        if part_type == "text":
            if not text or _is_page_number(text):
                continue
            for block in _split_text_blocks(text):
                if not block or _is_page_number(block):
                    continue
                _append_unit(paragraph_units, block, positions)
            continue

        if part_type == "table":
            paragraph_units = _flush_units(
                results,
                doc,
                paragraph_units,
                eng,
                max_tokens,
                overlap_tokens,
            )
            title_context = _build_paragraph_title(base_title, active_title)
            table_text = _format_table_content(title_context, caption, text, data)
            table_items.append(((None, table_text), positions))
            continue

        if part_type in {"image", "chart"}:
            paragraph_units = _flush_units(
                results,
                doc,
                paragraph_units,
                eng,
                max_tokens,
                overlap_tokens,
            )
            title_context = _build_paragraph_title(base_title, active_title)
            media_parts = []
            if title_context:
                media_parts.append(f"Section: {title_context}")
            if caption:
                media_parts.append(f"Caption: {caption}")
            if text:
                media_parts.append(f"Summary: {text}")
            if part.get("image_path"):
                media_parts.append(f"Image path: {part.get('image_path')}")
            media_text = "\n".join(media_parts).strip()
            if media_text:
                results.append(_build_media_doc(doc, media_text, eng, positions, "image" if part_type == "image" else "chart"))
            continue

        fallback = "\n".join(item for item in [text, caption, data] if item).strip()
        if fallback:
            _append_unit(paragraph_units, fallback, positions, force_new=True)

    paragraph_units = _flush_units(
        results,
        doc,
        paragraph_units,
        eng,
        max_tokens,
        overlap_tokens,
    )

    if table_items:
        results.extend(tokenize_table(table_items, copy.deepcopy(doc), eng))

    results = _merge_short_page_continuations(results, eng)

    if callback:
        callback(0.9, f"Custom paragraph parsing completed with {len(results)} chunks.")
    return results


chunk = custom_parse
