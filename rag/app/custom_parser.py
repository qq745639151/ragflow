import copy
import json
import logging
import mimetypes
import re
import os
from collections import Counter, defaultdict
from functools import lru_cache
from html import unescape
from typing import Any

import httpx

try:
    from rag.nlp import add_positions, rag_tokenizer, tokenize, tokenize_table
except Exception:  # pragma: no cover - local fallback for offline validation
    class _FallbackTokenizer:
        _tok_re = re.compile(r"[A-Za-z0-9_./+-]+|[\u4e00-\u9fff]+|[^\s]")

        def tokenize(self, text):
            if text is None:
                return ""
            return " ".join(self._tok_re.findall(str(text)))

        def fine_grained_tokenize(self, text):
            if text is None:
                return ""
            if isinstance(text, str):
                return text
            return " ".join(text)

    rag_tokenizer = _FallbackTokenizer()

    def tokenize(doc, text, eng):
        doc["content_with_weight"] = text
        ltks = rag_tokenizer.tokenize(text)
        doc["content_ltks"] = ltks
        doc["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(ltks)

    def add_positions(doc, positions):
        page_nums = []
        cleaned = []
        tops = []
        for pos in positions:
            if len(pos) != 5:
                continue
            page_idx, left, right, top, bottom = pos
            page_nums.append(int(page_idx) + 1)
            cleaned.append([int(page_idx) + 1, int(left), int(right), int(top), int(bottom)])
            tops.append(int(top))
        if cleaned:
            doc["page_num_int"] = page_nums
            doc["position_int"] = cleaned
            doc["top_int"] = tops

    def tokenize_table(table_items, doc, eng):
        docs = []
        for item, positions in table_items:
            table_text = ""
            if isinstance(item, tuple) and len(item) >= 2:
                table_text = item[1]
            elif isinstance(item, str):
                table_text = item
            d = copy.deepcopy(doc)
            tokenize(d, table_text, eng)
            d["doc_type_kwd"] = "table"
            if positions:
                add_positions(d, positions)
            docs.append(d)
        return docs


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f!?;\uff1b：:\n])")
_TOC_CHAPTER_RE = re.compile(r"^(?:第[一二三四五六七八九十百千万零〇两]+章|附录\s*[0-9一二三四五六七八九十]+|\d+\.(?=\s|[^\d]|$)|\d+(?:\.\d+)+)\b")
_BARE_NUMERIC_HEAD_RE = re.compile(r"^(\d{1,3})(?=\s|[^\d.]|$)")
_NUMERIC_HEAD_RE = re.compile(r"^(\d+(?:\.\d+){1,8})(?=\s|[^\d.])")
_NUMERIC_SINGLE_HEAD_RE = re.compile(r"^(\d+)\.(?=\s|[^\d]|$)")
_PAREN_HEAD_RE = re.compile(r"^(?:[（(][0-9一二三四五六七八九十]+[)）]|\d+[)）])")
_CHAPTER_HEAD_RE = re.compile(r"^第[一二三四五六七八九十百千万零〇两]+章\b")
_APPENDIX_HEAD_RE = re.compile(r"^附录\s*[0-9一二三四五六七八九十]+")
_PAGE_NUM_SUFFIX_RE = re.compile(r"(?:\.{2,}|…+)?\s*\d+$")
_TOKEN_CACHE_LIMIT = 50000
_TEXT_CHAPTER_HEAD_RE = re.compile(r"^第[一二三四五六七八九十百千0-9]+章\s*\S.+$")
_TEXT_NUMERIC_HEAD_RE = re.compile(r"^(?:[1-9]\s*\d?)\s*(?!\s*[\.．。]\s*\d)\s*[\.．。]?\s*\S.*$")
_CHAPTER_HEAD_RE = re.compile(r"^\u7b2c[一二三四五六七八九十百千万零〇两0-9]+章(?:\s|$)")
_APPENDIX_HEAD_RE = re.compile(r"^\u9644\u5f55\s*[0-9一二三四五六七八九十百千万零〇两]+(?:\s|$)")
_PAREN_HEAD_RE = re.compile(r"^(?:[（(][0-9一二三四五六七八九十百千万零〇两]+[)）]|\d+[)）])(?:\s|$)")


def _clean_text(text):
    if text is None:
        return ""
    text = unescape(str(text)).replace("\r", "\n").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_page_number(text):
    """
    Check if text is a page number污染.
    Matches patterns like: "— 83 —", "—83—", "91", "1", etc.
    """
    if not text:
        return False
    # 匹配纯数字（1-3位）
    if re.match(r"^\d{1,3}$", text):
        return True
    # 匹配破折号包围的数字，如 "— 83 —", "—83—", "— 1 —"
    if re.match(r"^[\u2014\u2013\u002d]{1,2}\s*\d{1,3}\s*[\u2014\u2013\u002d]{1,2}$", text):
        return True
    return False


@lru_cache(maxsize=_TOKEN_CACHE_LIMIT)
def _token_count_cached(text):
    tokenized = rag_tokenizer.tokenize(text or "")
    if isinstance(tokenized, str):
        tokenized = tokenized.strip()
        if not tokenized:
            return 0
        return len([tok for tok in tokenized.split() if tok])
    try:
        return len(tokenized)
    except Exception:
        return len([tok for tok in str(tokenized).split() if tok])


def _token_count(text):
    return _token_count_cached(text or "")


def _json_loads_maybe_wrapped(raw: Any):
    if isinstance(raw, str):
        raw = raw.strip()
        return json.loads(raw) if raw else {}
    return raw


def _optional_timeout_value(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "none", "null", "false", "off"}:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unwrap_xidian_payload(data):
    data = _json_loads_maybe_wrapped(data)

    if isinstance(data, list):
        return {"partitions": data}

    if not isinstance(data, dict):
        raise ValueError("Unexpected Xidian preprocess response type.")

    if isinstance(data.get("partitions"), list):
        return data

    for key in ("data", "result", "payload", "output", "json"):
        value = data.get(key)
        if value is None:
            continue
        value = _json_loads_maybe_wrapped(value)
        if isinstance(value, list):
            return {"partitions": value}
        if isinstance(value, dict) and isinstance(value.get("partitions"), list):
            return value

    raise ValueError("Xidian preprocess response does not contain 'partitions'.")


def _fetch_xidian_payload(filename, binary, parser_config, callback=None):
    api_url = (
        os.getenv("EXTERNAL_PREPROCESS_URL", "http://192.168.100.15:8003/api/v1/xidian/preprocess_required")
    )
    timeout = _optional_timeout_value(parser_config.get("xidian_timeout"), default=None)
    connect_timeout = _optional_timeout_value(parser_config.get("xidian_connect_timeout"), default=None)
    write_timeout = _optional_timeout_value(parser_config.get("xidian_write_timeout"), default=None)
    trust_env = str(parser_config.get("xidian_trust_env", "false")).lower() == "true"
    query = {
        "vlm_enable": str(parser_config.get("vlm_enable", True)).lower(),
        "red_title_enable": str(parser_config.get("red_title_enable", True)).lower(),
        "img_class": str(parser_config.get("img_class", True)).lower(),
        "img_desc": str(parser_config.get("img_desc", True)).lower(),
        "img_html": str(parser_config.get("img_html", True)).lower(),
        "table_kv": str(parser_config.get("table_kv", True)).lower(),
        "table_desc": str(parser_config.get("table_desc", True)).lower(),
        "table_html": str(parser_config.get("table_html", True)).lower(),
    }
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    if callback:
        callback(0.05, f"Calling Xidian preprocess API: {api_url}")

    client_timeout = httpx.Timeout(
        connect=connect_timeout,
        write=write_timeout,
        read=timeout,
        pool=connect_timeout,
    )

    try:
        with httpx.Client(timeout=client_timeout, trust_env=trust_env) as client:
            response = client.post(
                api_url,
                params=query,
                files={"file": (filename, binary, content_type)},
            )
            response.raise_for_status()
            return _unwrap_xidian_payload(response.json())
    except httpx.ReadTimeout as exc:
        raise RuntimeError(
            f"Xidian preprocess API read timeout after {timeout}s. "
            f"api_url={api_url}, trust_env={trust_env}. "
            "If you are using a system proxy, try setting parser_config['xidian_trust_env']=False "
            "or increase xidian_timeout."
        ) from exc


def _load_payload(filename, binary, parser_config=None, callback=None):
    parser_config = parser_config or {}
    if binary and not str(filename).lower().endswith(".json"):
        return _fetch_xidian_payload(filename, binary, parser_config, callback=callback)

    if binary:
        text = binary.decode("utf-8", errors="ignore")
    else:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
    return json.loads(text)


def _doc_meta(filename):
    title = re.sub(r"\.[a-zA-Z0-9]+$", "", filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
    return {
        "docnm_kwd": filename,
        "title_tks": title,
        "title_sm_tks": rag_tokenizer.tokenize(title),
    }


def _get_positions(part):
    extra = ((part.get("metadata") or {}).get("extra_data") or {})
    pages = extra.get("pages") or []
    bboxes = extra.get("bboxes") or []
    poss = []
    for idx, bbox in enumerate(bboxes):
        if len(bbox) != 4:
            continue
        page = pages[idx] if idx < len(pages) else (pages[0] if pages else 1)
        left, top, right, bottom = bbox
        poss.append((max(int(page) - 1, 0), int(left), int(right), int(top), int(bottom)))
    return poss


def _merge_positions(positions_list):
    merged = []
    seen = set()
    for poss in positions_list:
        for pos in poss or []:
            if pos in seen:
                continue
            seen.add(pos)
            merged.append(pos)
    return merged


def _page_number_from_positions(positions):
    if not positions:
        return None
    return positions[0][0] + 1


def _normalize_heading_text(text):
    text = _clean_text(text)
    if not text:
        return ""
    if _TOC_CHAPTER_RE.match(text):
        text = _PAGE_NUM_SUFFIX_RE.sub("", text).strip()
    return text


def _normalize_heading_for_match(text):
    text = _normalize_heading_text(text)
    if not text:
        return ""
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"\s*([.．。])\s*", ".", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _match_heading_number(text):
    normalized = _normalize_heading_for_match(text)
    if not normalized:
        return None
    # Only treat leading numeric section markers up to two digits as headings.
    # This prevents prose like "750千伏..." from being mistaken as a numbered title.
    match = re.match(r"^(?P<num>\d{1,2}(?:\.\d{1,3}){0,7})\.?(?P<rest>.*)$", normalized)
    if not match:
        return None
    rest = (match.group("rest") or "").lstrip()
    if not rest:
        return None
    if rest[0] in ".．。,:：;；、)]）】》〉＞-—0123456789":
        return None
    return match


def _normalized_repeat_key(text):
    text = _clean_text(text)
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _normalize_heading_level(level):
    if isinstance(level, int) and level > 0:
        return level
    return None


_CN_NUM_MAP = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT_MAP = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _cn_to_int(text):
    if not text:
        return None
    text = str(text)
    if text.isdigit():
        return int(text)
    total = 0
    section = 0
    num = 0
    for ch in text:
        if ch in _CN_NUM_MAP:
            num = _CN_NUM_MAP[ch]
            continue
        unit = _CN_UNIT_MAP.get(ch)
        if unit is None:
            return None
        if num == 0:
            num = 1
        if unit == 10000:
            section = (section + num) * unit
            total += section
            section = 0
        else:
            section += num * unit
        num = 0
    return total + section + num


def _extract_chapter_number(text):
    normalized = _normalize_heading_for_match(text)
    if not normalized:
        return None
    match = re.match(r"^第([一二三四五六七八九十百千万零〇两0-9]+)章\b", normalized)
    if not match:
        return None
    return _cn_to_int(match.group(1))


def _match_heading_number_relaxed(text):
    normalized = _normalize_heading_for_match(text)
    if not normalized:
        return None
    match = re.match(r"^(?P<num>\d{1,2}(?:\.\d{1,3}){1,7})\.?\s*(?P<rest>\S.*)$", normalized)
    if not match:
        return None
    rest = (match.group("rest") or "").strip()
    if not rest:
        return None
    if rest[0] in ".．。)]）】》〉＞-—":
        return None
    return match


def _extract_numeric_root(text):
    match = _match_heading_number_relaxed(text)
    if not match:
        return None
    try:
        return int(match.group("num").split(".")[0])
    except Exception:
        return None


def _is_short_parenthetical_heading(text, max_heading_chars=64):
    t = _normalize_heading_text(text)
    if not t or len(t) > max_heading_chars:
        return False
    if not re.match(r"^(?:[（(][0-9一二三四五六七八九十百千万零〇两]+[)）]|\d+[)）])\s*\S", t):
        return False
    body = re.sub(r"^(?:[（(][0-9一二三四五六七八九十百千万零〇两]+[)）]|\d+[)）])\s*", "", t).strip()
    if not body or len(body) > 40:
        return False
    if re.search(r"[。；;？！?!：:]", body):
        return False
    if len(re.findall(r"[，,、]", body)) >= 2:
        return False
    return True


def _is_chapter_heading_text(text):
    normalized = _normalize_heading_for_match(text)
    if not normalized:
        return False
    return bool(_CHAPTER_HEAD_RE.match(normalized) or _APPENDIX_HEAD_RE.match(normalized))


def _has_same_active_heading(headings, level, text):
    target = _normalize_heading_for_match(text)
    if not target:
        return False
    for item in reversed(headings):
        if item.get("level") != level:
            continue
        return _normalize_heading_for_match(item.get("text")) == target
    return False


def _collect_page_heading_candidates(partitions):
    page_candidates = defaultdict(list)
    for part in partitions:
        part_type = (part.get("type") or "").strip().lower()
        if part_type not in {"text", "title"}:
            continue
        text = _clean_text(part.get("text"))
        if not _is_chapter_heading_text(text):
            continue
        positions = _get_positions(part)
        page_no = _page_number_from_positions(positions)
        if page_no is None:
            continue
        top = positions[0][3] if positions else 10 ** 9
        page_candidates[page_no].append((top, _normalize_heading_text(text)))
    return {
        page_no: [text for _, text in sorted(items, key=lambda item: item[0])]
        for page_no, items in page_candidates.items()
    }


def _active_chapter_number(headings):
    for item in reversed(headings):
        num = _extract_chapter_number(item.get("text"))
        if num is not None:
            return num
    return None


def _reconcile_heading_context(headings, heading_text, page_no, page_heading_candidates):
    numeric_root = _extract_numeric_root(heading_text)
    if numeric_root is None:
        return
    current_chapter_num = _active_chapter_number(headings)
    if current_chapter_num == numeric_root:
        return

    matched_chapter_text = None
    for candidate in page_heading_candidates.get(page_no, []):
        if _extract_chapter_number(candidate) == numeric_root:
            matched_chapter_text = candidate
            break

    if current_chapter_num is not None and current_chapter_num != numeric_root:
        while headings and headings[-1]["level"] >= 2:
            headings.pop()

    if matched_chapter_text and not _has_same_active_heading(headings, 2, matched_chapter_text):
        _push_heading(headings, 2, matched_chapter_text)


def _resolve_parenthetical_level(headings, fallback_level=None):
    fallback = _normalize_heading_level(fallback_level)
    if fallback:
        return fallback
    for item in reversed(headings):
        text = item.get("text") or ""
        normalized = _normalize_heading_for_match(text)
        if _PAREN_HEAD_RE.match(normalized):
            return item.get("level") or 2
        if _match_heading_number_relaxed(normalized) or _match_heading_number(normalized) or _CHAPTER_HEAD_RE.match(normalized) or _APPENDIX_HEAD_RE.match(normalized):
            return min((item.get("level") or 1) + 1, 8)
    return 2



def _heading_level_from_text(text, fallback_level=None, previous_level=0):
    t = _normalize_heading_for_match(text)
    fallback = _normalize_heading_level(fallback_level)
    if not t:
        return fallback or 1
    if t == "目录":
        return 0
    if _CHAPTER_HEAD_RE.match(t):
        return fallback or 2
    if _APPENDIX_HEAD_RE.match(t):
        return fallback or 2
    match = _match_heading_number_relaxed(t) or _match_heading_number(t)
    if match:
        # For numbered headings, the textual numbering is more stable than
        # upstream title_type values in these policy documents.
        return len(match.group("num").split(".")) + 1
    if _PAREN_HEAD_RE.match(t):
        if fallback:
            return fallback
        return min(max(previous_level, 2), 8)
    if fallback:
        return fallback
    return max(previous_level, 1)

def _heading_path(headings):
    values = [item["text"] for item in headings if item.get("text")]
    return "   ".join(values)


def _apply_chunk_title(d, heading_path):
    heading_path = _clean_text(heading_path)
    if not heading_path:
        return
    d["title_tks"] = heading_path
    d["title_sm_tks"] = rag_tokenizer.tokenize(heading_path)



def _build_text_doc(doc, content, eng, positions, heading_path="", include_heading=True):
    d = copy.deepcopy(doc)
    text = content.strip()
    heading_path = _clean_text(heading_path)
    if include_heading and heading_path and not text.startswith(heading_path):
        text = f"{heading_path}\n\n{text}" if text else heading_path
    tokenize(d, text, eng)
    _apply_chunk_title(d, heading_path)
    if positions:
        add_positions(d, positions)
    return d


def _build_media_doc(doc, content, eng, positions, doc_type):
    d = copy.deepcopy(doc)
    tokenize(d, content, eng)
    d["doc_type_kwd"] = doc_type
    if positions:
        add_positions(d, positions)
    return d


def _format_table_content(heading_path, caption, summary, table_html):
    parts = []
    if heading_path:
        parts.append(f"Section: {heading_path}")
    if caption:
        parts.append(f"Caption: {caption}")
    if summary:
        parts.append(f"Summary: {summary}")
    if table_html:
        parts.append(table_html)
    return "\n".join(parts).strip()


def _starts_with_marker(text):
    t = _normalize_heading_for_match(text)
    if not t:
        return False
    return bool(
        _match_heading_number(t)
        or _CHAPTER_HEAD_RE.match(t)
        or _APPENDIX_HEAD_RE.match(t)
        or _PAREN_HEAD_RE.match(t)
    )


def _is_heading_like_block(text, max_heading_chars=64):
    t = _normalize_heading_text(text)
    if not t or len(t) > max_heading_chars:
        return False
    normalized = _normalize_heading_for_match(t)
    if not _starts_with_marker(normalized):
        return False
    if re.search(r"[。！？!?；;]$", normalized):
        return False
    colon_match = re.search(r"[：:]", normalized)
    if colon_match:
        right = normalized[colon_match.end():].strip()
        if right:
            return False
    comma_count = len(re.findall(r"[，,、]", normalized))
    if comma_count >= 2:
        return False
    return True



def _is_text_heading_fallback(text, max_heading_chars=64):
    t = _normalize_heading_text(text)
    if not t or len(t) > max_heading_chars:
        return False
    if re.search(r"[。；;？！?!：:]\s*$", t):
        return False
    normalized = re.sub(r"\s+", " ", t).strip()
    if _TEXT_CHAPTER_HEAD_RE.match(normalized):
        return True
    if _is_short_parenthetical_heading(normalized, max_heading_chars=max_heading_chars):
        return True
    match = _match_heading_number_relaxed(normalized)
    if not match:
        return False
    if len(re.findall(r"[，,、]", normalized)) >= 2:
        return False
    if re.search(r"[：:]", normalized):
        return False
    return True

def _should_merge_with_previous(prev_text, next_text):
    if not prev_text or not next_text:
        return False
    if _starts_with_marker(next_text):
        return False
    prev_text = prev_text.rstrip()
    if re.search(r"[。！？!?]$", prev_text):
        return False
    return True



def _chunk_first_page(doc):
    pages = doc.get("page_num_int") or []
    return pages[0] if pages else None


def _chunk_last_page(doc):
    pages = doc.get("page_num_int") or []
    return pages[-1] if pages else None


def _chunk_first_top(doc):
    tops = doc.get("top_int") or []
    return tops[0] if tops else None


def _merge_doc_positions(target, source):
    for key in ("page_num_int", "position_int", "top_int"):
        source_values = source.get(key) or []
        if not source_values:
            continue
        merged_values = list(target.get(key) or [])
        merged_values.extend(source_values)
        target[key] = merged_values


def _join_continuation_text(prev_text, current_text):
    prev_text = (prev_text or "").rstrip()
    current_text = (current_text or "").lstrip()
    if not prev_text:
        return current_text
    if not current_text:
        return prev_text
    if prev_text.endswith(("-", "/")):
        return f"{prev_text}{current_text}"
    if re.search(r"[\u4e00-\u9fffA-Za-z0-9]$", prev_text) and re.match(r"^[\u4e00-\u9fffA-Za-z0-9]", current_text):
        return f"{prev_text}{current_text}"
    return f"{prev_text} {current_text}"


def _is_short_page_continuation(prev_doc, current_doc, next_doc=None, max_chars=64, max_tokens=40, header_top_threshold=120):
    if not prev_doc or not current_doc:
        return False
    if prev_doc.get("doc_type_kwd") or current_doc.get("doc_type_kwd"):
        return False
    if prev_doc.get("docnm_kwd") != current_doc.get("docnm_kwd"):
        return False

    prev_title = _clean_text(prev_doc.get("title_tks"))
    current_title = _clean_text(current_doc.get("title_tks"))
    current_text = _clean_text(current_doc.get("content_with_weight"))
    prev_text = _clean_text(prev_doc.get("content_with_weight"))
    if not current_text or not prev_text:
        return False
    if _starts_with_marker(current_text):
        return False
    if len(current_text) > max_chars or _token_count(current_text) > max_tokens:
        return False

    current_top = _chunk_first_top(current_doc)
    prev_last_page = _chunk_last_page(prev_doc)
    current_first_page = _chunk_first_page(current_doc)
    if prev_last_page is None or current_first_page is None:
        return False
    if current_first_page not in {prev_last_page, prev_last_page + 1}:
        return False

    if re.search(r"[。！？!?；;:：]\s*$", prev_text):
        return False

    page_top_case = current_first_page == prev_last_page + 1 and current_top is not None and current_top <= header_top_threshold
    same_title_case = bool(prev_title and current_title and prev_title == current_title)
    title_shifted = False
    if next_doc:
        next_text = _clean_text(next_doc.get("content_with_weight"))
        next_title = _clean_text(next_doc.get("title_tks"))
        title_shifted = bool(next_title and next_title != current_title and next_title != prev_title)
        if next_text and not _starts_with_marker(next_text) and not title_shifted and not same_title_case:
            return False

    if not (page_top_case or same_title_case or title_shifted):
        return False

    return True


def _merge_short_page_continuations(results, eng):
    if not results:
        return results

    merged = []
    index = 0
    while index < len(results):
        current = results[index]
        previous = merged[-1] if merged else None
        next_doc = results[index + 1] if index + 1 < len(results) else None

        if _is_short_page_continuation(previous, current, next_doc=next_doc):
            merged_text = _join_continuation_text(
                previous.get("content_with_weight"),
                current.get("content_with_weight"),
            )
            tokenize(previous, merged_text, eng)
            _merge_doc_positions(previous, current)
            index += 1
            continue

        merged.append(current)
        index += 1

    return merged


def _split_inline_blocks(text):
    text = _clean_text(text)
    if not text:
        return []
    lines = []
    for raw_line in re.split(r"\n+", text):
        line = raw_line.strip()
        if not line:
            continue
        # More conservative than the original splitter:
        # keep multi-level numeric headings (4.1 / 4.6.9) and parenthetical clause titles,
        # but stop splitting on bare '2.' markers that often appear inside prose.
        line = re.sub(r"\s+(?=(?:\d{1,2}(?:\.\d+){1,7})(?:\s|[^\d.]))", "\n", line)
        line = re.sub(r"\s+(?=(?:[（(][0-9一二三四五六七八九十百千万零〇两]+[)）]|\d+[)）])(?:\s|[^\d]))", "\n", line)
        parts = [part.strip() for part in re.split(r"\n+", line) if part and part.strip()]
        lines.extend(parts)
    return lines

def _split_sentences(text):
    if not text:
        return []
    pieces = _SENTENCE_SPLIT_RE.split(text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def _split_long_unit(text, max_tokens):
    if not text:
        return []
    if max_tokens <= 0 or _token_count(text) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p and p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current_parts = []
    current_tokens = 0

    def flush_current():
        nonlocal current_parts, current_tokens
        chunk = "\n\n".join(current_parts).strip()
        if chunk:
            chunks.append(chunk)
        current_parts = []
        current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = _token_count(paragraph)
        if para_tokens <= max_tokens:
            if current_parts and current_tokens + para_tokens > max_tokens:
                flush_current()
            current_parts.append(paragraph)
            current_tokens += para_tokens
            continue

        sentences = _split_sentences(paragraph)
        if not sentences:
            sentences = [paragraph]

        for sentence in sentences:
            sent_tokens = _token_count(sentence)
            if sent_tokens > max_tokens:
                # last resort: hard split by characters while preserving rough order
                step = max(20, int(max_tokens * 2))
                for i in range(0, len(sentence), step):
                    piece = sentence[i:i + step].strip()
                    if not piece:
                        continue
                    piece_tokens = _token_count(piece)
                    if current_parts and current_tokens + piece_tokens > max_tokens:
                        flush_current()
                    current_parts.append(piece)
                    current_tokens += piece_tokens
                continue

            if current_parts and current_tokens + sent_tokens > max_tokens:
                flush_current()
            current_parts.append(sentence)
            current_tokens += sent_tokens

    if current_parts:
        flush_current()
    return chunks


def _expand_units(units, max_tokens):
    expanded = []
    for unit in units:
        text = unit.get("text", "").strip()
        if not text:
            continue
        if _token_count(text) <= max_tokens:
            expanded.append({"text": text, "positions": unit.get("positions") or []})
            continue
        for piece in _split_long_unit(text, max_tokens):
            expanded.append({"text": piece, "positions": unit.get("positions") or []})
    return expanded


def _chunk_units(units, max_tokens, overlap_tokens):
    if not units:
        return []
    units = _expand_units(units, max_tokens)
    chunks = []
    current = []
    current_tokens = 0

    def flush_current():
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(item["text"] for item in current if item.get("text")).strip()
        if text and (not chunks or text != chunks[-1]["text"]):
            chunks.append(
                {
                    "text": text,
                    "positions": _merge_positions([item.get("positions") or [] for item in current]),
                }
            )
        overlap = []
        if overlap_tokens > 0:
            token_budget = 0
            for item in reversed(current):
                item_tokens = _token_count(item["text"])
                if token_budget + item_tokens > overlap_tokens and overlap:
                    break
                overlap.insert(0, item)
                token_budget += item_tokens
        current = overlap
        current_tokens = sum(_token_count(item["text"]) for item in current)

    for item in units:
        item_tokens = _token_count(item["text"])
        if current and current_tokens + item_tokens > max_tokens:
            flush_current()
        current.append(item)
        current_tokens += item_tokens

    if current:
        text = "\n\n".join(item["text"] for item in current if item.get("text")).strip()
        if text and (not chunks or text != chunks[-1]["text"]):
            chunks.append(
                {
                    "text": text,
                    "positions": _merge_positions([item.get("positions") or [] for item in current]),
                }
            )
    return chunks


def _group_parts_by_page(partitions):
    pages = defaultdict(list)
    page_bottoms = defaultdict(int)
    for part in partitions:
        positions = _get_positions(part)
        page_no = _page_number_from_positions(positions)
        if page_no is None:
            continue
        pages[page_no].append(part)
        for _, _, _, _, bottom in positions:
            page_bottoms[page_no] = max(page_bottoms[page_no], bottom)
    return pages, page_bottoms


def _toc_entry_score(text):
    raw = _clean_text(text)
    if not raw:
        return 0
    if raw == "目录":
        return 3
    if re.search(r"(?:\.{2,}|…+)\s*\d+$", raw):
        return 2
    if re.match(r"^(?:第[一二三四五六七八九十百千万零〇两]+章|附录\s*[0-9一二三四五六七八九十]+)\s+.*\d+$", raw):
        return 2
    if re.match(r"^\d+(?:\.\d+)+\s+.*\d+$", raw):
        return 2
    return 0


def _detect_toc_pages(partitions, max_scan_pages=15):
    pages, _ = _group_parts_by_page(partitions)
    toc_pages = set()
    toc_start = None
    for page_no in sorted(pages):
        if page_no > max_scan_pages:
            break
        page_texts = [_clean_text((part.get("text") or "") if (part.get("type") or "").lower() != "title" else (part.get("text") or "")) for part in pages[page_no]]
        if any(text == "目录" for text in page_texts if text):
            toc_start = page_no
            break
    if toc_start is None:
        return toc_pages

    for page_no in sorted(pages):
        if page_no < toc_start:
            continue
        page_parts = pages[page_no]
        scores = 0
        long_body = 0
        for part in page_parts:
            text = _clean_text(part.get("text") or "")
            if not text:
                continue
            scores += _toc_entry_score(text)
            if len(text) >= 150 and _toc_entry_score(text) == 0:
                long_body += 1
        if page_no == toc_start and scores >= 3:
            toc_pages.add(page_no)
            continue
        if scores >= 2 and long_body == 0:
            toc_pages.add(page_no)
            continue
        break
    return toc_pages



def _detect_repeated_boilerplate(partitions, min_repeat_pages=3):
    pages, page_bottoms = _group_parts_by_page(partitions)
    occurrences = defaultdict(set)
    for page_no, page_parts in pages.items():
        page_bottom = page_bottoms.get(page_no) or 800
        header_threshold = min(max(int(page_bottom * 0.12), 55), 90)
        footer_threshold = max(int(page_bottom * 0.88), page_bottom - 80)
        for part in page_parts:
            if (part.get("type") or "").lower() != "text":
                continue
            text = _clean_text(part.get("text"))
            if not text or len(text) > 60:
                continue
            # Repeated chapter headings are semantically important.
            if _is_chapter_heading_text(text):
                continue
            positions = _get_positions(part)
            if not positions:
                continue
            _, _, _, top, _ = positions[0]
            if top <= header_threshold or top >= footer_threshold:
                occurrences[_normalized_repeat_key(text)].add(page_no)
    return {key for key, pageset in occurrences.items() if key and len(pageset) >= min_repeat_pages}


def _is_boilerplate_text(part, repeated_boilerplate_keys, keep_cover_page=True):
    if (part.get("type") or "").lower() != "text":
        return False
    text = _clean_text(part.get("text"))
    if not text:
        return True
    if _is_chapter_heading_text(text):
        return False
    positions = _get_positions(part)
    page_no = _page_number_from_positions(positions)
    if keep_cover_page and page_no in {1, 2}:
        return False
    key = _normalized_repeat_key(text)
    return key in repeated_boilerplate_keys


def _detect_repeated_title_boilerplate(partitions, min_repeat_pages=3):
    pages, page_bottoms = _group_parts_by_page(partitions)
    occurrences = defaultdict(set)
    for page_no, page_parts in pages.items():
        page_bottom = page_bottoms.get(page_no) or 800
        header_threshold = min(max(int(page_bottom * 0.12), 55), 90)
        footer_threshold = max(int(page_bottom * 0.88), page_bottom - 80)
        for part in page_parts:
            if (part.get("type") or "").lower() != "title":
                continue
            text = _clean_text(part.get("text"))
            if not text or len(text) > 80:
                continue
            positions = _get_positions(part)
            if not positions:
                continue
            _, _, _, top, _ = positions[0]
            if top <= header_threshold or top >= footer_threshold:
                occurrences[_normalized_repeat_key(text)].add(page_no)
    return {key for key, pageset in occurrences.items() if key and len(pageset) >= min_repeat_pages}


def _is_repeated_boilerplate_title(part, repeated_title_boilerplate_keys, headings, keep_cover_page=True):
    if (part.get("type") or "").lower() != "title":
        return False
    text = _normalize_heading_text(part.get("text"))
    if not text:
        return False
    positions = _get_positions(part)
    page_no = _page_number_from_positions(positions)
    if keep_cover_page and page_no in {1, 2}:
        return False
    if _normalized_repeat_key(text) not in repeated_title_boilerplate_keys:
        return False
    previous_level = headings[-1]["level"] if headings else 0
    level = _heading_level_from_text(text, fallback_level=part.get("title_type"), previous_level=previous_level)
    return _has_same_active_heading(headings, level, text)

def _push_heading(headings, level, text):
    heading_text = _normalize_heading_text(text)
    if not heading_text:
        return
    while headings and headings[-1]["level"] >= level:
        headings.pop()
    headings.append({"level": level, "text": heading_text})


def custom_parse(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, **kwargs):
    """
    Parse partitioned article JSON and chunk it into RAGFlow-compatible docs.

    Improvements over the original implementation:
    1. chunk_token_num is interpreted as real token count instead of string length.
    2. table/image/text chunks only keep positions belonging to the current chunk.
    3. repeated headers/footers and TOC pages can be stripped automatically.
    4. title hierarchy is inferred from heading text when title_type is unreliable.
    5. numbered clauses inside Text partitions are split before packing into chunks.
    """

    logging.info("Custom parsing %s", filename)
    if callback:
        callback(0.1, "Loading structured article JSON.")

    parser_config = kwargs.get("parser_config", {}) or {}
    payload = _load_payload(filename, binary, parser_config=parser_config, callback=callback)
    partitions = payload.get("partitions") or []
    if not partitions:
        logging.warning("No partitions found in custom parser input.")
        return []

    max_tokens = max(64, int(parser_config.get("chunk_token_num", 1024) or 1024))
    overlap_percent = max(0, min(50, int(parser_config.get("overlapped_percent", 10) or 10)))
    overlap_tokens = int(max_tokens * overlap_percent / 100)
    strip_toc = str(parser_config.get("strip_toc", "true")).lower() != "false"
    strip_repeated_boilerplate = str(parser_config.get("strip_repeated_boilerplate", "true")).lower() != "false"
    include_heading_in_chunk = str(parser_config.get("include_heading_in_chunk", "true")).lower() != "false"
    keep_cover_page_boilerplate = str(parser_config.get("keep_cover_page_boilerplate", "false")).lower() != "false"
    max_heading_chars = max(24, int(parser_config.get("max_heading_chars", 64) or 64))
    eng = lang.lower() == "english"

    doc = _doc_meta(filename)
    toc_pages = _detect_toc_pages(partitions) if strip_toc else set()
    repeated_boilerplate_keys = _detect_repeated_boilerplate(partitions) if strip_repeated_boilerplate else set()
    repeated_title_boilerplate_keys = _detect_repeated_title_boilerplate(partitions) if strip_repeated_boilerplate else set()
    page_heading_candidates = _collect_page_heading_candidates(partitions)

    results = []
    table_items = []
    headings = []
    section_units = []

    def flush_section():
        nonlocal section_units, results
        if not section_units:
            return
        heading_path = _heading_path(headings)
        for chunk in _chunk_units(section_units, max_tokens, overlap_tokens):
            results.append(
                _build_text_doc(
                    doc,
                    chunk["text"],
                    eng,
                    chunk["positions"],
                    heading_path=heading_path,
                    include_heading=include_heading_in_chunk,
                )
            )
        section_units = []

    def apply_heading(heading_text, fallback_level, page_no):
        heading_text = _normalize_heading_text(heading_text)
        if not heading_text or heading_text == "目录":
            return False
        previous_level = headings[-1]["level"] if headings else 0
        level = _heading_level_from_text(heading_text, fallback_level=fallback_level, previous_level=previous_level)
        if _PAREN_HEAD_RE.match(_normalize_heading_for_match(heading_text)):
            level = _resolve_parenthetical_level(headings, fallback_level=fallback_level)

        simulated = [dict(item) for item in headings]
        _reconcile_heading_context(simulated, heading_text, page_no, page_heading_candidates)
        duplicate = _has_same_active_heading(simulated, level, heading_text)

        if duplicate and simulated == headings:
            return True

        flush_section()
        headings[:] = simulated
        if not _has_same_active_heading(headings, level, heading_text):
            _push_heading(headings, level, heading_text)
        return True

    if callback:
        callback(0.2, "Detecting TOC pages and repeated page headers/footers.")

    for index, part in enumerate(partitions):
        positions = _get_positions(part)
        part_pages = [pos[0] for pos in positions] if positions else []
        if part_pages and (max(part_pages) < from_page or min(part_pages) > to_page):
            continue

        page_no = _page_number_from_positions(positions)
        if strip_toc and page_no in toc_pages:
            continue
        if strip_repeated_boilerplate and _is_boilerplate_text(part, repeated_boilerplate_keys, keep_cover_page=keep_cover_page_boilerplate):
            continue

        part_type = (part.get("type") or "").strip().lower()
        text = _clean_text(part.get("text"))
        caption = _clean_text(part.get("caption"))
        data = _clean_text(part.get("data"))

        if part_type == "title":
            if strip_repeated_boilerplate and _is_repeated_boilerplate_title(
                part,
                repeated_title_boilerplate_keys,
                headings,
                keep_cover_page=keep_cover_page_boilerplate,
            ):
                continue
            if apply_heading(text, part.get("title_type"), page_no):
                continue

        if part_type == "text":
            if not text:
                continue
            # 过滤页码污染：如 "— 83 —" 或 "91"
            if _is_page_number(text):
                continue
            blocks = _split_inline_blocks(text)
            if not blocks:
                continue
            for block in blocks:
                block = _clean_text(block)
                if not block:
                    continue
                # 过滤页码污染
                if _is_page_number(block):
                    continue
                if _is_text_heading_fallback(block, max_heading_chars=max_heading_chars):
                    apply_heading(block, None, page_no)
                    continue
                if section_units and _should_merge_with_previous(section_units[-1]["text"], block):
                    section_units[-1]["text"] = f"{section_units[-1]['text']} {block}".strip()
                    section_units[-1]["positions"] = _merge_positions([section_units[-1]["positions"], positions])
                else:
                    section_units.append({"text": block, "positions": positions})
            continue

        if part_type == "table":
            flush_section()
            heading_path = _heading_path(headings)
            table_text = _format_table_content(heading_path, caption, text, data)
            table_items.append(((None, table_text), positions))
            continue

        if part_type in {"image", "chart"}:
            flush_section()
            heading_path = _heading_path(headings)
            media_parts = []
            if heading_path:
                media_parts.append(f"Section: {heading_path}")
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

        if text:
            blocks = _split_inline_blocks(text)
            for block in blocks:
                block = _clean_text(block)
                if not block:
                    continue
                if section_units and _should_merge_with_previous(section_units[-1]["text"], block):
                    section_units[-1]["text"] = f"{section_units[-1]['text']} {block}".strip()
                    section_units[-1]["positions"] = _merge_positions([section_units[-1]["positions"], positions])
                else:
                    section_units.append({"text": block, "positions": positions})
        elif caption or data:
            flush_section()
            fallback = "\n".join(item for item in [caption, data] if item).strip()
            if fallback:
                results.append(_build_text_doc(doc, fallback, eng, positions, _heading_path(headings), include_heading=include_heading_in_chunk))

    flush_section()

    if table_items:
        results.extend(tokenize_table(table_items, doc, eng))

    results = _merge_short_page_continuations(results, eng)

    if callback:
        callback(0.9, f"Custom parsing completed with {len(results)} chunks.")
    return results


chunk = custom_parse
