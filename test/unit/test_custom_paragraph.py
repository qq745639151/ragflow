import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from rag.app.custom_paragraph import custom_parse


SPEECH_TITLE = "在国网西北分部安全生产工作会暨2025年安委会第一次会议上的讲话"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_input_path() -> Path | None:
    env_path = os.getenv("CUSTOM_PARAGRAPH_INPUT")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
        return None

    candidates = list((_project_root() / "test320" / "output").glob("2025*.json"))
    return candidates[0] if candidates else None


def _resolve_output_dir() -> Path:
    env_dir = os.getenv("CUSTOM_PARAGRAPH_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir)
    return _project_root() / "test320" / "chunk_result"


def _build_output_path(input_path: Path) -> Path:
    output_dir = _resolve_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{input_path.stem}_paragraph_chunks_{timestamp}.json"


@pytest.fixture(scope="module")
def speech_sample_path() -> Path:
    path = _resolve_input_path()
    if path is None:
        pytest.skip("speech sample json not found; set CUSTOM_PARAGRAPH_INPUT to a valid json path")
    return path


@pytest.fixture(scope="module")
def parsed_speech_chunks(speech_sample_path: Path):
    chunks = custom_parse(
        str(speech_sample_path),
        parser_config={
            "chunk_token_num": 256,
            "overlapped_percent": 10,
            "strip_toc": "true",
            "strip_repeated_boilerplate": "true",
            "include_title_in_chunk": "true",
            "include_title_paragraphs": "true",
        },
    )
    output_path = _build_output_path(speech_sample_path)
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks


def test_custom_paragraph_returns_chunks(parsed_speech_chunks):
    assert parsed_speech_chunks
    assert isinstance(parsed_speech_chunks, list)
    assert len(parsed_speech_chunks) > 20


def test_custom_paragraph_first_chunk_starts_from_real_speech(parsed_speech_chunks):
    first = parsed_speech_chunks[0]
    title = first.get("title_tks") or ""
    content = first.get("content_with_weight") or ""

    assert "目录" not in content
    assert SPEECH_TITLE in title
    assert SPEECH_TITLE in content
    assert "STATE GRID" not in content


def test_custom_paragraph_chunk_structure(parsed_speech_chunks):
    for chunk in parsed_speech_chunks[:20]:
        assert chunk.get("content_with_weight")
        assert chunk.get("title_tks")

        page_nums = chunk.get("page_num_int") or []
        positions = chunk.get("position_int") or []
        tops = chunk.get("top_int") or []

        assert page_nums
        assert positions
        assert tops
        assert len(page_nums) == len(positions) == len(tops)

        for page_num, pos, top in zip(page_nums, positions, tops):
            assert len(pos) == 5
            assert pos[0] == page_num
            assert pos[3] == top
            assert pos[1] <= pos[2]
            assert pos[3] <= pos[4]


def test_custom_paragraph_can_hide_title_paragraphs(speech_sample_path: Path):
    chunks = custom_parse(
        str(speech_sample_path),
        parser_config={
            "chunk_token_num": 256,
            "overlapped_percent": 10,
            "strip_toc": "true",
            "strip_repeated_boilerplate": "true",
            "include_title_in_chunk": "true",
            "include_title_paragraphs": "false",
        },
    )

    assert chunks
    first_content = chunks[0].get("content_with_weight") or ""
    assert "同志们：" not in first_content
    assert SPEECH_TITLE in first_content


def test_custom_paragraph_output_file_is_created(speech_sample_path: Path, parsed_speech_chunks):
    output_dir = _resolve_output_dir()
    matches = sorted(output_dir.glob(f"{speech_sample_path.stem}_paragraph_chunks_*.json"))
    assert matches
    latest = matches[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == len(parsed_speech_chunks)
