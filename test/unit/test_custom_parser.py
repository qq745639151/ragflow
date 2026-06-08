import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - allows direct script execution without pytest
    class _PytestShim:
        @staticmethod
        def fixture(*args, **kwargs):
            def _decorator(func):
                return func

            return _decorator

        @staticmethod
        def skip(message):
            raise SystemExit(message)

    pytest = _PytestShim()


DEFAULT_INPUT_FILE = r"D:\Develop\ragflow_clean\ragflow\test320\output\2026年国调直调安全自动装置调度运行管理规定（第一版).json"
DEFAULT_OUTPUT_DIR = r"D:\Develop\ragflow_clean\ragflow\test320"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.app.custom_parser import custom_parse


def _resolve_input_path() -> Path | None:
    env_path = os.getenv("CUSTOM_PARSER_INPUT")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
        return None

    if DEFAULT_INPUT_FILE:
        candidate = Path(DEFAULT_INPUT_FILE)
        if candidate.exists():
            return candidate
        return None

    candidates = list((_project_root() / "test320" / "output").glob("*.json"))
    return candidates[0] if candidates else None


def _resolve_output_dir() -> Path:
    env_dir = os.getenv("CUSTOM_PARSER_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir)

    if DEFAULT_OUTPUT_DIR:
        return Path(DEFAULT_OUTPUT_DIR)

    return _project_root() / "test320"


def _build_output_path(input_path: Path) -> Path:
    output_dir = _resolve_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{input_path.stem}_custom_parser_output_{timestamp}.json"


@pytest.fixture(scope="module")
def custom_parser_input_path() -> Path:
    path = _resolve_input_path()
    if path is None:
        pytest.skip("custom parser input json not found; set CUSTOM_PARSER_INPUT to a valid json path")
    return path


@pytest.fixture(scope="module")
def parsed_custom_chunks(custom_parser_input_path: Path):
    chunks = custom_parse(
        str(custom_parser_input_path),
        parser_config={
            "chunk_token_num": 256,
            "overlapped_percent": 10,
            "strip_toc": "true",
            "strip_repeated_boilerplate": "true",
            "include_heading_in_chunk": "true",
            "keep_cover_page_boilerplate": "false",
        },
    )
    output_path = _build_output_path(custom_parser_input_path)
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks


def test_custom_parser_returns_chunks(parsed_custom_chunks):
    assert parsed_custom_chunks
    assert isinstance(parsed_custom_chunks, list)


def test_custom_parser_chunk_structure(parsed_custom_chunks):
    for chunk in parsed_custom_chunks[:50]:
        assert chunk.get("content_with_weight")

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


def test_custom_parser_media_chunks_have_expected_shape(parsed_custom_chunks):
    media_chunks = [ck for ck in parsed_custom_chunks if ck.get("doc_type_kwd") in {"image", "chart", "table"}]
    if not media_chunks:
        pytest.skip("no media chunks found in current custom parser input")

    for chunk in media_chunks[:20]:
        content = chunk.get("content_with_weight") or ""
        assert content
        assert chunk.get("doc_type_kwd") in {"image", "chart", "table"}

        if chunk.get("doc_type_kwd") == "table":
            assert "Caption:" in content or "Summary:" in content or "<table" in content.lower() or "Image path:" in content
        else:
            assert "Caption:" in content or "Summary:" in content or "Image path:" in content


def test_custom_parser_output_file_is_created(custom_parser_input_path: Path, parsed_custom_chunks):
    output_dir = _resolve_output_dir()
    matches = sorted(output_dir.glob(f"{custom_parser_input_path.stem}_custom_parser_output_*.json"))
    assert matches
    latest = matches[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == len(parsed_custom_chunks)


def main():
    input_path = _resolve_input_path()
    if input_path is None:
        raise FileNotFoundError(
            "custom parser input json not found; set DEFAULT_INPUT_FILE or CUSTOM_PARSER_INPUT to a valid json path"
        )

    chunks = custom_parse(
        str(input_path),
        parser_config={
            "chunk_token_num": 256,
            "overlapped_percent": 10,
            "strip_toc": "true",
            "strip_repeated_boilerplate": "true",
            "include_heading_in_chunk": "true",
            "keep_cover_page_boilerplate": "false",
        },
    )
    output_path = _build_output_path(input_path)
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"input: {input_path}")
    print(f"chunks: {len(chunks)}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
