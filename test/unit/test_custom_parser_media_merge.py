import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.app.custom_parser import custom_parse


def _part(part_type, text="", caption="", data="", image_path="", page=1, bbox=None):
    bbox = bbox or [10, 20, 110, 40]
    return {
        "type": part_type,
        "sub_type": part_type,
        "title_type": -1,
        "caption": caption,
        "text": text,
        "data": data,
        "image_path": image_path,
        "metadata": {
            "extra_data": {
                "types": [part_type],
                "pages": [page],
                "bboxes": [bbox],
            }
        },
    }


def _parse(partitions):
    payload = json.dumps({"partitions": partitions}, ensure_ascii=False).encode("utf-8")
    return custom_parse(
        "media.json",
        binary=payload,
        parser_config={
            "chunk_token_num": 2048,
            "overlapped_percent": 0,
            "strip_toc": "false",
            "strip_repeated_boilerplate": "false",
            "include_heading_in_chunk": "false",
        },
    )


def test_captioned_table_remains_separate_without_summary_text():
    chunks = _parse(
        [
            _part("text", text="before table", bbox=[10, 20, 100, 30]),
            _part(
                "Table",
                caption="Table 1",
                text="unwanted table summary",
                data="<table><tr><td>A</td></tr></table>",
                image_path="http://example/table.jpg",
                bbox=[10, 40, 100, 80],
            ),
        ]
    )

    assert len(chunks) == 2
    table_chunk = chunks[1]
    content = table_chunk["content_with_weight"]
    assert table_chunk["doc_type_kwd"] == "table"
    assert "Caption: Table 1" in content
    assert "<table><tr><td>A</td></tr></table>" in content
    assert "Image path: http://example/table.jpg" in content
    assert "Summary:" not in content
    assert "unwanted table summary" not in content


def test_uncaptioned_table_merges_into_previous_chunk_without_summary_text():
    chunks = _parse(
        [
            _part("text", text="section body", bbox=[10, 20, 100, 30]),
            _part(
                "Table",
                text="unwanted table summary",
                data="<table><tr><td>B</td></tr></table>",
                image_path="http://example/no-caption-table.jpg",
                bbox=[10, 40, 100, 80],
            ),
        ]
    )

    assert len(chunks) == 1
    content = chunks[0]["content_with_weight"]
    assert chunks[0]["doc_type_kwd"] == "table"
    assert "section body" in content
    assert "<table><tr><td>B</td></tr></table>" in content
    assert "Image path: http://example/no-caption-table.jpg" in content
    assert "Summary:" not in content
    assert "unwanted table summary" not in content
    assert chunks[0]["position_int"] == [[1, 10, 100, 20, 30], [1, 10, 100, 40, 80]]


def test_image_and_chart_merge_into_previous_chunk_with_existing_content_shape():
    chunks = _parse(
        [
            _part("text", text="image lead", bbox=[10, 20, 100, 30]),
            _part(
                "Image",
                caption="Figure 1",
                text="image description",
                image_path="http://example/image.jpg",
                bbox=[10, 40, 100, 80],
            ),
            _part(
                "Chart",
                caption="Chart 1",
                text="chart description",
                image_path="http://example/chart.jpg",
                bbox=[10, 90, 100, 130],
            ),
        ]
    )

    assert len(chunks) == 1
    content = chunks[0]["content_with_weight"]
    assert chunks[0]["doc_type_kwd"] == "chart"
    assert "image lead" in content
    assert "Caption: Figure 1" in content
    assert "Summary: image description" in content
    assert "Image path: http://example/image.jpg" in content
    assert "Caption: Chart 1" in content
    assert "Summary: chart description" in content
    assert "Image path: http://example/chart.jpg" in content
    assert chunks[0]["position_int"] == [
        [1, 10, 100, 20, 30],
        [1, 10, 100, 40, 80],
        [1, 10, 100, 90, 130],
    ]


def test_second_uncaptioned_table_becomes_separate_chunk():
    chunks = _parse(
        [
            _part("text", text="section body", bbox=[10, 20, 100, 30]),
            _part(
                "Table",
                data="<table><tr><td>first</td></tr></table>",
                image_path="http://example/first-table.jpg",
                bbox=[10, 40, 100, 80],
            ),
            _part(
                "Table",
                data="<table><tr><td>second</td></tr></table>",
                image_path="http://example/second-table.jpg",
                bbox=[10, 90, 100, 130],
            ),
        ]
    )

    assert len(chunks) == 2
    first_content = chunks[0]["content_with_weight"]
    second_content = chunks[1]["content_with_weight"]
    assert chunks[0]["doc_type_kwd"] == "table"
    assert chunks[1]["doc_type_kwd"] == "table"
    assert "section body" in first_content
    assert "first</td>" in first_content
    assert "second</td>" not in first_content
    assert "second</td>" in second_content
    assert chunks[0]["position_int"] == [[1, 10, 100, 20, 30], [1, 10, 100, 40, 80]]
    assert chunks[1]["position_int"] == [[1, 10, 100, 90, 130]]


def test_second_image_becomes_separate_chunk():
    chunks = _parse(
        [
            _part("text", text="image lead", bbox=[10, 20, 100, 30]),
            _part(
                "Image",
                caption="Figure 1",
                text="first image",
                image_path="http://example/first-image.jpg",
                bbox=[10, 40, 100, 80],
            ),
            _part(
                "Image",
                caption="Figure 2",
                text="second image",
                image_path="http://example/second-image.jpg",
                bbox=[10, 90, 100, 130],
            ),
        ]
    )

    assert len(chunks) == 2
    first_content = chunks[0]["content_with_weight"]
    second_content = chunks[1]["content_with_weight"]
    assert chunks[0]["doc_type_kwd"] == "image"
    assert chunks[1]["doc_type_kwd"] == "image"
    assert "image lead" in first_content
    assert "Summary: first image" in first_content
    assert "Summary: second image" not in first_content
    assert "Summary: second image" in second_content
    assert chunks[0]["position_int"] == [[1, 10, 100, 20, 30], [1, 10, 100, 40, 80]]
    assert chunks[1]["position_int"] == [[1, 10, 100, 90, 130]]


def test_table_and_image_can_share_one_chunk_before_second_table_splits():
    chunks = _parse(
        [
            _part("text", text="mixed lead", bbox=[10, 20, 100, 30]),
            _part(
                "Table",
                data="<table><tr><td>table one</td></tr></table>",
                image_path="http://example/table-one.jpg",
                bbox=[10, 40, 100, 80],
            ),
            _part(
                "Image",
                caption="Figure 1",
                text="image one",
                image_path="http://example/image-one.jpg",
                bbox=[10, 90, 100, 130],
            ),
            _part(
                "Table",
                data="<table><tr><td>table two</td></tr></table>",
                image_path="http://example/table-two.jpg",
                bbox=[10, 140, 100, 180],
            ),
        ]
    )

    assert len(chunks) == 2
    first_content = chunks[0]["content_with_weight"]
    second_content = chunks[1]["content_with_weight"]
    assert chunks[0]["doc_type_kwd"] == "image"
    assert chunks[1]["doc_type_kwd"] == "table"
    assert "mixed lead" in first_content
    assert "table one</td>" in first_content
    assert "Summary: image one" in first_content
    assert "table two</td>" not in first_content
    assert "table two</td>" in second_content
    assert chunks[0]["position_int"] == [
        [1, 10, 100, 20, 30],
        [1, 10, 100, 40, 80],
        [1, 10, 100, 90, 130],
    ]
    assert chunks[1]["position_int"] == [[1, 10, 100, 140, 180]]


def test_leading_media_without_previous_chunk_is_preserved():
    chunks = _parse(
        [
            _part(
                "Image",
                caption="Leading figure",
                text="leading image description",
                image_path="http://example/leading.jpg",
            )
        ]
    )

    assert len(chunks) == 1
    assert chunks[0]["doc_type_kwd"] == "image"
    assert "Caption: Leading figure" in chunks[0]["content_with_weight"]
