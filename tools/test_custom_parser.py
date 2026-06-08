#!/usr/bin/env python3
#
# Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.app.custom_parser import custom_parse  # noqa: E402


DEFAULT_RUN_CONFIG: dict[str, Any] = {
    # Fill these values when you want to run this script directly in the IDE
    # without passing command-line arguments. CLI arguments still take priority.
    "input": str(PROJECT_ROOT / "test320" / "output" / "《西北电网稳定运行规定（2026第一版）》.json"),
    "output": str(PROJECT_ROOT / "test320"),
    "parser_config_file": "",
    "parser_config_json": "",
    "from_page": 0,
    "to_page": 100000,
    "lang": "Chinese",
    "no_callback": False,
    "summary": True,
}


def _load_parser_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.parser_config_file:
        return json.loads(Path(args.parser_config_file).read_text(encoding="utf-8"))
    if args.parser_config_json:
        return json.loads(args.parser_config_json)
    return {}


def _resolve_output_path(output_arg: str, input_path: Path) -> Path:
    output_path = Path(output_arg)
    if output_path.exists() and output_path.is_dir():
        return output_path / f"{input_path.stem}_custom_parser_output.json"
    if not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f"{input_path.stem}_custom_parser_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _progress_callback(progress: float, message: str) -> None:
    print(f"[{progress:>4.0%}] {message}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rag.app.custom_parser.custom_parse and export the chunks to JSON."
    )
    parser.add_argument(
        "--input",
        help="Input file path. Supports local JSON payloads or source files such as PDF passed as binary.",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path, or an output directory for auto-generated file naming.",
    )
    parser.add_argument(
        "--parser-config-file",
        help="Path to a JSON file containing parser_config.",
    )
    parser.add_argument(
        "--parser-config-json",
        help="Inline parser_config JSON string.",
    )
    parser.add_argument(
        "--from-page",
        type=int,
        default=0,
        help="Start page index, inclusive. Default: 0.",
    )
    parser.add_argument(
        "--to-page",
        type=int,
        default=100000,
        help="End page index, inclusive. Default: 100000.",
    )
    parser.add_argument(
        "--lang",
        default="Chinese",
        help='Document language passed to custom_parse. Default: "Chinese".',
    )
    parser.add_argument(
        "--no-callback",
        action="store_true",
        help="Disable progress logging callback.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a short summary of the parsed chunks after export.",
    )
    return parser


def _apply_default_run_config(args: argparse.Namespace) -> argparse.Namespace:
    for key, value in DEFAULT_RUN_CONFIG.items():
        current = getattr(args, key, None)
        if current in (None, ""):
            setattr(args, key, value)
    if not args.summary and DEFAULT_RUN_CONFIG.get("summary"):
        args.summary = True
    return args


def main() -> int:
    parser = build_arg_parser()
    args = _apply_default_run_config(parser.parse_args())

    if not args.input:
        parser.error(
            "missing input path. Pass --input or set DEFAULT_RUN_CONFIG['input'] in tools/test_custom_parser.py"
        )
    if not args.output:
        parser.error(
            "missing output path. Pass --output or set DEFAULT_RUN_CONFIG['output'] in tools/test_custom_parser.py"
        )

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"input file does not exist: {input_path}")

    output_path = _resolve_output_path(args.output, input_path)
    parser_config = _load_parser_config(args)

    binary = None
    if input_path.suffix.lower() != ".json":
        binary = input_path.read_bytes()

    chunks = custom_parse(
        str(input_path),
        binary=binary,
        from_page=args.from_page,
        to_page=args.to_page,
        lang=args.lang,
        callback=None if args.no_callback else _progress_callback,
        parser_config=parser_config,
    )

    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(chunks)} chunks to: {output_path}")
    if args.summary:
        doc_types: dict[str, int] = {}
        for chunk in chunks:
            doc_type = chunk.get("doc_type_kwd", "text")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        print(f"Doc types: {json.dumps(doc_types, ensure_ascii=False, sort_keys=True)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
