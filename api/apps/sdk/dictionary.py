#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import os
import re
import tempfile
from quart import request

from api.utils.api_utils import get_error_data_result, get_json_result, get_request_json, token_required
from rag.nlp import rag_tokenizer
from rag.nlp.rag_tokenizer import USER_DICT_FILE


# Valid part-of-speech tags are ASCII letters (jieba style, e.g. n, v, nr, ns).
_POS_RE = re.compile(r"^[A-Za-z]+$")


def _validate_dict_entry(term, frequency, pos):
    """Validate a single dictionary entry. Returns (ok, cleaned_term, cleaned_freq, cleaned_pos, reason)."""
    if not isinstance(term, str):
        return False, None, None, None, "term must be a string"
    term = term.strip()
    if not term:
        return False, None, None, None, "term is empty"
    if any(ch.isspace() for ch in term):
        return False, None, None, None, "term contains whitespace"
    if len(term) > 100:
        return False, None, None, None, "term exceeds 100 characters"
    try:
        freq = int(frequency)
    except (TypeError, ValueError):
        return False, None, None, None, f"frequency is not an integer: {frequency}"
    pos = str(pos).strip()
    if not pos or not _POS_RE.match(pos):
        return False, None, None, None, f"invalid pos tag: {pos}"
    return True, term, freq, pos, None


@manager.route("/dictionary/upload", methods=["POST"])  # noqa: F821
@token_required
async def upload_dictionary(tenant_id):
    """
    Upload a dictionary file to add professional terms.
    ---
    tags:
      - Dictionary
    security:
      - ApiKeyAuth: []
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Dictionary file (txt format, each line: term frequency pos)
    responses:
      200:
        description: Dictionary uploaded successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
    """
    try:
        # In Quart, request.files is an async property
        files = await request.files
        if 'file' not in files:
            return get_error_data_result(message="No file uploaded")

        file = files['file']
        if file.filename == '':
            return get_error_data_result(message="No file selected")

        # Read file content (synchronous read)
        content = file.read()
        content = content.decode('utf-8')

        # Validate and filter lines: problematic entries are skipped
        valid_lines = []
        skipped = 0
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                skipped += 1
                continue
            ok, term, freq, pos, _ = _validate_dict_entry(parts[0], parts[1], parts[2])
            if not ok:
                skipped += 1
                continue
            valid_lines.append(f"{term} {freq} {pos}\n")

        if not valid_lines:
            return get_error_data_result(message="No valid dictionary entries found in uploaded file")

        # Ensure directory exists
        os.makedirs(os.path.dirname(USER_DICT_FILE), exist_ok=True)

        # Append to persistent user dictionary file
        logging.info(f"Writing {len(valid_lines)} entries to user dictionary file: {USER_DICT_FILE}")
        with open(USER_DICT_FILE, 'a', encoding='utf-8') as f:  # noqa: ASYNC230
            f.writelines(valid_lines)
        logging.info("Successfully wrote to user dictionary file")

        # Also load into current tokenizer
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.writelines(valid_lines)
            temp_file = f.name

        rag_tokenizer.tokenizer.add_user_dict(temp_file)
        # Mark that dictionary needs to be reloaded in all calls
        rag_tokenizer.tokenizer._dict_loaded = False

        os.unlink(temp_file)

        return get_json_result(
            data={
                "success": True,
                "message": f"Dictionary uploaded: {len(valid_lines)} entries added, {skipped} entries skipped",
                "added_count": len(valid_lines),
                "skipped_count": skipped,
            }
        )
    except Exception as e:
        logging.exception("Failed to upload dictionary")
        return get_error_data_result(message=f"Failed to upload dictionary: {str(e)}")


@manager.route("/dictionary/add_term", methods=["POST"])  # noqa: F821
@token_required
async def add_term(tenant_id):
    """
    Add a single term to the dictionary.
    ---
    tags:
      - Dictionary
    security:
      - ApiKeyAuth: []
    parameters:
      - name: term
        in: body
        type: string
        required: true
        description: The term to add
      - name: frequency
        in: body
        type: number
        required: false
        description: The frequency of the term (default: 1000000)
      - name: pos
        in: body
        type: string
        required: false
        description: The part of speech (default: n)
    responses:
      200:
        description: Term added successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
    """
    try:
        req = await get_request_json()
        term = req.get("term")
        frequency = req.get("frequency", 3)
        pos = req.get("pos", "n")

        ok, term, freq, pos, reason = _validate_dict_entry(term, frequency, pos)
        if not ok:
            return get_error_data_result(message=f"Invalid term: {reason}")

        # Append to persistent user dictionary file
        with open(USER_DICT_FILE, 'a', encoding='utf-8') as f:  # noqa: ASYNC230
            f.write(f"{term} {freq} {pos}\n")

        # Also load into current tokenizer
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(f"{term} {freq} {pos}\n")
            temp_file = f.name

        rag_tokenizer.tokenizer.add_user_dict(temp_file)
        # Mark that dictionary needs to be reloaded in all calls
        rag_tokenizer.tokenizer._dict_loaded = False

        os.unlink(temp_file)

        return get_json_result(data={"success": True, "message": f"Term '{term}' added successfully"})
    except Exception as e:
        logging.exception("Failed to add term")
        return get_error_data_result(message=f"Failed to add term: {str(e)}")


@manager.route("/dictionary/batch_add_terms", methods=["POST"])  # noqa: F821
@token_required
async def batch_add_terms(tenant_id):
    """
    Add multiple terms to the dictionary in batch.
    ---
    tags:
      - Dictionary
    security:
      - ApiKeyAuth: []
    parameters:
      - name: terms
        in: body
        type: array
        required: true
        description: List of terms to add
        items:
          type: object
          properties:
            term:
              type: string
              required: true
              description: The term to add
            frequency:
              type: number
              required: false
              description: The frequency of the term (default: 3)
            pos:
              type: string
              required: false
              description: The part of speech (default: n)
    responses:
      200:
        description: Terms added successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
            added_count:
              type: integer
    """
    try:
        req = await get_request_json()
        terms = req.get("terms")

        if not terms or not isinstance(terms, list):
            return get_error_data_result(message="Terms list is required")

        if len(terms) == 0:
            return get_error_data_result(message="Terms list cannot be empty")

        # Limit batch size to prevent performance issues
        if len(terms) > 1000:
            return get_error_data_result(message="Batch size exceeds limit (maximum 1000 terms)")

        # Validate and collect batch entries; problematic items are skipped
        added_count = 0
        skipped_count = 0
        batch_terms = []
        for item in terms:
            term = item.get("term")
            frequency = item.get("frequency", 3)
            pos = item.get("pos", "n")
            ok, term, freq, pos, _ = _validate_dict_entry(term, frequency, pos)
            if not ok:
                skipped_count += 1
                continue
            term_line = f"{term} {freq} {pos}\n"
            batch_terms.append(term_line)
            added_count += 1

        if added_count == 0:
            return get_error_data_result(message="No valid terms to add")

        # Create temporary file for valid batch terms
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.writelines(batch_terms)
            temp_file = f.name

        # Append to persistent user dictionary file
        with open(USER_DICT_FILE, 'a', encoding='utf-8') as f:  # noqa: ASYNC230
            f.writelines(batch_terms)

        # Add all terms at once to current tokenizer
        rag_tokenizer.tokenizer.add_user_dict(temp_file)
        # Mark that dictionary needs to be reloaded in all calls
        rag_tokenizer.tokenizer._dict_loaded = False

        os.unlink(temp_file)

        return get_json_result(
            data={
                "success": True,
                "message": f"Successfully added {added_count} terms, skipped {skipped_count} invalid terms",
                "added_count": added_count,
                "skipped_count": skipped_count,
            }
        )
    except Exception as e:
        logging.exception("Failed to add terms in batch")
        return get_error_data_result(message=f"Failed to add terms: {str(e)}")


@manager.route("/dictionary/test", methods=["POST"])  # noqa: F821
@token_required
async def test_tokenization(tenant_id):
    """
    Test tokenization with the current dictionary.
    ---
    tags:
      - Dictionary
    security:
      - ApiKeyAuth: []
    parameters:
      - name: text
        in: body
        type: string
        required: true
        description: Text to tokenize
    responses:
      200:
        description: Tokenization result
        schema:
          type: object
          properties:
            original:
              type: string
            tokenized:
              type: string
    """
    try:
        req = await get_request_json()
        text = req.get("text")

        if not text:
            return get_error_data_result(message="Text is required")

        tokenized = rag_tokenizer.tokenize(text)

        return get_json_result(data={"original": text, "tokenized": tokenized})
    except Exception as e:
        logging.exception("Failed to test tokenization")
        return get_error_data_result(message=f"Failed to test tokenization: {str(e)}")


@manager.route("/dictionary/status", methods=["GET"])  # noqa: F821
@token_required
def get_dictionary_status(tenant_id):
    """
    Get dictionary status information.
    ---
    tags:
      - Dictionary
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: Dictionary status
        schema:
          type: object
          properties:
            status:
              type: string
            info:
              type: object
    """
    try:
        info = {
            "tokenizer_type": type(rag_tokenizer.tokenizer).__name__,
            "has_user_dict": os.path.exists(USER_DICT_FILE) and os.path.getsize(USER_DICT_FILE) > 0,
            "user_dict_file": USER_DICT_FILE,
            "user_dict_file_exists": os.path.exists(USER_DICT_FILE),
            "user_dict_file_size": os.path.getsize(USER_DICT_FILE) if os.path.exists(USER_DICT_FILE) else 0,
            "env_user_dict": os.getenv("USER_DICT")
        }

        return get_json_result(data={"status": "active", "info": info})
    except Exception as e:
        logging.exception("Failed to get dictionary status")
        return get_error_data_result(message=f"Failed to get dictionary status: {str(e)}")
