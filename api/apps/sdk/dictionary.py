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
import tempfile
from quart import request

from api.utils.api_utils import get_error_data_result, get_json_result, get_request_json, token_required
from rag.nlp import rag_tokenizer
from rag.nlp.rag_tokenizer import USER_DICT_FILE


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

        # Ensure directory exists
        os.makedirs(os.path.dirname(USER_DICT_FILE), exist_ok=True)
        
        # Append to persistent user dictionary file
        logging.info(f"Writing to user dictionary file: {USER_DICT_FILE}")
        with open(USER_DICT_FILE, 'a', encoding='utf-8') as f:
            f.write(content)
        logging.info(f"Successfully wrote to user dictionary file")

        # Also load into current tokenizer
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(content)
            temp_file = f.name

        rag_tokenizer.tokenizer.add_user_dict(temp_file)
        # Mark that dictionary needs to be reloaded in all calls
        rag_tokenizer.tokenizer._dict_loaded = False

        os.unlink(temp_file)

        return get_json_result(data={"success": True, "message": "Dictionary uploaded and loaded successfully"})
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

        if not term:
            return get_error_data_result(message="Term is required")

        # Append to persistent user dictionary file
        with open(USER_DICT_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{term} {frequency} {pos}\n")

        # Also load into current tokenizer
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(f"{term} {frequency} {pos}\n")
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

        # Create temporary file for batch terms and append to persistent file
        added_count = 0
        batch_terms = []
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            for item in terms:
                term = item.get("term")
                if not term:
                    continue
                # Validate term length
                if len(term) > 100:
                    continue
                frequency = item.get("frequency", 3)
                pos = item.get("pos", "n")
                term_line = f"{term} {frequency} {pos}\n"
                f.write(term_line)
                batch_terms.append(term_line)
                added_count += 1
            temp_file = f.name

        if added_count == 0:
            os.unlink(temp_file)
            return get_error_data_result(message="No valid terms to add")

        # Append to persistent user dictionary file
        with open(USER_DICT_FILE, 'a', encoding='utf-8') as f:
            f.writelines(batch_terms)

        # Add all terms at once to current tokenizer
        rag_tokenizer.tokenizer.add_user_dict(temp_file)
        # Mark that dictionary needs to be reloaded in all calls
        rag_tokenizer.tokenizer._dict_loaded = False

        os.unlink(temp_file)

        return get_json_result(data={"success": True, "message": f"Successfully added {added_count} terms", "added_count": added_count})
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
