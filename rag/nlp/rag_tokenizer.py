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

import infinity.rag_tokenizer
import os
import logging
import re
import tempfile
from typing import Set

# Valid part-of-speech tags are ASCII letters (jieba style, e.g. n, v, nr, ns).
_USER_DICT_POS_RE = re.compile(r"^[A-Za-z]+$")

# Configure logger
logger = logging.getLogger(__name__)

# User dictionary file path for persistence
# Read from environment variable, default to root directory
user_dict_file = os.getenv("USER_DICT_FILE")
if user_dict_file is None:
    # Default to docker directory for persistence
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    docker_dir = os.path.join(root_dir, "docker")
    user_dict_file = os.path.join(docker_dir, "user_dict.txt")
USER_DICT_FILE = user_dict_file


def _is_valid_user_dict_line(line):
    """Return True if the line conforms to 'word freq pos' format."""
    parts = line.split()
    if len(parts) < 3:
        return False
    term, freq, pos = parts[0], parts[1], parts[2]
    if not term or any(ch.isspace() for ch in term) or len(term) > 100:
        return False
    try:
        int(freq)
    except (TypeError, ValueError):
        return False
    if not _USER_DICT_POS_RE.match(pos):
        return False
    return True


def _sanitize_user_dict_file(src_path):
    """Write only valid entries from src_path to a temporary file and return its path."""
    fd, dst_path = tempfile.mkstemp(suffix=".txt", text=True)
    skipped = 0
    kept = 0
    with os.fdopen(fd, "w", encoding="utf-8") as out, open(src_path, "r", encoding="utf-8") as src:
        for raw in src:
            line = raw.strip()
            if not line:
                continue
            if _is_valid_user_dict_line(line):
                out.write(f"{line.split()[0]} {int(line.split()[1])} {line.split()[2]}\n")
                kept += 1
            else:
                skipped += 1
    if skipped:
        logger.warning(f"Skipped {skipped} invalid entries while loading user dict: {src_path}")
    logger.info(f"Sanitized user dict {src_path}: kept {kept} entries")
    return dst_path


class RagTokenizer(infinity.rag_tokenizer.RagTokenizer):
    def __init__(self, debug=False, user_dict=None):
        super().__init__()
        self._dict_loaded = False
        self._last_mtime = 0
        self._user_terms: Set[str] = set()

    def _ensure_dict_loaded(self):
        """Ensure user dictionary is loaded in this process, reload if file modified"""
        if not os.path.exists(USER_DICT_FILE):
            self._dict_loaded = True
            return

        # Check if file has been modified since last load
        current_mtime = os.path.getmtime(USER_DICT_FILE)
        if not self._dict_loaded or current_mtime > self._last_mtime:
            logger.info(f"Loading/reloading user dictionary from: {USER_DICT_FILE} (process: {os.getpid()}, mtime changed: {self._last_mtime} -> {current_mtime})")
            sanitized = _sanitize_user_dict_file(USER_DICT_FILE)
            try:
                self.add_user_dict(sanitized)
            finally:
                os.unlink(sanitized)
            if os.getenv("USER_DICT"):
                user_dict = os.getenv("USER_DICT")
                if os.path.exists(user_dict) and os.path.getmtime(user_dict) > self._last_mtime:
                    logger.info(f"Loading user dictionary from env: {user_dict} (process: {os.getpid()})")
                    sanitized_env = _sanitize_user_dict_file(user_dict)
                    try:
                        self.add_user_dict(sanitized_env)
                    finally:
                        os.unlink(sanitized_env)
            self._update_user_terms_cache()
            self._dict_loaded = True
            self._last_mtime = current_mtime

    def _update_user_terms_cache(self):
        """Update the cached set of user terms from dictionary file"""
        self._user_terms.clear()
        # Read from USER_DICT_FILE
        if os.path.exists(USER_DICT_FILE):
            with open(USER_DICT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or not _is_valid_user_dict_line(line):
                        continue
                    parts = line.split()
                    self._user_terms.add(parts[0])
        # Read from USER_DICT environment variable if set
        env_path = os.getenv("USER_DICT")
        if env_path and os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or not _is_valid_user_dict_line(line):
                        continue
                    parts = line.split()
                    self._user_terms.add(parts[0])

    def is_in_user_dict(self, term: str) -> bool:
        """Check if a term exists in the user dictionary"""
        self._ensure_dict_loaded()
        return term in self._user_terms

    def tokenize(self, line: str) -> str:
        from common import settings # moved from the top of the file to avoid circular import
        self._ensure_dict_loaded()
        if settings.DOC_ENGINE_INFINITY:
            return line
        else:
            return super().tokenize(line)

    def fine_grained_tokenize(self, tks: str) -> str:
        from common import settings # moved from the top of the file to avoid circular import
        self._ensure_dict_loaded()
        if settings.DOC_ENGINE_INFINITY:
            return tks
        else:
            return super().fine_grained_tokenize(tks)


def is_chinese(s):
    return infinity.rag_tokenizer.is_chinese(s)


def is_number(s):
    return infinity.rag_tokenizer.is_number(s)


def is_alphabet(s):
    return infinity.rag_tokenizer.is_alphabet(s)


def naive_qie(txt):
    return infinity.rag_tokenizer.naive_qie(txt)


tokenizer = RagTokenizer()

# Load user dictionary from persistent file if it exists
if os.path.exists(USER_DICT_FILE):
    logger.info(f"Loading user dictionary from: {USER_DICT_FILE}")
    sanitized = _sanitize_user_dict_file(USER_DICT_FILE)
    try:
        tokenizer.add_user_dict(sanitized)
        logger.info("User dictionary loaded successfully")
    finally:
        os.unlink(sanitized)
else:
    logger.warning(f"User dictionary file not found: {USER_DICT_FILE}")

# Also load from environment variable if set (for backward compatibility)
env_path = os.getenv("USER_DICT")
if env_path and os.path.exists(env_path):
    logger.info(f"Loading user dictionary from environment variable: {env_path}")
    sanitized_env = _sanitize_user_dict_file(env_path)
    try:
        tokenizer.add_user_dict(sanitized_env)
        logger.info("User dictionary from environment variable loaded successfully")
    finally:
        os.unlink(sanitized_env)

tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B
