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

class RagTokenizer(infinity.rag_tokenizer.RagTokenizer):
    def __init__(self, debug=False, user_dict=None):
        super().__init__()
        self._dict_loaded = False
        self._last_mtime = 0

    def _ensure_dict_loaded(self):
        """Ensure user dictionary is loaded in this process, reload if file modified"""
        if not os.path.exists(USER_DICT_FILE):
            self._dict_loaded = True
            return

        # Check if file has been modified since last load
        current_mtime = os.path.getmtime(USER_DICT_FILE)
        if not self._dict_loaded or current_mtime > self._last_mtime:
            logger.info(f"Loading/reloading user dictionary from: {USER_DICT_FILE} (process: {os.getpid()}, mtime changed: {self._last_mtime} -> {current_mtime})")
            self.add_user_dict(USER_DICT_FILE)
            if os.getenv("USER_DICT"):
                user_dict = os.getenv("USER_DICT")
                if os.path.exists(user_dict) and os.path.getmtime(user_dict) > self._last_mtime:
                    logger.info(f"Loading user dictionary from env: {user_dict} (process: {os.getpid()})")
                    self.add_user_dict(user_dict)
            self._dict_loaded = True
            self._last_mtime = current_mtime

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
    tokenizer.add_user_dict(USER_DICT_FILE)
    logger.info(f"User dictionary loaded successfully")
else:
    logger.warning(f"User dictionary file not found: {USER_DICT_FILE}")

# Also load from environment variable if set (for backward compatibility)
if os.getenv("USER_DICT"):
    user_dict = os.getenv("USER_DICT")
    logger.info(f"Loading user dictionary from environment variable: {user_dict}")
    tokenizer.add_user_dict(user_dict)
    logger.info(f"User dictionary from environment variable loaded successfully")

tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B
