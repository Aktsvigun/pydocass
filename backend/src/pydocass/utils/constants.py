import os

DEFAULT_MODEL_CHECKPOINT = "Qwen/Qwen2.5-Coder-32B-Instruct-fast"
LONG_CONTEXT_MODEL_CHECKPOINT = "meta-llama/Llama-3.3-70B-Instruct-fast"
DEFAULT_TOKENIZER_CHECKPOINT = "Qwen/Qwen2.5-Coder-32B-Instruct"

MAX_TOTAL_TOKENS = 31_000
MAX_TOKENS_WITH_LONG_CONTEXT = 8_092

NUM_SYSTEM_PROMPT_TOKENS_DICT = {
    "annotations": 12756,
    "docstrings": 13753,
    "docstrings_addition": 6103,
    "comments": 14108,
}

DEFAULT_MAX_TOKENS_DICT = {"annotations": 2048, "docstrings": 4096, "comments": 2048}
MAX_MAX_TOKENS_DICT = {"annotations": 2048, "docstrings": 4096, "comments": 2048}

DEFAULT_TOP_P_ANNOTATIONS = 0.5
DEFAULT_TOP_P_DOCSTRINGS = 0.01
DEFAULT_TOP_P_COMMENTS = 0.01

BASE_URL = (
    os.getenv("NEBIUS_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://api.studio.nebius.ai/v1"
)
