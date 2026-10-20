from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

REPORTS_DIR = RESULTS_DIR / "reports"
LOG_DIR = PROJECT_ROOT / "logs"


# --------------------------------------------------
# Training
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------
# Class names
# --------------------------------------------------

CLASS_NORMAL = "Normal"
CLASS_FAULT = "Fault"
CLASS_ATTACK = "Attack"


# --------------------------------------------------
# Report mode
# --------------------------------------------------

# "grounded_template":
#     Build the report directly from evidence without invoking an LLM.
#
# "llm_draft_guarded":
#     Generate an LLM draft, but keep the final narrative grounded
#     in deterministic templates under the current Guard V2 policy.

REPORT_MODE = "llm_draft_guarded"

# The current grounded narrative is implemented in English.
REPORT_LANG = "en"


# --------------------------------------------------
# Language model
# --------------------------------------------------

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

LLM_FALLBACKS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]

LLM_MAX_NEW_TOKENS = 450

# Allow retrieving missing model files.
LLM_ALLOW_DOWNLOAD = True

# Wait before retrying after a model-loading failure.
LLM_RETRY_SECONDS = 120

# None preserves the library's existing/default cache configuration.
# To use a project-local cache instead, replace None with:
# PROJECT_ROOT / "models" / "llm_cache"
LLM_CACHE_DIR = None


# --------------------------------------------------
# Evidence sources
# --------------------------------------------------

BASELINE_PATH = MODELS_DIR / "baseline_stats.json"

FEATURE_DICT_PATH = (
    PROJECT_ROOT
    / "src"
    / "feature_dictionary.json"
)

REPORT_RELIABILITY_DIR = (
    RESULTS_DIR / "classification_reports"
)

# RF-specific path kept for backward compatibility.
# llm_report.py selects XGBoost and CatBoost files separately.
RELIABILITY_CSV = (
    REPORT_RELIABILITY_DIR / "random_forest.csv"
)


# --------------------------------------------------
# Evidence and report thresholds
# --------------------------------------------------

# Maximum number of features included in the evidence packet.
REPORT_PACKET_FEATURES = 20

# Maximum number of features displayed in the report table.
REPORT_TABLE_ROWS = 10

# Evidence is considered weak when all examined top deviations
# are below this absolute z-score threshold.
WEAK_Z_THRESHOLD = 2.0

# Historical Fault recall below this threshold triggers
# the low-reliability warning.
LOW_RECALL_THRESHOLD = 0.50

# Limit extreme displayed z-scores.
Z_CAP = 50.0