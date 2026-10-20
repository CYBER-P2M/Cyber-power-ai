
# ======================================================================
# ADD THESE LINES TO THE END OF src/config.py  (LLM report feature)
# Check first that none of these names already exist in your config.py.
# ======================================================================
from pathlib import Path as _Path
_REPORT_ROOT = _Path(__file__).resolve().parents[1]

CLASS_NORMAL = "Normal"
CLASS_FAULT = "Fault"
CLASS_ATTACK = "Attack"

REPORT_LANG = "en"                       # "ar" -> English report + warning
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_FALLBACKS = ["Qwen/Qwen2.5-0.5B-Instruct"]
LLM_MAX_NEW_TOKENS = 450

REPORTS_DIR = _REPORT_ROOT / "results" / "reports"
LOG_DIR = _REPORT_ROOT / "logs"
BASELINE_PATH = _REPORT_ROOT / "models" / "baseline_stats.json"
FEATURE_DICT_PATH = _REPORT_ROOT / "src" / "feature_dictionary.json"
RELIABILITY_CSV = _REPORT_ROOT / "results" / "classification_reports" / "random_forest.csv"

REPORT_PACKET_FEATURES = 20
REPORT_TABLE_ROWS = 10
WEAK_Z_THRESHOLD = 2.0
LOW_RECALL_THRESHOLD = 0.5
Z_CAP = 50.0
