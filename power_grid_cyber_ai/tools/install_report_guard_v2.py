"""Install the conservative report guard. Run from project root.
python tools/install_report_guard_v2.py --check
python tools/install_report_guard_v2.py --apply
"""
from pathlib import Path
import argparse
import ast
import shutil
import hashlib

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "llm_report.py"
PATCH = ROOT / "tools" / "report_guard_v2.block.txt"


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not TARGET.is_file():
        raise SystemExit(f"Missing file: {TARGET}")
    source = TARGET.read_text(encoding="utf-8-sig")
    ast.parse(source)
    required = {"template_sections", "verify_report", "generate_report"}
    actual = {n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    if not required.issubset(actual):
        raise SystemExit("Incompatible llm_report.py: required functions missing. No change made.")
    if "# REPORT_GUARD_V2_BEGIN" in source:
        print("Guard already installed. No change made.")
        return
    block = PATCH.read_text(encoding="utf-8")
    combined = source.rstrip() + "\n\n" + block
    ast.parse(combined)
    print("Target:", TARGET)
    print("Compatible function names and valid syntax. Runtime integration still needs testing.")
    if args.check:
        print("CHECK ONLY: no files changed.")
        return
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()[:12]
    backup = TARGET.with_name(f"llm_report.py.before_guard_v2_{digest}.bak")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
    temp = TARGET.with_name("llm_report.py.guard_v2.tmp")
    temp.write_text(combined, encoding="utf-8")
    temp.replace(TARGET)
    print("Installed. Backup:", backup)
    print("Restart Streamlit and create a fresh report (not a cached report).")


if __name__ == "__main__":
    main()
