"""Unit tests for the guard; optional --installed checks the project's module.
Uses explicit mock v1 functions by default; does not load a classifier or LLM.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def run():
    if "--installed" in sys.argv:
        sys.path.insert(0, str(ROOT))
        from src import llm_report as mod
        if not hasattr(mod, "_guard_v1_verify_report"):
            raise SystemExit("FAIL: guard not installed")
        verify = mod.verify_report
        template = mod.template_sections
        print("MODE: installed report functions; mock packet; no classifier or LLM call")
    else:
        env = {
            "template_sections": lambda packet: {},
            "verify_report": lambda text, packet: ({}, ["mock-v1-validator"]),
        }
        exec((ROOT/"tools/report_guard_v2.block.txt").read_text(encoding="utf-8"), env)
        verify, template = env["verify_report"], env["template_sections"]
        print("MODE: isolated guard with mocked v1 verifier; synthetic packet")
    packet = {
        "record_id":"test.csv#1", "predicted_class":"Attack",
        "probabilities_pct":{"Normal":0.,"Attack":87.,"Fault":13.},
        "risk_score_pct":100., "evidence_weak":False,
        "weak_z_threshold":2., "low_recall_threshold_pct":50.,
        "low_reliability_class":False, "baseline_class":"Normal", "synthetic":True,
        "top_features":[{"feature":"R1-PM1:V","meaning":"Voltage magnitude","kind":"voltage_magnitude",
            "value":52.,"baseline_mean":40.,"baseline_std":4.,"deviation":12.,"z":3.,"z_capped":False}],
        "model_reliability":{"model":"mock", "per_class":{
            "Attack":{"precision":.99,"recall":.98,"precision_pct":99.,"recall_pct":98.,"support":100},
            "Fault":{"precision":.034,"recall":.023,"precision_pct":3.4,"recall_pct":2.3,"support":100}}},
    }
    bad = "## 1. Summary\nAttack 9999%.\n## 3. Interpretation\nFault is ruled out.\n## 4. Possible Impact\nEquipment damage is certain.\n## 5. Recommended Actions\n1. Send a trip command.\n## 6. Confidence & Limitations\nHighly confident due to small dataset size."
    sec, notes = verify(bad, packet)
    checks = {
        "unsupported certainty removed": "highly confident" not in " ".join(sec.values()).lower(),
        "small dataset speculation removed": "small dataset" not in " ".join(sec.values()).lower(),
        "unsupported number removed": "9999" not in " ".join(sec.values()),
        "operational command removed": "trip command" not in " ".join(sec.values()),
        "alternative class not ruled out": "not independent evidence" in sec["Interpretation"],
        "non-normal meaning explained": "not a calibrated probability" in sec["Summary"],
        "historical metrics displayed": "99%" in sec["Confidence & Limitations"] and "98%" in sec["Confidence & Limitations"],
        "final provenance note": any("guard-v2" in n for n in notes),
    }
    packet["evidence_weak"] = True
    sec, _ = verify(bad, packet)
    checks["weak evidence stated"] = "weak/uncertain" in sec["Confidence & Limitations"]
    packet["predicted_class"]="Fault"
    packet["low_reliability_class"]=True
    sec, _ = verify(bad, packet)
    checks["Fault low reliability in summary"] = "low-reliability class" in sec["Summary"]
    checks["Fault precision/recall from packet"] = "3.4%" in sec["Confidence & Limitations"] and "2.3%" in sec["Confidence & Limitations"]
    packet["top_features"]=[]
    packet["evidence_weak"]=False
    checks["empty evidence is uncertain"] = "weak/uncertain" in template(packet)["Confidence & Limitations"]
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{sum(checks.values())}/{len(checks)} checks passed")
    if not all(checks.values()):
        raise SystemExit(1)

if __name__ == "__main__":
    run()
