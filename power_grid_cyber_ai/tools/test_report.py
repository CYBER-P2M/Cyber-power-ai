"""PASS/FAIL checks for the report feature.  From the project root:
    python tools/test_report.py [--real-llm]
The LLM is MOCKED (Qwen is never downloaded) unless --real-llm is given.
If models/ or baseline_stats.json are missing, a SYNTHETIC packet is used and the run is marked SYNTHETIC.
"""
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from src import llm_report as L  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


class MockLLM:
    model_name = "mock"

    def __init__(self, text=None, fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def __call__(self, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("forced LLM failure")
        return self.text(messages) if callable(self.text) else self.text


def good_text(packet):
    cls, other = packet["predicted_class"], L._other_class(packet["predicted_class"])
    f = packet["top_features"][0]
    rel = packet["model_reliability"]["per_class"].get("Fault", {})
    summary = (f"The model detects {cls} with probability {packet['probabilities_pct'][cls]}%. "
               f"The risk score is {packet['risk_score_pct']}%.")
    if packet["low_reliability_class"]:
        summary = (f"The model detects {cls}, a low-reliability class, with probability "
                   f"{packet['probabilities_pct'][cls]}%. The risk score is {packet['risk_score_pct']}%.")
    conf = "Evidence is uncertain." if packet["evidence_weak"] else "Evidence is moderate."
    if packet["low_reliability_class"]:
        conf += f" Historical Fault precision is low ({rel['precision_pct']}%) and recall is low ({rel['recall_pct']}%)."
    return (f"## 1. Summary\n{summary}\n\n"
            f"## 3. Interpretation\n{f['feature']} deviates by z = {f['z']} from the baseline. "
            f"The evidence does not fit {other} because its probability is only {packet['probabilities_pct'][other]}%.\n\n"
            "## 4. Possible Impact\nMeasurements may be unreliable until checked.\n\n"
            "## 5. Recommended Actions\n1. Review the listed measurements.\n2. Compare with independent data.\n3. Escalate to an engineer.\n\n"
            f"## 6. Confidence & Limitations\n{conf}\n")


def synthetic_setup(seed=0):
    rng = np.random.default_rng(seed)
    dic = L.load_feature_dictionary()
    base = {"baseline_class": "Normal", "n_rows": 1000, "features": {
        n: {"mean": float(rng.normal(50, 10)), "std": float(abs(rng.normal(2, 0.5)) + 0.1)} for n in dic}}
    for n in dic:
        if "log" in n:
            base["features"][n] = {"mean": 0.0, "std": 0.0}
    rel = {"model": "random_forest", "per_class": {   # numbers from results/classification_reports/random_forest.csv
        "Normal": {"precision": 0.8772, "recall": 0.9175, "precision_pct": 87.7, "recall_pct": 91.8, "support": 28138},
        "Fault": {"precision": 0.03397, "recall": 0.02267, "precision_pct": 3.4, "recall_pct": 2.3, "support": 3662},
        "Attack": {"precision": 0.9959, "recall": 0.9934, "precision_pct": 99.6, "recall_pct": 99.3, "support": 30020}}}
    return base, rel, dic


def make_record(base, shifted, shift_z):
    rec = {n: v["mean"] for n, v in base["features"].items()}
    for n in shifted:
        rec[n] = base["features"][n]["mean"] + shift_z * max(base["features"][n]["std"], 1e-3)
    return rec


def pred(cls, idx=0):
    p = {"Normal": 0.01, "Fault": 0.01, "Attack": 0.98} if cls != "Fault" else {"Normal": 0.04, "Fault": 0.9, "Attack": 0.06}
    if cls == "Normal":
        p = {"Normal": 0.97, "Fault": 0.01, "Attack": 0.02}
    return {"predicted_class": cls, "probabilities": p, "risk_score": 1 - p["Normal"], "index": idx, "source": "test.csv"}


def main():
    real = (ROOT / "models" / "baseline_stats.json").exists() and (ROOT / "models" / "random_forest.joblib").exists()
    synthetic = not real
    print("MODE:", "SYNTHETIC" if synthetic else "REAL (baseline + models found)")
    base, rel, dic = synthetic_setup()
    if not synthetic:
        base, rel = L.load_baseline(), L.load_reliability()
    kw = dict(baseline=base, reliability=rel, dictionary=dic, synthetic=synthetic, save=False)
    names = list(dic)
    strong, weak = make_record(base, names[0:6], 6.0), make_record(base, names[0:6], 0.8)

    m = MockLLM(good_text)
    r = L.generate_report(strong, pred("Normal"), llm=m, **kw)
    check("Normal -> no LLM call", m.calls == 0 and r["status"] == "skipped_normal")

    reports = {}
    for cls in ("Attack", "Fault"):
        pk = L.build_evidence(strong, pred(cls), base, rel, dic, synthetic)
        mm = MockLLM(lambda msgs, pk=pk: good_text(pk))
        r = L.generate_report(strong, pred(cls), llm=mm, **kw)
        reports[cls] = r
        check(f"{cls}: LLM called once, no auto-corrections", mm.calls == 1 and not r["notes"], str(r["notes"]))
        _, notes = L.verify_report(r["raw_llm_text"], r["packet"])
        check(f"{cls}: all numbers pass verify_report", "auto-corrected: unsupported number" not in notes)
    h = lambda md: re.findall(r"^## \d\. .+$", md, re.M)
    check("Attack/Fault: same 6 headings", h(reports["Attack"]["markdown"]) == h(reports["Fault"]["markdown"]) and len(h(reports["Attack"]["markdown"])) == 6)
    check("Attack/Fault: different reports", reports["Attack"]["markdown"] != reports["Fault"]["markdown"])
    check("Report ends with the disclaimer", reports["Fault"]["markdown"].strip().endswith("requires review by a qualified engineer._"))

    r = L.generate_report(strong, pred("Attack"), llm=MockLLM(fail=True), **kw)
    check("LLM failure -> template report, no crash", r["status"] == "template_fallback" and "## 6. Confidence" in r["markdown"])

    pk = L.build_evidence(strong, pred("Attack"), base, rel, dic, synthetic)
    bad = good_text(pk).replace("Measurements may be unreliable", "Measurements may drift by 7777 units")
    r = L.generate_report(strong, pred("Attack"), llm=MockLLM(bad), **kw)
    check("Unsupported number -> section replaced + note", "auto-corrected: unsupported number" in r["notes"] and "7777" not in r["markdown"])

    pkw = L.build_evidence(weak, pred("Attack"), base, rel, dic, synthetic)
    check("Weak evidence flagged (all top-5 |z| < 2)", pkw["evidence_weak"])
    r = L.generate_report(weak, pred("Attack"), llm=MockLLM(good_text(pkw).replace("Evidence is uncertain.", "Evidence is solid.")), **kw)
    check("Weak evidence -> report states weak/uncertain", re.search(r"weak|uncertain", r["markdown"], re.I) is not None and any("weak-evidence" in n for n in r["notes"]))

    pkf = L.build_evidence(strong, pred("Fault"), base, rel, dic, synthetic)
    fr = rel["per_class"]["Fault"]
    check("Fault recall < 0.5 -> low_reliability_class", pkf["low_reliability_class"] == (fr["recall"] < 0.5))
    plain = good_text(pkf).replace(", a low-reliability class,", "").replace(
        f" Historical Fault precision is low ({fr['precision_pct']}%) and recall is low ({fr['recall_pct']}%).", "")
    md = L.generate_report(strong, pred("Fault"), llm=MockLLM(plain), **kw)["markdown"]
    check("Fault: Summary says 'low-reliability class'", "low-reliability class" in md.split("## 2.")[0].lower())
    check("Fault: Confidence quotes precision & recall from packet", str(fr["precision_pct"]) in md.split("## 6.")[1] and str(fr["recall_pct"]) in md.split("## 6.")[1])

    txt = good_text(pk).replace("Measurements may be unreliable", "Check the Snort alert logs and send a trip command")
    r = L.generate_report(strong, pred("Attack"), llm=MockLLM(txt), **kw)
    check("Claims about logs/alerts/commands not in packet -> replaced", "unsupported claim" in " ".join(r["notes"]) and "trip command" not in r["markdown"])

    txt = good_text(pk).replace("Measurements may be unreliable", "R9-PM9:V looks abnormal")
    r = L.generate_report(strong, pred("Attack"), llm=MockLLM(txt), **kw)
    check("Feature name not in packet -> replaced", "R9-PM9:V" not in r["markdown"])

    print("NOTE: the Fault case uses a MOCKED prediction on a " + ("synthetic record." if synthetic else "real record (no real row predicted Fault is guaranteed)."))

    tmp = Path(tempfile.mkdtemp())
    orig = L.settings
    L.settings = lambda: {**orig(), "REPORTS_DIR": tmp, "LOG_DIR": tmp}
    r = L.generate_report(strong, pred("Attack", 3), llm=MockLLM(good_text(pk)), baseline=base, reliability=rel, dictionary=dic, synthetic=synthetic, save=True)
    check("Files saved (md + 2 charts)", Path(r["md_path"]).exists() and all(Path(c).stat().st_size > 0 for c in r["charts"]))
    L.settings = orig

    if "--real-llm" in sys.argv:
        try:
            r = L.generate_report(strong, pred("Attack"), llm=L.load_llm(), save=False, **{k: v for k, v in kw.items() if k != "save"})
            check("Real LLM produced a verified report", r["status"] == "llm", str(r["notes"]))
        except Exception as exc:
            check("Real LLM", False, f"{type(exc).__name__}: {exc}")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed" + ("  [SYNTHETIC]" if synthetic else ""))
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
