from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/uah_triple")

expected = {"Attack", "Natural", "NoEvents"}
report = []

for path in sorted(DATA_DIR.glob("data*.csv")):
    df = pd.read_csv(path)

    if "marker" not in df.columns:
        report.append({
            "file": path.name,
            "status": "ERROR: marker column missing",
            "rows": len(df),
            "labels": "",
        })
        continue

    labels = set(df["marker"].dropna().astype(str).str.strip().unique())
    missing = expected - labels
    unknown = labels - expected

    status = "OK"
    if missing:
        status = f"MISSING: {sorted(missing)}"
    if unknown:
        status += f" | UNKNOWN: {sorted(unknown)}"

    report.append({
        "file": path.name,
        "status": status,
        "rows": len(df),
        "labels": ", ".join(sorted(labels)),
    })

result = pd.DataFrame(report)
print(result.to_string(index=False))

result.to_csv("results/dataset_validation_report.csv", index=False)