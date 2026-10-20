from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/uah_triple")
OUT_DIR = Path("data/selected")
OUT_DIR.mkdir(parents=True, exist_ok=True)

label_map = {
    "NoEvents": "Normal",
    "Attack": "Attack",
    "Natural": "Fault",
}

frames = []

for path in sorted(RAW_DIR.glob("data*.csv")):
    df = pd.read_csv(path)

    df["source_file"] = path.name
    df["label"] = df["marker"].map(label_map)

    frames.append(df)

final_df = pd.concat(frames, ignore_index=True)

if final_df["label"].isna().any():
    bad = final_df.loc[
        final_df["label"].isna(),
        "marker"
    ].value_counts(dropna=False)

    raise ValueError(f"Unknown labels found:\n{bad}")

expected = {"Normal", "Attack", "Fault"}
found = set(final_df["label"].unique())

if found != expected:
    raise ValueError(
        f"Expected {expected}, but found {found}"
    )

final_df.to_csv(
    OUT_DIR / "final_verified_multiclass.csv",
    index=False,
)

print("Dataset prepared successfully.")
print(f"Total rows: {len(final_df):,}")
print("\nFinal class distribution:")
print(final_df["label"].value_counts())
print(f"\nSaved to: {OUT_DIR / 'final_verified_multiclass.csv'}")