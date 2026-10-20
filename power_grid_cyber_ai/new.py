import json
from pathlib import Path

metadata_path = Path("models/model_metadata.json")

with open(metadata_path, encoding="utf-8") as file:
    metadata = json.load(file)

print("Removed columns:")
print(metadata["removed_columns"])

print("\nNumber of features used:")
print(metadata["features"])